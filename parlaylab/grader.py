"""Grade parlay legs against live games and box scores.

Pure functions (`grade_game_leg`, `grade_prop_leg`, `parlay_status`) are
separated from network calls so they can be unit-tested.

Leg results:
  won / lost / push        settled
  winning / losing         game in progress
  pending                  game hasn't started
  not_found                couldn't match the game or player
"""
from __future__ import annotations

from typing import Callable

from . import scores
from .learned import stat_lookups
from .scores import player_matches, team_matches
from .stats_catalog import PROP_STATS

SETTLED = {"won", "lost", "push"}


def _fmt(x: float) -> str:
    return f"{x:g}"


def _sides(leg: dict, game: dict) -> tuple[dict, dict] | None:
    """Return (bet team side, other side)."""
    home, away = game["home"], game["away"]
    if team_matches(leg.get("team"), home["team"]):
        return home, away
    if team_matches(leg.get("team"), away["team"]):
        return away, home
    if leg.get("opponent"):
        if team_matches(leg["opponent"], home["team"]):
            return away, home
        if team_matches(leg["opponent"], away["team"]):
            return home, away
    return None


def grade_game_leg(leg: dict, game: dict) -> dict:
    state = game["state"]
    score_line = (
        f"{game['away']['team'].get('abbreviation', 'AWY')} {_fmt(game['away']['score'])} @ "
        f"{game['home']['team'].get('abbreviation', 'HME')} {_fmt(game['home']['score'])}"
    )
    out = {"game": game.get("name"), "state": state, "clock": game.get("detail"), "score": score_line}
    if state == "pre":
        return {**out, "result": "pending", "detail": game.get("detail") or "Not started"}

    btype = leg.get("type")
    line = leg.get("line")

    if btype == "total":
        total = game["home"]["score"] + game["away"]["score"]
        side = (leg.get("side") or "over").lower()
        if line is None:
            return {**out, "result": "not_found", "detail": "Total has no line"}
        margin = total - line if side == "over" else line - total
        # Overs can cash early; unders can die early.
        if state == "in":
            if side == "over" and margin > 0:
                return {**out, "result": "won", "current": total, "detail": f"Total {_fmt(total)} already over {_fmt(line)}"}
            if side == "under" and margin < 0:
                return {**out, "result": "lost", "current": total, "detail": f"Total {_fmt(total)} already over {_fmt(line)}"}
            need = line - total
            return {**out, "result": "winning" if side == "under" else "losing", "current": total,
                    "detail": f"Total {_fmt(total)} / {_fmt(line)} ({_fmt(need)} to go)"}
        result = "push" if margin == 0 else ("won" if margin > 0 else "lost")
        return {**out, "result": result, "current": total, "detail": f"Final total {_fmt(total)} vs {_fmt(line)}"}

    sides = _sides(leg, game)
    if not sides:
        return {**out, "result": "not_found", "detail": f"Couldn't match '{leg.get('team')}' in {game.get('name')}"}
    mine, theirs = sides
    diff = mine["score"] - theirs["score"]
    adj = diff + (line or 0) if btype == "spread" else diff
    if adj > 0:
        result = "won" if state == "post" else "winning"
    elif adj < 0:
        result = "lost" if state == "post" else "losing"
    else:
        result = "push" if state == "post" else "losing"
    label = f"{mine['team'].get('abbreviation', '')} {'+' if diff >= 0 else ''}{_fmt(diff)}"
    if btype == "spread":
        label += f" (needs {'+' if (line or 0) > 0 else ''}{_fmt(line or 0)})"
    return {**out, "result": result, "current": diff, "detail": label}


def stat_value(player: dict, lookups: list[tuple[str | None, str]]) -> float | None:
    total, found = 0.0, False
    for group, label in lookups:
        for gname, stats in player["groups"].items():
            if group and gname != group:
                continue
            if label in stats:
                raw = str(stats[label]).split("-")[0].strip()  # "3-8" -> made
                try:
                    total += float(raw)
                    found = True
                except ValueError:
                    pass
                break
    return total if found else None


def grade_prop_leg(leg: dict, game: dict, players: list[dict]) -> dict:
    state = game["state"]
    out = {"game": game.get("name"), "state": state, "clock": game.get("detail")}
    if state == "pre":
        return {**out, "result": "pending", "detail": game.get("detail") or "Not started"}

    sport = (leg.get("sport") or "").upper()
    lookups = (stat_lookups(sport).get(leg.get("stat") or "")
               or PROP_STATS.get(sport, {}).get(leg.get("stat") or ""))
    if not lookups:
        return {**out, "result": "not_found", "detail": f"Unknown stat '{leg.get('stat')}' for {sport}"}

    player = next((p for p in players if player_matches(leg.get("player"), p["name"])), None)
    if not player:
        msg = "not in box score yet" if state == "in" else "not in box score (DNP props are usually voided)"
        return {**out, "result": "not_found", "detail": f"{leg.get('player')} {msg}"}

    value = stat_value(player, lookups) or 0.0
    line = leg.get("line") if leg.get("line") is not None else 0.5
    side = (leg.get("side") or "over").lower()
    label = f"{player['name']}: {_fmt(value)} {leg.get('stat')} (line {_fmt(line)})"

    if side == "over":
        if value > line:
            return {**out, "result": "won", "current": value, "detail": label}
        if state == "post":
            return {**out, "result": "push" if value == line else "lost", "current": value, "detail": label}
        return {**out, "result": "losing", "current": value, "detail": f"{label}, needs {_fmt(line - value)} more"}
    # under
    if value > line:
        return {**out, "result": "lost", "current": value, "detail": label}
    if state == "post":
        return {**out, "result": "push" if value == line else "won", "current": value, "detail": label}
    return {**out, "result": "winning", "current": value, "detail": label}


def parlay_status(results: list[str]) -> str:
    if any(r == "lost" for r in results):
        return "lost"
    if results and all(r in SETTLED for r in results):
        return "push" if all(r == "push" for r in results) else "won"
    if any(r in ("winning", "losing") or r in SETTLED for r in results):
        return "live"
    return "pending"


def grade_leg(leg: dict, find_game: Callable = scores.find_game, get_box: Callable = scores.boxscore) -> dict:
    try:
        game = find_game(leg)
    except Exception as e:  # network trouble shouldn't crash the tracker
        return {"result": "not_found", "detail": f"Score feed error: {e}"}
    if not game:
        return {"result": "not_found", "detail": f"No {leg.get('sport')} game found for {leg.get('team')}"}
    if leg.get("type") == "player_prop":
        if game["state"] == "pre":
            return grade_prop_leg(leg, game, [])
        try:
            players = get_box(leg["sport"].upper(), game["id"])
        except Exception as e:
            return {"result": "not_found", "detail": f"Box score error: {e}", "game": game.get("name")}
        return grade_prop_leg(leg, game, players)
    return grade_game_leg(leg, game)


def grade_parlay(legs: list[dict]) -> tuple[list[dict], str]:
    graded = [grade_leg(leg) for leg in legs]
    return graded, parlay_status([g["result"] for g in graded])
