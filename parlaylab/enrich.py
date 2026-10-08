"""Match each leg to its real game right after a slip is read.

Uses the free ESPN schedule (no AI, no credits). For each leg it:
  * finds the game (team + sport, today and the next week, or the slip's date)
  * fills in the opponent and game date if the slip didn't show them
  * adds a "game" line you can check at a glance, e.g. "BOS @ NY · Sun Oct 11, 1:00 PM"
Legs it can't match are flagged so you can fix the team or date before saving.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from . import scores
from .grader import _sides

NOT_FOUND = "❓ No game found: check the team, sport or date"


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(os.getenv("PARLAYLAB_TZ", "America/New_York"))
    except Exception:
        return None


def _local(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    tz = _tz()
    return dt.astimezone(tz) if tz and dt.tzinfo else dt


def game_label(game: dict) -> str:
    away = game["away"]["team"].get("abbreviation") or game["away"]["team"].get("shortDisplayName", "?")
    home = game["home"]["team"].get("abbreviation") or game["home"]["team"].get("shortDisplayName", "?")
    matchup = f"{away} @ {home}"
    if game.get("state") in ("in", "post"):
        score = f"{away} {game['away']['score']:g}-{game['home']['score']:g} {home}"
        return f"{score} · {game.get('detail') or game['state']}"
    when = _local(game.get("date"))
    return f"{matchup} · {when.strftime('%a %b %-d, %-I:%M %p')}" if when else matchup


def enrich_leg(leg: dict, find=None) -> dict:
    """Return a copy of the leg with its game matched and blanks filled in."""
    find = find or scores.find_game
    out = dict(leg)
    if not (out.get("team") or out.get("opponent")):
        out["game"] = "❓ No team on this leg: add the player's team so the game can be found"
        return out
    try:
        game = find(out)
    except Exception:
        game = None
    if not game:
        out["game"] = NOT_FOUND
        return out
    out["game"] = game_label(game)
    sides = _sides(out, game)
    if sides and not out.get("opponent"):
        out["opponent"] = sides[1]["team"].get("displayName")
    if sides and out.get("team") and out.get("type") != "total":
        out["team"] = sides[0]["team"].get("displayName") or out["team"]   # "NYK" → "New York Knicks"
    when = _local(game.get("date"))
    if when and not out.get("game_date"):
        out["game_date"] = when.strftime("%Y-%m-%d")
    return out


def enrich_slips(slips: list[dict], find=None, workers: int = 6) -> list[dict]:
    """Match every leg of every slip, in parallel. Slips are changed in place and returned."""
    jobs = [(si, li) for si, s in enumerate(slips) for li in range(len(s.get("legs") or []))]

    def run(job):
        si, li = job
        return si, li, enrich_leg(slips[si]["legs"][li], find)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for si, li, leg in pool.map(run, jobs):
            slips[si]["legs"][li] = leg
    return slips


def unmatched(slip: dict) -> int:
    return sum(1 for l in slip.get("legs") or [] if str(l.get("game", "")).startswith("❓"))
