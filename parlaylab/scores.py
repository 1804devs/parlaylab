"""Live scores and box scores from ESPN's public JSON feeds.

No API key needed. These are the same feeds ESPN's site uses; they are
unofficial, so the parsers below are defensive.
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import date, datetime, timedelta

import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports"
PATHS = {
    "NFL": "football/nfl",
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
    "NHL": "hockey/nhl",
}
_CACHE: dict[str, tuple[float, dict]] = {}
CACHE_SECONDS = 30


def _get(url: str, params: dict | None = None) -> dict:
    key = url + str(sorted((params or {}).items()))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _CACHE[key] = (time.time(), data)
    return data


# ---------- name matching ----------

def norm(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text)
    return " ".join(text.split())


def team_matches(query: str | None, team: dict) -> bool:
    from .learned import team_alias  # rules taught by the R&D team

    q = norm(team_alias(query))
    if not q:
        return False
    names = [norm(team.get(k)) for k in ("displayName", "shortDisplayName", "name", "location", "nickname")]
    abbr = norm(team.get("abbreviation"))
    if q == abbr:
        return True
    nick0 = norm(team.get("name"))[:1]
    # Sportsbook 3-letter codes vs ESPN's 2-letter ones: "NYK" = "NY" + Knicks, "GSW" = "GS" + Warriors.
    if abbr and len(q) == 3 and len(abbr) == 2 and q.startswith(abbr) and q[2] == nick0:
        return True
    # ...and the other way round: "UTA" for ESPN's "UTAH".
    if abbr and len(q) >= 3 and " " not in q and len(abbr) > len(q) and abbr.startswith(q):
        return True
    full = names[0]
    if q == full or (full and full in q):
        return True
    nickname = norm(team.get("name"))
    # "Knicks", "NY Knicks", "Boston Red Sox" all contain the nickname as whole words.
    return bool(nickname) and re.search(rf"\b{re.escape(nickname)}\b", q) is not None


def player_matches(query: str | None, name: str | None) -> bool:
    from .learned import player_alias

    q, n = norm(player_alias(query)), norm(name)
    if not q or not n:
        return False
    if q == n:
        return True
    qp, np_ = q.split(), n.split()
    # Same last name and first initial ("J. Brunson" vs "Jalen Brunson").
    return qp[-1] == np_[-1] and qp[0][0] == np_[0][0]


# ---------- scoreboard ----------

def parse_event(ev: dict) -> dict:
    comp = (ev.get("competitions") or [{}])[0]
    status = (ev.get("status") or comp.get("status") or {}).get("type", {})
    side = {}
    for c in comp.get("competitors", []):
        try:
            score = float(c.get("score")) if c.get("score") not in (None, "") else 0.0
        except (TypeError, ValueError):
            score = 0.0
        side[c.get("homeAway", "home")] = {"team": c.get("team", {}), "score": score}
    return {
        "id": ev.get("id"),
        "date": ev.get("date"),
        "name": ev.get("shortName") or ev.get("name"),
        "state": status.get("state", "pre"),  # pre | in | post
        "detail": status.get("shortDetail") or status.get("detail") or "",
        "home": side.get("home", {"team": {}, "score": 0.0}),
        "away": side.get("away", {"team": {}, "score": 0.0}),
        "odds": (comp.get("odds") or [None])[0],
    }


def scoreboard(sport: str, day: date) -> list[dict]:
    data = _get(f"{ESPN}/{PATHS[sport]}/scoreboard", {"dates": day.strftime("%Y%m%d")})
    return [parse_event(e) for e in data.get("events", [])]


def _candidate_days(game_date: str | None) -> list[date]:
    if game_date:
        try:
            d = datetime.strptime(game_date[:10], "%Y-%m-%d").date()
            return [d, d - timedelta(days=1), d + timedelta(days=1)]
        except ValueError:
            pass
    today = date.today()
    return [today, today - timedelta(days=1)] + [today + timedelta(days=i) for i in range(1, 7)]


def find_game(leg: dict) -> dict | None:
    sport = (leg.get("sport") or "").upper()
    if sport not in PATHS:
        return None
    for day in _candidate_days(leg.get("game_date")):
        try:
            events = scoreboard(sport, day)
        except requests.RequestException:
            continue
        for ev in events:
            teams = [ev["home"]["team"], ev["away"]["team"]]
            if any(team_matches(leg.get("team"), t) for t in teams):
                return ev
            if leg.get("opponent") and any(team_matches(leg.get("opponent"), t) for t in teams):
                return ev
    return None


# ---------- box scores ----------

def parse_boxscore(summary: dict) -> list[dict]:
    """Flatten ESPN summary JSON into [{name, team, groups: {group: {label: value}}}]."""
    players: dict[str, dict] = {}
    for team_block in (summary.get("boxscore") or {}).get("players", []):
        team_name = (team_block.get("team") or {}).get("displayName")
        for group in team_block.get("statistics", []):
            gname = (group.get("name") or group.get("type") or "").lower() or None
            labels = group.get("labels") or group.get("keys") or []
            for ath in group.get("athletes", []):
                name = (ath.get("athlete") or {}).get("displayName")
                if not name:
                    continue
                p = players.setdefault(name, {"name": name, "team": team_name, "groups": {}})
                p["groups"][gname or f"group{len(p['groups'])}"] = dict(zip(labels, ath.get("stats") or []))
    return list(players.values())


def boxscore(sport: str, event_id: str) -> list[dict]:
    return parse_boxscore(_get(f"{ESPN}/{PATHS[sport]}/summary", {"event": event_id}))


def game_summary(sport: str, event_id: str) -> dict:
    """Raw ESPN summary (injuries, news, predictor, odds) for the research agents."""
    return _get(f"{ESPN}/{PATHS[sport]}/summary", {"event": event_id})
