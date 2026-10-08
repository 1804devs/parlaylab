"""The research team: five agents on NVIDIA Nemotron 3 Super.

  1. Odds Analyst    - payout math, implied probability, line comparison
  2. Scout           - injuries, lineup news, headlines
  3. Stats Analyst   - records, matchup, ESPN win projection, recent form
  4. Devil's Advocate- attacks the parlay: weakest leg, correlation, traps
  5. Lead Analyst    - reads everyone and writes the final scorecard

Agents 1-3 run in parallel on shared context gathered from ESPN (and
Tavily web search if TAVILY_API_KEY is set). 4 reads their notes; 5
reads everything. Nothing here predicts outcomes with certainty, and
the prompts say so.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Callable

import requests

from . import scores
from .nebius_client import AGENT_MODEL, chat, extract_json
from .odds import parlay_summary

GROUND_RULES = (
    "You are part of a sports-betting research team. Use only the context "
    "provided plus general sports knowledge, and say when data is missing "
    "instead of inventing it. Never promise a bet will win. Be concise, "
    "plain-spoken and specific. Use short bullet points."
)


# ---------- context gathering (no LLM) ----------

def leg_label(leg: dict) -> str:
    if leg.get("description"):
        return leg["description"]
    t = leg.get("type")
    if t == "player_prop":
        return f"{leg.get('player')} {leg.get('side')} {leg.get('line')} {leg.get('stat')}"
    if t == "total":
        return f"{leg.get('team')} vs {leg.get('opponent')} {leg.get('side')} {leg.get('line')}"
    if t == "spread":
        return f"{leg.get('team')} {leg.get('line'):+g}" if leg.get("line") is not None else str(leg.get("team"))
    return f"{leg.get('team')} moneyline"


def _compact_summary(summary: dict) -> dict:
    """Keep the parts of ESPN's game summary that matter for research."""
    out: dict = {}
    try:
        comps = summary["header"]["competitions"][0]["competitors"]
        out["teams"] = [
            {
                "team": c["team"].get("displayName"),
                "home_away": c.get("homeAway"),
                "record": ", ".join(r.get("summary", "") for r in c.get("record", [])[:2]),
            }
            for c in comps
        ]
    except (KeyError, IndexError, TypeError):
        pass
    injuries = []
    for team in summary.get("injuries", []) or []:
        tname = (team.get("team") or {}).get("displayName")
        for inj in (team.get("injuries") or [])[:8]:
            injuries.append(
                f"{tname}: {(inj.get('athlete') or {}).get('displayName')} - "
                f"{inj.get('status')} ({(inj.get('details') or {}).get('type') or inj.get('type', {}).get('description', '')})"
            )
    if injuries:
        out["injuries"] = injuries[:16]
    pred = summary.get("predictor") or {}
    if pred:
        out["espn_projection"] = {
            "home_win_pct": (pred.get("homeTeam") or {}).get("gameProjection"),
            "away_win_pct": (pred.get("awayTeam") or {}).get("gameProjection"),
        }
    pick = (summary.get("pickcenter") or [None])[0]
    if pick:
        out["market_line"] = {
            "provider": (pick.get("provider") or {}).get("name"),
            "details": pick.get("details"),
            "over_under": pick.get("overUnder"),
            "spread": pick.get("spread"),
        }
    news = ((summary.get("news") or {}).get("articles") or [])[:5]
    if news:
        out["headlines"] = [a.get("headline") for a in news if a.get("headline")]
    last5 = []
    for team in summary.get("lastFiveGames", []) or []:
        tname = (team.get("team") or {}).get("displayName")
        results = [e.get("gameResult") for e in (team.get("events") or []) if e.get("gameResult")]
        if results:
            last5.append(f"{tname}: {''.join(results)}")
    if last5:
        out["last_five"] = last5
    return out


def _tavily(query: str) -> list[str]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": query, "max_results": 3, "topic": "news", "days": 3},
            timeout=15,
        )
        r.raise_for_status()
        return [f"{x.get('title')}: {(x.get('content') or '')[:300]}" for x in r.json().get("results", [])]
    except requests.RequestException:
        return []


def gather_context(legs: list[dict]) -> list[dict]:
    ctx = []
    for leg in legs:
        item = {"leg": leg_label(leg), "bet": {k: v for k, v in leg.items() if v is not None}}
        try:
            game = scores.find_game(leg)
        except Exception:
            game = None
        if game:
            item["game"] = {"matchup": game.get("name"), "start": game.get("date"), "status": game.get("detail")}
            try:
                item["espn"] = _compact_summary(scores.game_summary(leg["sport"].upper(), game["id"]))
            except Exception as e:
                item["espn"] = {"error": str(e)}
        else:
            item["game"] = "Not found on ESPN schedule"
        subject = leg.get("player") or leg.get("team")
        web = _tavily(f"{subject} {leg.get('sport')} injury news lineup {date.today():%B %d %Y}")
        if web:
            item["web_news"] = web
        ctx.append(item)
    return ctx


# ---------- agents ----------

def _agent(role: str, task: str, payload: dict, api_key: str | None) -> str:
    return chat(
        [
            {"role": "system", "content": f"{GROUND_RULES}\n\nYour role: {role}"},
            {"role": "user", "content": f"{task}\n\nDATA:\n{json.dumps(payload, indent=1, default=str)}"},
        ],
        model=AGENT_MODEL,
        api_key=api_key,
    )


def odds_analyst(parlay: dict, math: dict, ctx: list[dict], api_key=None) -> str:
    return _agent(
        "Odds Analyst",
        "Explain what this parlay pays and how likely the sportsbook's own odds say it is to hit. "
        "For each leg, give the implied probability and compare the slip's line to the ESPN market "
        "line when one is provided (better or worse number?). Point out any leg where the price looks bad.",
        {"stake": parlay.get("stake"), "math": math,
         "legs": [{"leg": c["leg"], "odds": c["bet"].get("odds"), "market_line": c.get("espn", {}).get("market_line")} for c in ctx]},
        api_key,
    )


def scout(ctx: list[dict], api_key=None) -> str:
    return _agent(
        "Scout",
        "For each leg, report injuries, lineup changes and news that could affect it. "
        "Flag anything that directly threatens a leg (e.g. the player is questionable, a key teammate is out). "
        "If there's no news for a leg, say so in one line.",
        {"legs": [{"leg": c["leg"], "game": c.get("game"), "injuries": c.get("espn", {}).get("injuries"),
                   "headlines": c.get("espn", {}).get("headlines"), "web_news": c.get("web_news")} for c in ctx]},
        api_key,
    )


def stats_analyst(ctx: list[dict], api_key=None) -> str:
    return _agent(
        "Stats Analyst",
        "For each leg, assess the matchup using records, recent form (last five), and ESPN's win "
        "projection. For player props, say whether the line looks high, fair or low based on what "
        "you know of the player's typical production, and be clear when you are relying on general knowledge.",
        {"legs": [{"leg": c["leg"], "bet": c["bet"], "game": c.get("game"),
                   "teams": c.get("espn", {}).get("teams"), "last_five": c.get("espn", {}).get("last_five"),
                   "espn_projection": c.get("espn", {}).get("espn_projection")} for c in ctx]},
        api_key,
    )


def devils_advocate(ctx: list[dict], notes: dict, api_key=None) -> str:
    return _agent(
        "Devil's Advocate",
        "Your job is to find what could sink this parlay. Name the single weakest leg and why. "
        "Point out correlated legs (legs that win or lose together, or work against each other), "
        "trap lines, and anything the other analysts glossed over.",
        {"legs": [c["leg"] for c in ctx], "analyst_notes": notes},
        api_key,
    )


def lead_analyst(ctx: list[dict], math: dict, notes: dict, api_key=None) -> dict:
    reply = _agent(
        "Lead Analyst",
        "Read the team's notes and return ONLY a JSON object:\n"
        '{"verdict": "solid" | "coin flip" | "long shot",\n'
        ' "summary": "2-3 plain sentences",\n'
        ' "legs": [{"leg": "...", "confidence": "high" | "medium" | "low", "why": "one sentence"}],\n'
        ' "weakest_leg": "...",\n'
        ' "suggestion": "one concrete suggestion, e.g. drop or swap a leg, or no change"}\n'
        "Keep the legs in the same order as given. Confidence describes how well the research supports "
        "the leg, not a guarantee.",
        {"legs": [c["leg"] for c in ctx], "math": math, "notes": notes},
        api_key,
    )
    try:
        return extract_json(reply)
    except Exception:
        return {"verdict": "n/a", "summary": reply, "legs": [], "weakest_leg": "", "suggestion": ""}


def run_research_team(parlay: dict, api_key: str | None = None,
                      progress: Callable[[str], None] = lambda msg: None) -> dict:
    legs = parlay["legs"]
    progress("Gathering game data, injuries and news")
    ctx = gather_context(legs)

    odds_list = [l.get("odds") for l in legs if l.get("odds") is not None]
    math = parlay_summary(odds_list, parlay.get("stake") or 10) if len(odds_list) == len(legs) and legs else {
        "note": "Some legs are missing odds, so combined math is incomplete."}

    progress("Odds Analyst, Scout and Stats Analyst are working")
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_odds = pool.submit(odds_analyst, parlay, math, ctx, api_key)
        f_scout = pool.submit(scout, ctx, api_key)
        f_stats = pool.submit(stats_analyst, ctx, api_key)
        notes = {
            "odds_analyst": f_odds.result(),
            "scout": f_scout.result(),
            "stats_analyst": f_stats.result(),
        }

    progress("Devil's Advocate is attacking the parlay")
    notes["devils_advocate"] = devils_advocate(ctx, notes, api_key)

    progress("Lead Analyst is writing the scorecard")
    final = lead_analyst(ctx, math, notes, api_key)
    return {"created": date.today().isoformat(), "math": math, "notes": notes, "final": final,
            "model": AGENT_MODEL}
