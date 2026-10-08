"""Rules the R&D team has taught the app.

Stored as JSON so the R&D agents can improve matching and slip reading
without touching code:

  team_aliases    {"bkn": "Brooklyn Nets", "the garden": "New York Knicks"}
  player_aliases  {"shai": "Shai Gilgeous-Alexander"}
  stat_labels     {"NBA": {"double_double": [[null, "DD"]]}}   extra / fixed prop stats
  slip_rules      ["Hard Rock shows player props as 'Name - Stat O/U line'"]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

EMPTY = {"team_aliases": {}, "player_aliases": {}, "stat_labels": {}, "slip_rules": []}
_cache: dict = {"mtime": None, "data": None, "path": None}


def path() -> Path:
    return Path(os.getenv("PARLAY_LEARNED", Path(__file__).with_name("learned.json")))


def load() -> dict:
    p = path()
    mtime = p.stat().st_mtime if p.exists() else None
    if _cache["path"] == p and _cache["mtime"] == mtime and _cache["data"] is not None:
        return _cache["data"]
    data = json.loads(json.dumps(EMPTY))
    if p.exists():
        try:
            raw = json.loads(p.read_text())
            for k in EMPTY:
                if k in raw:
                    data[k] = raw[k]
        except json.JSONDecodeError:
            pass
    _cache.update(mtime=mtime, data=data, path=p)
    return data


def save(data: dict) -> None:
    path().write_text(json.dumps(data, indent=2, sort_keys=True))
    _cache["data"] = None


def _norm(s: str) -> str:
    from .scores import norm  # local import avoids a cycle
    return norm(s)


def validate(changes: dict) -> list[str]:
    """Return a list of problems with a proposed rule change (empty = OK)."""
    from .stats_catalog import SPORTS

    problems = []
    if not isinstance(changes, dict):
        return ["rule changes must be an object"]
    for key in changes:
        if key not in EMPTY:
            problems.append(f"unknown rule type '{key}'")
    for key in ("team_aliases", "player_aliases"):
        val = changes.get(key, {})
        if not isinstance(val, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and v
                                                for k, v in val.items()):
            problems.append(f"{key} must map text to text")
    labels = changes.get("stat_labels", {})
    if not isinstance(labels, dict):
        problems.append("stat_labels must be an object")
    else:
        for sport, stats in labels.items():
            if sport not in SPORTS:
                problems.append(f"stat_labels: unknown sport '{sport}'")
                continue
            if not isinstance(stats, dict):
                problems.append(f"stat_labels.{sport} must be an object")
                continue
            for stat, lookups in stats.items():
                ok = isinstance(lookups, list) and lookups and all(
                    isinstance(l, (list, tuple)) and len(l) == 2
                    and (l[0] is None or isinstance(l[0], str)) and isinstance(l[1], str)
                    for l in lookups)
                if not ok:
                    problems.append(f"stat_labels.{sport}.{stat} must be a list of [group or null, label]")
    rules = changes.get("slip_rules", [])
    if not isinstance(rules, list) or not all(isinstance(r, str) and 0 < len(r) <= 300 for r in rules):
        problems.append("slip_rules must be a list of short strings")
    return problems


def merge(base: dict, changes: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key in ("team_aliases", "player_aliases"):
        for k, v in (changes.get(key) or {}).items():
            out[key][_norm(k)] = v
    for sport, stats in (changes.get("stat_labels") or {}).items():
        out["stat_labels"].setdefault(sport, {}).update(
            {s: [list(l) for l in lookups] for s, lookups in stats.items()})
    for rule in changes.get("slip_rules") or []:
        if rule not in out["slip_rules"]:
            out["slip_rules"].append(rule)
    return out


def is_empty(changes: dict | None) -> bool:
    return not changes or not any(changes.get(k) for k in EMPTY)


# ---- lookups used by the rest of the app ----

def team_alias(query: str | None) -> str | None:
    if not query:
        return query
    return load()["team_aliases"].get(_norm(query), query)


def player_alias(query: str | None) -> str | None:
    if not query:
        return query
    return load()["player_aliases"].get(_norm(query), query)


def stat_lookups(sport: str) -> dict:
    return {s: [tuple(l) for l in v] for s, v in (load()["stat_labels"].get(sport) or {}).items()}


def slip_rules() -> list[str]:
    return list(load()["slip_rules"])
