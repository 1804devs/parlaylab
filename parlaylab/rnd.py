"""The R&D team: agents that study how ParlayLab is doing and improve it.

Every agent runs on NVIDIA Nemotron 3 Super via Nebius Token Factory.

  1. QA Analyst          reads what went wrong (slip fixes you made, legs the
                         tracker couldn't find, errors, bad report ratings)
  2. Product Researcher  reads your feedback and usage, proposes features
  3. R&D Lead            picks the most valuable work and writes the tasks
  4. Engineer            writes the change: a learned rule (aliases, stat
                         labels, slip-reading lessons) or a code edit + test
  5. Tester              (code, not a model) runs the full test suite on a
                         patched copy of the app; failures go back to the
                         Engineer for one more try
  6. Reviewer            checks the diff and test results, rates the risk

Nothing touches the real app until you press Approve. Approving backs up
every file first, re-runs the tests on the real app, and rolls back if
they fail. Every applied change can be undone.

Guardrails: agents can only edit the files in EDITABLE. They can't edit
this file, the model client, the database layer, the learned-rules
loader, or the existing test suite (so they can't weaken the tests that
judge them). They can add new test files named tests/test_rnd_*.py.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import db, learned
from .nebius_client import AGENT_MODEL, chat, extract_json

PROJECT = Path(__file__).resolve().parent.parent
BACKUPS = PROJECT / ".rnd_backups"

EDITABLE = {
    "app.py",
    "parlaylab/agents.py",
    "parlaylab/grader.py",
    "parlaylab/odds.py",
    "parlaylab/scores.py",
    "parlaylab/slip_reader.py",
    "parlaylab/stats_catalog.py",
}
NEW_TEST_PATTERN = re.compile(r"^tests/test_rnd_[a-z0-9_]{1,40}\.py$")

AREA_FILES = {
    "slip_reading": ["parlaylab/slip_reader.py", "parlaylab/stats_catalog.py"],
    "tracking": ["parlaylab/scores.py", "parlaylab/grader.py", "parlaylab/stats_catalog.py"],
    "research": ["parlaylab/agents.py"],
    "ui": ["app.py"],
    "odds": ["parlaylab/odds.py"],
}
AREAS = list(AREA_FILES)

APP_OVERVIEW = """ParlayLab is a Streamlit app for sports bettors (NFL, NBA, MLB, NHL).
- Add a parlay: upload a bet-slip screenshot; NVIDIA Nemotron 3 Nano Omni reads it into legs
  (sport, type, team, opponent, player, stat, side, line, odds, game_date) that the user checks in an
  editable table before saving. Text paste also works.
- Live tracker: grades moneyline, spread, total and player-prop legs against ESPN public scoreboards and
  box scores, refreshing every 30-300s, with pop-ups when a leg settles and a 'one leg away' alert.
- Research team: five Nemotron 3 Super agents (Odds Analyst, Scout, Stats Analyst, Devil's Advocate,
  Lead Analyst) write a scorecard per parlay.
- R&D Lab: this team, which improves the app.
Tech: Python, Streamlit, SQLite, OpenAI-compatible client to Nebius Token Factory, ESPN JSON feeds.
Learned rules (no code needed): team_aliases, player_aliases, stat_labels per sport, slip_rules."""

RULES_HELP = """Learned-rule format (merged into parlaylab/learned.json):
{"team_aliases": {"alias text": "Full Team Name"},
 "player_aliases": {"alias text": "Full Player Name"},
 "stat_labels": {"NBA": {"stat_key": [[group_or_null, "ESPN_LABEL"], ...]}},   // sums the labels
 "slip_rules": ["short instruction added to the slip reader's prompt"]}
ESPN box score groups: NFL passing/rushing/receiving/..., MLB batting/pitching, NBA and NHL use null."""

GROUND_RULES = (
    "You are on the R&D team for ParlayLab, a parlay tracking app. Your job is to make the app work "
    "better for its user. Base every claim on the evidence given; if the evidence is thin, say so. "
    "Prefer small, safe, testable improvements over big rewrites. Keep responsible-gambling messaging. "
    "Return only what is asked for."
)

ProgressFn = Callable[[str], None]


# ======================= signals =======================

LEG_COMPARE_FIELDS = ["sport", "type", "team", "opponent", "player", "stat", "side", "line", "odds", "game_date"]


def diff_legs(ai_legs: list[dict], saved_legs: list[dict]) -> dict:
    """What the user had to fix after the slip reader ran."""
    changes = []
    for i, (a, s) in enumerate(zip(ai_legs, saved_legs)):
        for f in LEG_COMPARE_FIELDS:
            av, sv = a.get(f), s.get(f)
            if isinstance(av, (int, float)) and isinstance(sv, (int, float)):
                same = abs(float(av) - float(sv)) < 1e-9
            else:
                same = (str(av).strip().lower() if av is not None else None) == (
                    str(sv).strip().lower() if sv is not None else None)
            if not same:
                changes.append({"leg": i + 1, "field": f, "slip_reader_said": av, "user_fixed_to": sv,
                                "slip_text": a.get("description")})
    return {
        "changes": changes,
        "legs_added_by_user": max(0, len(saved_legs) - len(ai_legs)),
        "legs_removed_by_user": max(0, len(ai_legs) - len(saved_legs)),
    }


def collect_signals(events: list[dict]) -> dict:
    sig: dict = {"slip_fixes": [], "not_found_legs": [], "errors": [], "feedback": [], "report_ratings": []}
    for e in events:
        k, p = e["kind"], e["payload"]
        if k == "slip_correction":
            d = diff_legs(p.get("ai_legs") or [], p.get("saved_legs") or [])
            if d["changes"] or d["legs_added_by_user"] or d["legs_removed_by_user"]:
                sig["slip_fixes"].append({"book": p.get("book"), "source": p.get("source"), **d})
        elif k == "leg_not_found":
            leg = p.get("leg") or {}
            sig["not_found_legs"].append({**{f: leg.get(f) for f in
                                             ("sport", "type", "team", "opponent", "player", "stat", "description")},
                                          "tracker_said": p.get("detail")})
        elif k == "error":
            sig["errors"].append({"where": p.get("where"), "message": str(p.get("message"))[:300]})
        elif k == "feedback":
            sig["feedback"].append({"area": p.get("area"), "text": p.get("text")})
        elif k == "report_rating":
            sig["report_ratings"].append({"rating": p.get("rating"), "comment": p.get("comment")})
    # Cap sizes so prompts stay small.
    sig["slip_fixes"] = sig["slip_fixes"][:40]
    sig["not_found_legs"] = sig["not_found_legs"][:60]
    sig["errors"] = sig["errors"][:40]
    sig["usage"] = usage_stats()
    return sig


def usage_stats() -> dict:
    parlays = db.list_parlays()
    legs = [l for p in parlays for l in p["legs"]]
    results = Counter(s.get("result") for p in parlays for s in p["leg_status"] if s.get("result"))
    return {
        "parlays": len(parlays),
        "legs": len(legs),
        "legs_by_sport": dict(Counter(l.get("sport") for l in legs)),
        "legs_by_type": dict(Counter(l.get("type") for l in legs)),
        "stats_used": dict(Counter(l.get("stat") for l in legs if l.get("stat"))),
        "books": dict(Counter(p.get("book") for p in parlays if p.get("book"))),
        "leg_results": dict(results),
        "reports_run": sum(1 for p in parlays if p.get("report")),
    }


def signal_count(sig: dict) -> int:
    return sum(len(sig[k]) for k in ("slip_fixes", "not_found_legs", "errors", "feedback", "report_ratings"))


# ======================= patch mechanics =======================

def _safe_rel(path: str) -> str | None:
    rel = Path(path.replace("\\", "/").lstrip("/")).as_posix()
    if ".." in Path(rel).parts:
        return None
    return rel


def check_edits(edits: list[dict], root: Path = PROJECT) -> list[str]:
    problems = []
    if not isinstance(edits, list):
        return ["edits must be a list"]
    if len(edits) > 12:
        problems.append("too many edits in one proposal (max 12)")
    for i, e in enumerate(edits, 1):
        if not isinstance(e, dict) or not isinstance(e.get("file"), str):
            problems.append(f"edit {i}: needs a 'file'")
            continue
        rel = _safe_rel(e["file"])
        find, replace = e.get("find", ""), e.get("replace", "")
        if not isinstance(find, str) or not isinstance(replace, str):
            problems.append(f"edit {i}: find/replace must be text")
            continue
        if len(replace) > 20000:
            problems.append(f"edit {i}: replacement too large")
        if rel is None:
            problems.append(f"edit {i}: bad path {e['file']}")
            continue
        is_new_test = bool(NEW_TEST_PATTERN.match(rel))
        if rel not in EDITABLE and not is_new_test:
            problems.append(f"edit {i}: {rel} is not editable by the R&D team")
            continue
        target = root / rel
        if find == "":
            if not is_new_test:
                problems.append(f"edit {i}: empty 'find' is only allowed to create tests/test_rnd_*.py files")
            continue
        if not target.exists():
            problems.append(f"edit {i}: {rel} does not exist")
            continue
        n = target.read_text().count(find)
        if n != 1:
            problems.append(f"edit {i}: 'find' text appears {n} times in {rel} (must be exactly once, copied exactly)")
    return problems


def patched_files(edits: list[dict], root: Path = PROJECT) -> dict[str, tuple[str | None, str]]:
    """Return {rel_path: (old_text or None if new, new_text)} after applying edits in order."""
    out: dict[str, tuple[str | None, str]] = {}
    for e in edits:
        rel = _safe_rel(e["file"])
        target = root / rel
        if rel in out:
            old, cur = out[rel]
        else:
            old = target.read_text() if target.exists() else None
            cur = old or ""
        if e.get("find", "") == "":
            cur = cur + ("\n" if cur and not cur.endswith("\n") else "") + e["replace"]
        else:
            if cur.count(e["find"]) != 1:
                raise ValueError(f"'find' text no longer matches exactly once in {rel}")
            cur = cur.replace(e["find"], e["replace"], 1)
        out[rel] = (old, cur)
    return out


def make_diff(edits: list[dict], rule_changes: dict | None, root: Path = PROJECT) -> str:
    parts = []
    for rel, (old, new) in patched_files(edits, root).items():
        parts.extend(difflib.unified_diff((old or "").splitlines(keepends=True), new.splitlines(keepends=True),
                                          fromfile=f"a/{rel}" if old is not None else "/dev/null",
                                          tofile=f"b/{rel}"))
    if not learned.is_empty(rule_changes):
        parts.append("\n# learned rules to add\n" + json.dumps(rule_changes, indent=2) + "\n")
    return "".join(parts)


def _copy_project(dst: Path) -> None:
    ignore = shutil.ignore_patterns("*.db", ".rnd_backups", "__pycache__", ".pytest_cache", ".git", ".env", "*.zip")
    shutil.copytree(PROJECT, dst, ignore=ignore, dirs_exist_ok=True)


def _run_tests(root: Path, learned_path: Path, timeout: int = 240) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PARLAY_LEARNED", "PARLAY_DB", "NEBIUS_API_KEY")}
    env["PARLAY_LEARNED"] = str(learned_path)
    env["PARLAY_DB"] = str(root / "rnd_test.db")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PARLAY_IN_SANDBOX"] = "1"  # lets the R&D tests skip themselves (no test-inside-test loops)
    # app.py isn't imported by the tests, so syntax-check every file first.
    for f in [root / "app.py", *(root / "parlaylab").glob("*.py"), *(root / "tests").glob("*.py")]:
        try:
            ast.parse(f.read_text(), filename=str(f))
        except SyntaxError as e:
            return {"passed": False, "summary": "syntax error",
                    "output": f"Syntax error in {f.relative_to(root)} line {e.lineno}: {e.msg}"}
    try:
        res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                             cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"passed": False, "summary": "timed out", "output": "Tests timed out."}
    finally:
        Path(env["PARLAY_DB"]).unlink(missing_ok=True)
    out = (res.stdout + res.stderr).strip()
    summary = out.splitlines()[-1] if out else ""
    return {"passed": res.returncode == 0, "summary": summary, "output": out[-3000:]}


def sandbox_test(edits: list[dict], rule_changes: dict | None) -> dict:
    """Apply a proposal to a throwaway copy of the app and run the whole test suite."""
    with tempfile.TemporaryDirectory(prefix="parlaylab_rnd_") as tmp:
        root = Path(tmp)
        _copy_project(root)
        for rel, (_old, new) in patched_files(edits, PROJECT).items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(new)
        lp = root / "parlaylab" / "learned.json"
        lp.write_text(json.dumps(learned.merge(learned.load(), rule_changes or {}), indent=2))
        return _run_tests(root, lp)


def apply_proposal(proposal: dict) -> dict:
    """Back up, apply, re-test on the real app; roll back if tests fail."""
    data = proposal["data"]
    edits, rules = data.get("edits") or [], data.get("rule_changes") or {}
    problems = check_edits(edits) + (learned.validate(rules) if rules else [])
    if problems:
        return {"ok": False, "message": "Can't apply: " + "; ".join(problems)}

    files = patched_files(edits)
    backup = BACKUPS / f"proposal-{proposal['id']}-{datetime.now():%Y%m%d-%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    created = []
    for rel, (old, _new) in files.items():
        if old is None:
            created.append(rel)
        else:
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            (backup / rel).write_text(old)
    lp = learned.path()
    if lp.exists():
        (backup / "learned.json").write_text(lp.read_text())
    (backup / "manifest.json").write_text(json.dumps({"created": created, "learned_existed": lp.exists(),
                                                       "learned_path": str(lp)}))

    for rel, (_old, new) in files.items():
        (PROJECT / rel).parent.mkdir(parents=True, exist_ok=True)
        (PROJECT / rel).write_text(new)
    if not learned.is_empty(rules):
        learned.save(learned.merge(learned.load(), rules))

    result = _run_tests(PROJECT, lp)
    if not result["passed"]:
        restore_backup(backup)
        return {"ok": False, "message": "Tests failed on the real app, so the change was rolled back.",
                "tests": result}
    return {"ok": True, "backup": str(backup), "tests": result}


def restore_backup(backup: Path) -> None:
    backup = Path(backup)
    manifest = json.loads((backup / "manifest.json").read_text())
    for f in backup.rglob("*"):
        if f.is_file() and f.name not in ("manifest.json", "learned.json"):
            rel = f.relative_to(backup)
            (PROJECT / rel).write_text(f.read_text())
    for rel in manifest.get("created", []):
        p = PROJECT / rel
        if p.exists():
            p.unlink()
    lp = Path(manifest.get("learned_path") or learned.path())
    if (backup / "learned.json").exists():
        lp.write_text((backup / "learned.json").read_text())
    elif not manifest.get("learned_existed") and lp.exists():
        lp.unlink()
    learned._cache["data"] = None


def undo_proposal(proposal: dict) -> dict:
    backup = (proposal["data"].get("apply_result") or {}).get("backup")
    if not backup or not Path(backup).exists():
        return {"ok": False, "message": "No backup found for this change."}
    restore_backup(Path(backup))
    return {"ok": True}


# ======================= agents =======================

def _parse_json(reply: str):
    try:
        return json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return extract_json(reply)


def _ask(role: str, task: str, payload: dict, api_key: str | None, max_tokens: int = 6000):
    reply = chat(
        [
            {"role": "system", "content": f"{GROUND_RULES}\n\nYour role: {role}"},
            {"role": "user", "content": f"{task}\n\nDATA:\n{json.dumps(payload, indent=1, default=str)}"},
        ],
        model=AGENT_MODEL, api_key=api_key, max_tokens=max_tokens, temperature=0.2,
    )
    return _parse_json(reply)


def qa_analyst(signals: dict, api_key=None) -> list[dict]:
    if not any(signals[k] for k in ("slip_fixes", "not_found_legs", "errors", "report_ratings")):
        return []
    out = _ask(
        "QA Analyst",
        "Find the real problems in this app's records. Group repeats into one issue. For each issue give "
        "the evidence (how many times, concrete examples) and what kind of fix would solve it. Common fixes: "
        "a team or player alias, an ESPN stat label, a slip-reading lesson, or a code change.\n"
        'Return JSON: {"issues": [{"title": "...", "area": one of ' + json.dumps(AREAS) + ', '
        '"severity": "high"|"medium"|"low", "evidence": "...", "fix_hint": "..."}]}',
        {k: signals[k] for k in ("slip_fixes", "not_found_legs", "errors", "report_ratings")},
        api_key,
    )
    return (out or {}).get("issues", []) if isinstance(out, dict) else []


def product_researcher(signals: dict, goal: str, api_key=None) -> list[dict]:
    out = _ask(
        "Product Researcher",
        "Suggest improvements that would make this app more useful for someone who bets parlays. Start "
        "from the user's own feedback and goal; then use the usage data. Favour things that can be built "
        "in a small change. Don't suggest anything that pushes people to bet more.\n"
        'Return JSON: {"ideas": [{"title": "...", "area": one of ' + json.dumps(AREAS) + ', '
        '"why": "...", "effort": "small"|"medium"|"large"}]}',
        {"app": APP_OVERVIEW, "user_goal": goal or None, "feedback": signals["feedback"],
         "report_ratings": signals["report_ratings"], "usage": signals["usage"]},
        api_key,
    )
    return (out or {}).get("ideas", []) if isinstance(out, dict) else []


def rnd_lead(issues: list[dict], ideas: list[dict], goal: str, max_items: int, api_key=None) -> list[dict]:
    out = _ask(
        "R&D Lead",
        f"Pick at most {max_items} pieces of work with the best value for the effort. The user's goal comes "
        "first, then high-severity issues, then quick wins. For each, write a precise task for the Engineer. "
        "Use kind 'rule' when a learned rule fixes it (aliases, stat labels, slip lessons), otherwise 'code'. "
        "Skip large efforts.\n"
        'Return JSON: {"plan": [{"title": "...", "area": one of ' + json.dumps(AREAS) + ', '
        '"kind": "rule"|"code", "why": "...", "task": "exact instructions"}]}',
        {"user_goal": goal or None, "issues": issues, "ideas": ideas, "app": APP_OVERVIEW, "rules": RULES_HELP},
        api_key,
    )
    plan = (out or {}).get("plan", []) if isinstance(out, dict) else []
    return [p for p in plan if isinstance(p, dict) and p.get("title")][:max_items]


def _files_for(area: str) -> dict[str, str]:
    files = AREA_FILES.get(area) or AREA_FILES["ui"]
    return {f: (PROJECT / f).read_text() for f in files}


def engineer(item: dict, api_key=None, previous: dict | None = None, problem: str | None = None) -> dict:
    files = _files_for(item.get("area", "ui"))
    task = (
        "Implement this task for ParlayLab.\n"
        "- If a learned rule solves it, put it in rule_changes and leave edits empty.\n"
        "- Otherwise make the smallest code edits that work. Each edit replaces one exact block: 'find' must "
        "be copied character-for-character from the file and appear exactly once; include a few lines of "
        "context so it's unique.\n"
        "- For code changes, also add a pytest file named tests/test_rnd_<short_name>.py (edit with empty "
        "'find' and the whole file as 'replace') that checks the new behaviour without network access. "
        "Import from the parlaylab package.\n"
        f"- You may only edit: {sorted(EDITABLE)} and create tests/test_rnd_*.py. No new dependencies.\n"
        "- Keep existing behaviour and function signatures working; other code and tests depend on them.\n"
        'Return JSON: {"summary": "what you changed, in plain words for a non-programmer", '
        '"rule_changes": {...} or {}, "edits": [{"file": "...", "find": "...", "replace": "..."}]}'
    )
    payload = {"task": item, "rules_format": RULES_HELP, "current_rules": learned.load(), "files": files,
               "test_style_example": (PROJECT / "tests/test_core.py").read_text()[:2500]}
    if previous:
        payload["your_previous_attempt"] = previous
        payload["what_went_wrong"] = problem
        task += "\n\nYour previous attempt failed. Fix it using what_went_wrong."
    out = _ask("Engineer", task, payload, api_key, max_tokens=16000)
    if not isinstance(out, dict):
        out = {}
    return {"summary": out.get("summary", ""), "rule_changes": out.get("rule_changes") or {},
            "edits": out.get("edits") or []}


def reviewer(item: dict, work: dict, diff: str, tests: dict, api_key=None) -> dict:
    out = _ask(
        "Reviewer",
        "Review this change before it goes to the user for approval. Check it does what the task asks, "
        "doesn't break other behaviour, doesn't remove safety or responsible-gambling text, and has no "
        "security problems (no shell commands, no sending data anywhere new, no secrets).\n"
        'Return JSON: {"risk": "low"|"medium"|"high", "recommend": "approve"|"reject", '
        '"notes": "2-4 short sentences in plain words"}',
        {"task": item, "summary": work.get("summary"), "diff": diff[:30000], "tests": tests},
        api_key,
    )
    return out if isinstance(out, dict) else {"risk": "unknown", "recommend": "reject", "notes": str(out)}


def build_proposal(item: dict, api_key=None, progress: ProgressFn = lambda m: None) -> int:
    """Engineer → checks → Tester → (one retry) → Reviewer. Saves and returns the proposal id."""
    title = item.get("title", "Improvement")
    attempts, work, tests, problems = 0, None, None, []
    previous, problem = None, None
    while attempts < 2:
        attempts += 1
        progress(f"Engineer is building: {title}" + (" (second try)" if attempts > 1 else ""))
        try:
            work = engineer(item, api_key, previous, problem)
        except Exception as e:
            problems = [f"Engineer failed: {e}"]
            break
        if not work["edits"] and learned.is_empty(work["rule_changes"]):
            problems = ["The Engineer didn't produce a change."]
            previous, problem = work, problems[0]
            continue
        problems = check_edits(work["edits"]) + (learned.validate(work["rule_changes"])
                                                 if work["rule_changes"] else [])
        if problems:
            previous, problem = work, "; ".join(problems)
            continue
        progress(f"Tester is running the test suite on a copy of the app: {title}")
        tests = sandbox_test(work["edits"], work["rule_changes"])
        if tests["passed"]:
            break
        previous, problem = work, "Tests failed:\n" + tests["output"]

    data = {"item": item, "attempts": attempts, "model": AGENT_MODEL}
    if work:
        data.update(summary=work["summary"], edits=work["edits"], rule_changes=work["rule_changes"])
    if problems or not tests or not tests.get("passed"):
        data["problems"] = problems or ["Tests failed after two tries."]
        data["tests"] = tests
        try:
            data["diff"] = make_diff(work["edits"], work["rule_changes"]) if work and not problems else ""
        except Exception:
            data["diff"] = ""
        return db.add_proposal(title, data, "failed")

    diff = make_diff(work["edits"], work["rule_changes"])
    progress(f"Reviewer is checking: {title}")
    try:
        review = reviewer(item, work, diff, tests, api_key)
    except Exception as e:
        review = {"risk": "unknown", "recommend": "reject", "notes": f"Review failed: {e}"}
    data.update(diff=diff, tests=tests, review=review)
    return db.add_proposal(title, data, "ready")


def run_rnd_cycle(api_key: str | None = None, goal: str = "", max_items: int = 3,
                  progress: ProgressFn = lambda m: None) -> dict:
    events = db.list_events(only_new=True)
    signals = collect_signals(events)
    progress(f"Collected {signal_count(signals)} new signals from the app")

    progress("QA Analyst and Product Researcher are studying the app")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_qa = pool.submit(qa_analyst, signals, api_key)
        f_pr = pool.submit(product_researcher, signals, goal, api_key)
        issues, ideas = f_qa.result(), f_pr.result()

    progress(f"R&D Lead is planning ({len(issues)} issues, {len(ideas)} ideas)")
    plan = rnd_lead(issues, ideas, goal, max_items, api_key)
    if not plan:
        return {"issues": issues, "ideas": ideas, "plan": [], "proposals": []}

    with ThreadPoolExecutor(max_workers=min(3, len(plan))) as pool:
        ids = list(pool.map(lambda it: build_proposal(it, api_key, progress), plan))

    db.mark_events_used([e["id"] for e in events])
    return {"issues": issues, "ideas": ideas, "plan": plan, "proposals": ids}
