"""The R&D team's report to you, built to be clear and honest.

How it stays honest:
  * Every fact (what was looked at, what passed or failed, what changed,
    what it cost) is counted by code from the app's own records. The AI
    models never write the numbers, the statuses or the bottom line.
  * Anything a model wrote (issues, ideas, reviewer notes) sits under a
    "model's view" label.
  * "Not proven yet" is always included: passing tests is not proof that a
    change fixed the real problem.
  * Failures, rejections, undos and errors are reported, not hidden.
  * Cost is labelled as an estimate.

The R&D agents cannot edit this file (it is not in rnd.EDITABLE).
"""
from __future__ import annotations

import re
from datetime import datetime

SIGNAL_LABELS = {
    "slip_correction": "Slips you corrected after the reader ran",
    "leg_not_found": "Legs the tracker couldn't match",
    "error": "Errors",
    "feedback": "Feedback you sent",
    "report_rating": "Research report ratings",
}
STATUS_TEXT = {"ready": "Ready for your review", "failed": "Didn't pass its checks", "applied": "Applied",
               "rejected": "Rejected by you", "undone": "Undone"}
REAL_WORLD_CHECK = {
    "tracking": "After the next games, check the legs that were unmatched now match (playbook TR tasks).",
    "slip_reading": "Read 3 slips from the same sportsbook and count how many fields you still fix (SR tasks).",
    "research": "Rate the next 3 research reports 👍/👎 and compare with before (RS tasks).",
    "ui": "Use the changed screen on a real parlay and decide if it's actually better.",
    "odds": "Compare the app's payout with your sportsbook's on a real slip.",
}


def parse_tests(tests: dict | None) -> tuple[int | None, int | None]:
    """(passed, failed) from a pytest summary like '1 failed, 43 passed in 4.2s'."""
    if not tests:
        return None, None
    text = (tests.get("summary") or "") + "\n" + (tests.get("output") or "")[-400:]
    passed = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) failed", text)
    p = int(passed.group(1)) if passed else None
    f = int(failed.group(1)) if failed else (0 if tests.get("passed") else None)
    return p, f


def tests_text(tests: dict | None) -> str:
    if not tests:
        return "not run"
    p, f = parse_tests(tests)
    if tests.get("passed"):
        return f"✅ {p} passed" if p is not None else "✅ passed"
    bits = [f"{f} failed" if f else "failed", f"{p} passed" if p is not None else ""]
    return "❌ " + ", ".join(b for b in bits if b)


def _cell(text, limit: int = 140) -> str:
    text = " ".join(str(text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _when(iso: str | None) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %I:%M %p").replace(" 0", " ")
    except (TypeError, ValueError):
        return str(iso or "")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def changes_since(all_proposals: list[dict], since: str | None, exclude_ids: set[int] = frozenset()) -> list[dict]:
    """Status changes (applied/rejected/undone, plus new ones) after `since`."""
    out = []
    for p in all_proposals:
        if p["id"] in exclude_ids:
            continue
        for h in p["data"].get("history") or []:
            if since is None or h.get("at", "") > since:
                out.append({"id": p["id"], "title": p["title"], "status": h["status"], "at": h.get("at")})
    return sorted(out, key=lambda c: (c["at"] or "", c["id"]))


def build_report(
    *,
    kind: str,                      # "cycle" (after an R&D cycle) or "status" (free, no models)
    created_at: str,
    goal: str = "",
    signal_counts: dict | None = None,
    prev_report: dict | None = None,
    issues: list[dict] | None = None,
    ideas: list[dict] | None = None,
    cycle_proposals: list[dict] | None = None,
    all_proposals: list[dict] | None = None,
    usage: dict | None = None,
    error: str | None = None,
) -> tuple[str, dict]:
    signal_counts = signal_counts or {}
    cycle_proposals = cycle_proposals or []
    all_proposals = all_proposals or []
    issues, ideas, usage = issues or [], ideas or [], usage or {}
    since = prev_report["created_at"] if prev_report else None
    cycle_ids = {p["id"] for p in cycle_proposals}
    changes = changes_since(all_proposals, since, exclude_ids=cycle_ids)
    waiting = [p for p in all_proposals if p["status"] == "ready"]
    ready = [p for p in cycle_proposals if p["status"] == "ready"]
    failed = [p for p in cycle_proposals if p["status"] == "failed"]
    applied_since = [c for c in changes if c["status"] == "applied"]
    total_signals = sum(signal_counts.values())
    total_cost = sum(v["est_cost"] for v in usage.values() if v.get("est_cost") is not None)
    unknown_price = [m for m, v in usage.items() if v.get("est_cost") is None]

    L: list[str] = []
    title = "R&D cycle report" if kind == "cycle" else "R&D status report"
    L += [f"# {title} · {_when(created_at)}", ""]

    # ---- bottom line (facts only) ----
    if error:
        bottom = (f"The R&D cycle stopped with an error before it finished: {_cell(error, 300)}. "
                  "Nothing in the app was changed.")
    elif kind == "cycle":
        if cycle_proposals:
            bottom = (f"The team made {_plural(len(cycle_proposals), 'proposal')}: {len(ready)} ready for your "
                      f"review, {len(failed)} didn't pass its checks. Nothing in the app has changed yet; "
                      "changes only happen when you approve them.")
        else:
            bottom = ("The team didn't produce any proposals this cycle, so nothing changed. "
                      + ("There was no new evidence and no goal to work from." if not total_signals and not goal
                         else "See what it looked at below."))
    else:
        bottom = (f"Since the last report: {len(applied_since)} applied, "
                  f"{sum(c['status'] == 'rejected' for c in changes)} rejected, "
                  f"{sum(c['status'] == 'undone' for c in changes)} undone. "
                  f"{_plural(len(waiting), 'proposal')} waiting for your review. This report used no credits.")
    L += ["**Bottom line:** " + bottom, ""]

    # ---- evidence ----
    if kind == "cycle":
        L += ["## What the team looked at", ""]
        if goal:
            L += [f"**Your goal:** {_cell(goal, 400)}", ""]
        if total_signals:
            L += ["| Signal from real use | New since last cycle |", "| --- | --- |"]
            for k, label in SIGNAL_LABELS.items():
                L.append(f"| {label} | {signal_counts.get(k, 0)} |")
            prev_counts = (prev_report or {}).get("data", {}).get("signal_counts")
            if prev_counts:
                L += ["", f"Last cycle had {sum(prev_counts.values())} signals; this one had {total_signals}. "
                      "Fewer signals can mean the app improved, or simply that it was used less."]
        else:
            L.append("**No new signals from real use.** Everything below is based on "
                     + ("your goal" if goal else "the app's design") + " only, not on evidence from the app.")
        L.append("")

        if issues or ideas:
            L += ["## What the team found (model's view)", "",
                  "_Written by an AI model from the signals above. It can be wrong._", ""]
            for i in issues:
                L.append(f"- **Issue ({_cell(i.get('severity', '?'), 10)}):** {_cell(i.get('title'))}. "
                         f"Evidence: {_cell(i.get('evidence'), 200)}")
            for i in ideas:
                L.append(f"- **Idea ({_cell(i.get('effort', '?'), 10)} effort):** {_cell(i.get('title'))}. "
                         f"{_cell(i.get('why'), 200)}")
            L.append("")

    # ---- proposals ----
    if kind == "cycle" and cycle_proposals:
        L += ["## Proposals", "",
              "| # | Proposal | Type | Result | Tests (facts) | Tries | Reviewer (model's view) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for p in cycle_proposals:
            d = p["data"]
            rv = d.get("review") or {}
            reviewer = f"{rv.get('recommend', '-')}, risk {rv.get('risk', '-')}" if rv else "-"
            L.append(f"| {p['id']} | {_cell(p['title'], 80)} | {_cell((d.get('item') or {}).get('kind', '-'), 10)} | "
                     f"{STATUS_TEXT.get(p['status'], p['status'])} | {tests_text(d.get('tests'))} | "
                     f"{d.get('attempts', '-')} | {_cell(reviewer, 40)} |")
        L.append("")
        flagged = [p for p in ready if (p["data"].get("review") or {}).get("recommend") == "reject"
                   or (p["data"].get("review") or {}).get("risk") == "high"]
        for p in flagged:
            rv = p["data"]["review"]
            L.append(f"- ⚠️ **#{p['id']} passed its tests, but the Reviewer has concerns** "
                     f"({rv.get('recommend')}, risk {rv.get('risk')}): {_cell(rv.get('notes'), 250)}")
        for p in ready:
            if p["data"].get("summary"):
                L.append(f"- **#{p['id']} in plain words** (Engineer's own description): "
                         f"{_cell(p['data']['summary'], 250)}")
        if flagged or ready:
            L.append("")

    if failed:
        L += ["## What didn't work", ""]
        for p in failed:
            reasons = "; ".join(_cell(r, 200) for r in (p["data"].get("problems") or ["unknown reason"]))
            L.append(f"- **#{p['id']} {_cell(p['title'], 80)}**: {reasons} "
                     f"(tests: {tests_text(p['data'].get('tests'))}, tries: {p['data'].get('attempts', '-')})")
        L.append("")

    # ---- changes since last report ----
    L += ["## Changes since the last report", ""]
    if prev_report is None:
        L.append("This is the first report.")
    elif not changes:
        L.append(f"Nothing changed since the last report ({_when(since)}).")
    else:
        for c in changes:
            L.append(f"- #{c['id']} {_cell(c['title'], 80)}: **{STATUS_TEXT.get(c['status'], c['status'])}** "
                     f"({_when(c['at'])})")
    L.append("")

    # ---- not proven ----
    L += ["## Not proven yet", "",
          "- Passing tests shows a change doesn't break what the tests check. It does **not** prove the change "
          "fixes the real problem."]
    to_check = ready + [p for p in all_proposals if p["id"] in {c["id"] for c in applied_since}]
    for p in to_check:
        area = (p["data"].get("item") or {}).get("area", "")
        L.append(f"- #{p['id']}: {REAL_WORLD_CHECK.get(area, 'Try it on a real slip or game and see.')}")
    L.append("- Issues, ideas, reviewer notes and the Engineer's descriptions are written by AI models and "
             "can be wrong. Read the code change before approving.")
    if applied_since or any(p["status"] == "applied" for p in all_proposals):
        L.append("- Approved changes live only in your Codespace until you push them to GitHub.")
    L.append("")

    # ---- cost ----
    L += ["## Cost (estimate)", ""]
    if usage:
        L += ["| Model | Calls | Tokens in | Tokens out | Est. cost |", "| --- | --- | --- | --- | --- |"]
        for m, v in sorted(usage.items()):
            cost = "unknown price" if v.get("est_cost") is None else f"${v['est_cost']:.4f}"
            L.append(f"| `{m}` | {v['calls']} | {v['input_tokens']:,} | {v['output_tokens']:,} | {cost} |")
        L += ["", f"Estimated total: **${total_cost:.4f}**"
              + (f" plus {len(unknown_price)} model(s) with unknown price" if unknown_price else "")
              + ". Based on approximate prices; your Nebius billing page has the real number."]
    else:
        L.append("No model calls were made, so this report used no credits." if kind == "status" or error
                 else "No model usage was recorded.")
    L.append("")

    # ---- next steps (facts → actions) ----
    steps = []
    if waiting:
        steps.append(f"Review the {_plural(len(waiting), 'proposal')} waiting in the R&D Lab: read each code "
                     "change, then approve or reject.")
    if applied_since:
        steps.append("Save approved changes to GitHub: `git add -A && git commit -m \"R&D changes\" && git push`")
    steps.append("Add a row to the Run log in your Eval Playbook.")
    if kind == "cycle" and total_signals < 5:
        steps.append("Use the app on real games so the next cycle has more real evidence to work from.")
    L += ["## Next steps for you", ""] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]

    data = {
        "kind": kind, "goal": goal, "signal_counts": signal_counts, "error": error,
        "proposals": [p["id"] for p in cycle_proposals], "ready": [p["id"] for p in ready],
        "failed": [p["id"] for p in failed], "changes": changes, "usage": usage,
        "est_cost": round(total_cost, 4),
    }
    return "\n".join(L) + "\n", data
