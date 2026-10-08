"""The R&D report must be clear and honest. These tests check the rules."""
import json
from types import SimpleNamespace as NS

import pytest

from parlaylab import db, nebius_client, rnd, rnd_report


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))


def _bottom(md: str) -> str:
    return next(line for line in md.splitlines() if line.startswith("**Bottom line:**"))


def _make_cycle():
    ready = db.add_proposal("Teach Bklyn alias", {
        "item": {"area": "tracking", "kind": "rule"}, "attempts": 1, "summary": "Bklyn now means Brooklyn Nets",
        "tests": {"passed": True, "summary": "44 passed in 4.1s"},
        "review": {"risk": "high", "recommend": "reject", "notes": "Could match the wrong team."}}, "ready")
    failed = db.add_proposal("Progress bars", {
        "item": {"area": "ui", "kind": "code"}, "attempts": 2, "problems": ["Tests failed after two tries."],
        "tests": {"passed": False, "summary": "1 failed, 43 passed in 4.4s"}}, "failed")
    return ready, failed


def test_cycle_report_facts_come_from_records(tmp_db):
    ready, failed = _make_cycle()
    md, data = rnd_report.build_report(
        kind="cycle", created_at="2026-10-08T13:40:00", goal="",
        signal_counts={"leg_not_found": 3, "feedback": 1},
        issues=[{"title": "We FIXED EVERYTHING", "severity": "high", "evidence": "trust me"}],
        cycle_proposals=[db.get_proposal(ready), db.get_proposal(failed)], all_proposals=db.list_proposals(),
        usage={"nvidia/nemotron-3-super-120b-a12b": {"calls": 4, "input_tokens": 50000, "output_tokens": 9000,
                                                     "est_cost": 0.0231}})
    bottom = _bottom(md)
    assert "2 proposals: 1 ready for your review, 1 didn't pass its checks" in bottom
    assert "Nothing in the app has changed yet" in bottom
    assert "FIXED EVERYTHING" not in bottom                      # model text never in the bottom line
    assert "## What the team found (model's view)" in md          # model opinions are labelled
    assert "✅ 44 passed" in md and "❌ 1 failed, 43 passed" in md  # test numbers parsed, not invented
    assert "passed its tests, but the Reviewer has concerns" in md  # disagreement surfaced
    assert "## What didn't work" in md and "Tests failed after two tries." in md
    assert "## Not proven yet" in md and "does **not** prove" in md
    assert "Cost (estimate)" in md and "$0.0231" in md and "billing page" in md
    assert "This is the first report." in md
    assert data["ready"] == [ready] and data["failed"] == [failed]


def test_no_evidence_is_said_plainly(tmp_db):
    md, _ = rnd_report.build_report(kind="cycle", created_at="2026-10-08T13:40:00", goal="Add double-doubles",
                                    signal_counts={})
    assert "No new signals from real use" in md
    assert "based on your goal only" in md
    assert "didn't produce any proposals" in _bottom(md)
    assert "Use the app on real games" in md


def test_status_report_tracks_changes_and_is_free(tmp_db):
    ready, failed = _make_cycle()
    rnd.write_report("cycle", signal_counts={}, cycle_ids=[ready, failed])
    first = db.latest_rnd_report()
    db.update_proposal(ready, "applied")
    later = db.add_proposal("Another", {"item": {"area": "odds"}}, "ready")
    db.update_proposal(later, "rejected")
    rnd.write_report("status")
    md = db.latest_rnd_report()["markdown"]
    assert db.latest_rnd_report()["id"] != first["id"]
    bottom = _bottom(md)
    assert "1 applied, 1 rejected, 0 undone" in bottom and "no credits" in bottom
    assert "Teach Bklyn alias: **Applied**" in md and "Another: **Rejected by you**" in md
    assert "git push" in md                                   # approved changes aren't on GitHub yet
    assert "check the legs that were unmatched now match" in md  # real-world check for the applied fix


def test_failed_cycle_still_writes_an_honest_report(tmp_db, monkeypatch):
    def broken_chat(*a, **k):
        raise nebius_client.NebiusError("Nebius says you're out of credits")

    monkeypatch.setattr(rnd, "chat", broken_chat)
    db.log_event("feedback", {"area": "ui", "text": "x"})
    with pytest.raises(RuntimeError, match="report about what happened was saved"):
        rnd.run_rnd_cycle(goal="", max_items=1)
    md = db.latest_rnd_report()["markdown"]
    assert "stopped with an error" in _bottom(md) and "out of credits" in md
    assert "Nothing in the app was changed" in md
    assert db.event_counts(only_new=True) == {"feedback": 1}  # evidence kept for next time


def test_usage_is_recorded_and_priced():
    nebius_client.usage_reset()
    resp = NS(usage=NS(prompt_tokens=1_000_000, completion_tokens=500_000))
    nebius_client._record_usage("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", resp)
    nebius_client._record_usage("some/unknown-model", resp)
    snap = nebius_client.usage_snapshot()
    assert snap["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"]["est_cost"] == pytest.approx(0.06 + 0.12)
    assert snap["some/unknown-model"]["est_cost"] is None   # unknown price is not guessed
    nebius_client.usage_reset()
    assert nebius_client.usage_snapshot() == {}


def test_agents_cannot_edit_the_report_code():
    problems = rnd.check_edits([{"file": "parlaylab/rnd_report.py", "find": "import re", "replace": ""}])
    assert problems and "not editable" in problems[0]


def test_history_records_every_status_change(tmp_db):
    pid = db.add_proposal("x", {"item": {}}, "ready")
    db.update_proposal(pid, "applied")
    db.update_proposal(pid, "undone")
    assert [h["status"] for h in db.get_proposal(pid)["data"]["history"]] == ["ready", "applied", "undone"]


# ---- fixes after the first real report (Oct 8) ----
def test_times_are_shown_in_your_timezone_and_labelled(monkeypatch):
    monkeypatch.setenv("PARLAYLAB_TZ", "America/New_York")
    assert rnd_report._when("2026-10-08T17:44:00+00:00") == "Oct 8, 1:44 PM EDT"


def test_failed_cycle_after_failed_cycle_is_not_compared(tmp_db):
    prev = {"kind": "cycle", "created_at": "2026-10-08T17:43:00", "data": {"error": "x", "signal_counts": {"error": 7}}}
    md, _ = rnd_report.build_report(kind="cycle", created_at="2026-10-08T17:44:00", signal_counts={"error": 8},
                                    prev_report=prev, error="Nebius rejected the API key. Check it was copied in full.",
                                    error_samples=[{"where": "slip_reader.image", "message": "Nebius rejected the API key.",
                                                    "count": 5}])
    assert "same signals still waiting" in md
    assert "Last cycle had" not in md
    assert "full.." not in md                                   # no double period
    assert "5× in slip_reader.image: Nebius rejected the API key." in md
    assert "Check your Nebius API key" in md                     # a concrete fix, first in next steps
    assert "| Signal from real use | Waiting to be studied |" in md


def test_finished_cycle_comparison(tmp_db):
    last_ok = {"kind": "cycle", "created_at": "2026-10-07T12:00:00+00:00", "data": {"signal_counts": {"feedback": 4}}}
    md, _ = rnd_report.build_report(kind="cycle", created_at="2026-10-08T17:44:00", signal_counts={"feedback": 2},
                                    prev_report=last_ok, last_finished_cycle=last_ok)
    assert "studied 4 signals; 2 new ones have come in since" in md


def test_status_report_shows_waiting_errors(tmp_db):
    db.log_event("error", {"where": "research_team", "message": "Nebius rejected the API key."})
    db.log_event("error", {"where": "research_team", "message": "Nebius rejected the API key."})
    rnd.write_report("status")
    md = db.latest_rnd_report()["markdown"]
    assert "2× in research_team" in md and "Check your Nebius API key" in md
