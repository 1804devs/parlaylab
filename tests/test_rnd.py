"""Tests for learned rules and the R&D team's safety machinery."""
import json
import os
import shutil
from pathlib import Path

import pytest

from parlaylab import db, grader, learned, rnd, slip_reader
from parlaylab.scores import parse_boxscore, player_matches, team_matches

IN_SANDBOX = bool(os.getenv("PARLAY_IN_SANDBOX"))
NETS = {"displayName": "Brooklyn Nets", "name": "Nets", "location": "Brooklyn", "abbreviation": "BKN"}


@pytest.fixture
def tmp_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("PARLAY_LEARNED", str(tmp_path / "learned.json"))
    learned._cache["data"] = None
    yield tmp_path / "learned.json"
    learned._cache["data"] = None


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))


# ---------------- learned rules ----------------
def test_validate_rejects_bad_rules():
    assert learned.validate({"team_aliases": {"bk": "Brooklyn Nets"}}) == []
    assert learned.validate({"stat_labels": {"NBA": {"dd": [[None, "DD"]]}}}) == []
    assert learned.validate({"hack": 1})
    assert learned.validate({"stat_labels": {"CRICKET": {"x": [[None, "X"]]}}})
    assert learned.validate({"stat_labels": {"NBA": {"x": "PTS"}}})
    assert learned.validate({"slip_rules": ["x" * 500]})


def test_aliases_change_matching(tmp_rules):
    assert not team_matches("Bklyn", NETS)
    assert not player_matches("Shai", "Shai Gilgeous-Alexander")
    learned.save(learned.merge(learned.load(), {"team_aliases": {"Bklyn": "Brooklyn Nets"},
                                                "player_aliases": {"SHAI": "Shai Gilgeous-Alexander"}}))
    assert team_matches("bklyn", NETS)
    assert player_matches("Shai", "Shai Gilgeous-Alexander")


def test_learned_stat_label_grades_new_prop(tmp_rules):
    summary = {"boxscore": {"players": [{"team": {"displayName": "Brooklyn Nets"}, "statistics": [
        {"labels": ["PTS", "REB", "OREB"], "athletes": [{"athlete": {"displayName": "Nic Claxton"},
                                                          "stats": ["10", "12", "5"]}]}]}]}}
    leg = {"sport": "NBA", "type": "player_prop", "player": "Nic Claxton", "stat": "offensive_rebounds",
           "side": "over", "line": 3.5}
    game = {"name": "x", "state": "post", "detail": "Final"}
    assert grader.grade_prop_leg(leg, game, parse_boxscore(summary))["result"] == "not_found"
    learned.save(learned.merge(learned.load(), {"stat_labels": {"NBA": {"offensive_rebounds": [[None, "OREB"]]}}}))
    assert grader.grade_prop_leg(leg, game, parse_boxscore(summary))["result"] == "won"


def test_slip_rules_reach_the_prompt(tmp_rules):
    learned.save(learned.merge(learned.load(), {"slip_rules": ["Hard Rock lists odds under the leg"]}))
    assert "Hard Rock lists odds under the leg" in slip_reader._instructions()


# ---------------- signals ----------------
def test_diff_legs_finds_user_fixes():
    ai = [{"team": "Nets", "line": -3.5, "odds": -110, "description": "BKN -3.5"}]
    saved = [{"team": "Brooklyn Nets", "line": -3.5, "odds": -110.0}, {"team": "Knicks"}]
    d = rnd.diff_legs(ai, saved)
    assert [c["field"] for c in d["changes"]] == ["team"]
    assert d["legs_added_by_user"] == 1


def test_collect_signals(tmp_db):
    db.log_event("leg_not_found", {"leg": {"sport": "NBA", "team": "Bklyn"}, "detail": "No game"})
    db.log_event("feedback", {"area": "ui", "text": "dark mode please"})
    sig = rnd.collect_signals(db.list_events())
    assert sig["not_found_legs"][0]["team"] == "Bklyn"
    assert sig["feedback"][0]["text"] == "dark mode please"
    assert rnd.signal_count(sig) == 2


# ---------------- guardrails ----------------
def test_check_edits_guardrails():
    ok_find = "def american_to_decimal(odds: float) -> float:"
    assert rnd.check_edits([{"file": "parlaylab/odds.py", "find": ok_find, "replace": ok_find}]) == []
    assert rnd.check_edits([{"file": "tests/test_rnd_new_thing.py", "find": "", "replace": "x = 1\n"}]) == []
    bad = [
        {"file": "parlaylab/rnd.py", "find": "import", "replace": ""},          # protected
        {"file": "tests/test_core.py", "find": "import", "replace": ""},        # can't weaken tests
        {"file": "../etc/passwd", "find": "", "replace": "x"},                  # path escape
        {"file": "parlaylab/odds.py", "find": "return", "replace": "pass"},     # not unique
        {"file": "parlaylab/odds.py", "find": "", "replace": "x"},              # append to non-test
        {"file": "parlaylab/odds.py", "find": "nope not here", "replace": ""},  # missing
    ]
    for e in bad:
        assert rnd.check_edits([e]), e


def test_make_diff_shows_change():
    find = "def implied_probability(odds: float) -> float:"
    diff = rnd.make_diff([{"file": "parlaylab/odds.py", "find": find, "replace": find + "  # tweak"}],
                         {"slip_rules": ["x"]})
    assert "+def implied_probability(odds: float) -> float:  # tweak" in diff
    assert "learned rules" in diff


# ---------------- apply / undo on a copy of the project ----------------
@pytest.fixture
def fake_project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    shutil.copytree(rnd.PROJECT, root, ignore=shutil.ignore_patterns("*.db", ".rnd_backups", "__pycache__",
                                                                      ".pytest_cache", "*.zip"))
    monkeypatch.setattr(rnd, "PROJECT", root)
    monkeypatch.setattr(rnd, "BACKUPS", root / ".rnd_backups")
    monkeypatch.setenv("PARLAY_LEARNED", str(root / "parlaylab" / "learned.json"))
    learned._cache["data"] = None
    yield root
    learned._cache["data"] = None


def _proposal(edits, rules=None):
    return {"id": 7, "data": {"edits": edits, "rule_changes": rules or {}}}


def test_apply_and_undo(fake_project, monkeypatch):
    monkeypatch.setattr(rnd, "_run_tests", lambda root, lp: {"passed": True, "summary": "ok", "output": ""})
    find = "def implied_probability(odds: float) -> float:"
    original = (fake_project / "parlaylab/odds.py").read_text()
    lp = fake_project / "parlaylab/learned.json"
    rules_before = lp.read_text() if lp.exists() else None
    prop = _proposal([
        {"file": "parlaylab/odds.py", "find": find, "replace": "def is_plus_money(o):\n    return o > 0\n\n\n" + find},
        {"file": "tests/test_rnd_plus_money.py", "find": "", "replace": "def test_x():\n    assert True\n"},
    ], {"team_aliases": {"Bklyn": "Brooklyn Nets"}})
    res = rnd.apply_proposal(prop)
    assert res["ok"], res
    assert "is_plus_money" in (fake_project / "parlaylab/odds.py").read_text()
    assert (fake_project / "tests/test_rnd_plus_money.py").exists()
    assert learned.load()["team_aliases"]["bklyn"] == "Brooklyn Nets"

    prop["data"]["apply_result"] = res
    assert rnd.undo_proposal(prop)["ok"]
    assert (fake_project / "parlaylab/odds.py").read_text() == original
    assert not (fake_project / "tests/test_rnd_plus_money.py").exists()
    assert (lp.read_text() if lp.exists() else None) == rules_before


def test_apply_rolls_back_when_tests_fail(fake_project, monkeypatch):
    monkeypatch.setattr(rnd, "_run_tests", lambda root, lp: {"passed": False, "summary": "1 failed", "output": "x"})
    find = "def implied_probability(odds: float) -> float:"
    original = (fake_project / "parlaylab/odds.py").read_text()
    res = rnd.apply_proposal(_proposal([{"file": "parlaylab/odds.py", "find": find, "replace": find + " # x"}]))
    assert not res["ok"]
    assert (fake_project / "parlaylab/odds.py").read_text() == original


def test_apply_refuses_protected_file(fake_project):
    res = rnd.apply_proposal(_proposal([{"file": "parlaylab/db.py", "find": "import json", "replace": ""}]))
    assert not res["ok"] and "not editable" in res["message"]


# ---------------- real sandbox + full cycle with a scripted model ----------------
pytestmark_sandbox = pytest.mark.skipif(IN_SANDBOX, reason="already inside an R&D sandbox run")


@pytestmark_sandbox
def test_sandbox_catches_broken_code():
    find = "def implied_probability(odds: float) -> float:"
    good = rnd.sandbox_test([{"file": "parlaylab/odds.py", "find": find, "replace": find + "  # ok"}], {})
    assert good["passed"], good["output"]
    bad = rnd.sandbox_test([{"file": "parlaylab/odds.py", "find": find, "replace": "def implied_probability(:"}], {})
    assert not bad["passed"]
    broken_logic = rnd.sandbox_test([{"file": "parlaylab/odds.py",
                                      "find": "    return 1 / american_to_decimal(odds)",
                                      "replace": "    return 0.5"}], {})
    assert not broken_logic["passed"]


@pytestmark_sandbox
def test_full_cycle_with_scripted_agents(tmp_db, tmp_rules, monkeypatch):
    db.log_event("leg_not_found", {"leg": {"sport": "NBA", "type": "moneyline", "team": "Bklyn"},
                                   "detail": "No NBA game found for Bklyn"})
    tries = {"engineer": 0}

    def fake_chat(messages, model=None, api_key=None, **kw):
        role = messages[0]["content"].split("Your role: ")[1]
        if role == "QA Analyst":
            return json.dumps({"issues": [{"title": "Bklyn not matched", "area": "tracking", "severity": "high",
                                           "evidence": "1 leg", "fix_hint": "alias"}]})
        if role == "Product Researcher":
            return '```json\n{"ideas": []}\n```'
        if role == "R&D Lead":
            return json.dumps({"plan": [{"title": "Teach Bklyn alias", "area": "tracking", "kind": "rule",
                                         "why": "unmatched leg", "task": "add alias"}]})
        if role == "Engineer":
            tries["engineer"] += 1
            if tries["engineer"] == 1:  # first try breaks a guardrail, gets sent back
                return json.dumps({"summary": "x", "rule_changes": {},
                                   "edits": [{"file": "parlaylab/db.py", "find": "import json", "replace": ""}]})
            return json.dumps({"summary": "Bklyn now means Brooklyn Nets",
                               "rule_changes": {"team_aliases": {"Bklyn": "Brooklyn Nets"}}, "edits": []})
        if role == "Reviewer":
            return json.dumps({"risk": "low", "recommend": "approve", "notes": "Data-only change."})
        raise AssertionError(role)

    monkeypatch.setattr(rnd, "chat", fake_chat)
    out = rnd.run_rnd_cycle(goal="", max_items=2)
    assert len(out["proposals"]) == 1
    prop = db.get_proposal(out["proposals"][0])
    assert prop["status"] == "ready", prop["data"]
    assert prop["data"]["attempts"] == 2
    assert prop["data"]["tests"]["passed"]
    assert prop["data"]["review"]["recommend"] == "approve"
    assert db.event_counts(only_new=True) == {}
    report = db.latest_rnd_report()
    assert report["id"] == out["report"]
    assert "1 proposal: 1 ready for your review, 0 didn't pass" in report["markdown"]
