"""Each leg is matched to its real game right after reading (ESPN schedule, no credits)."""
import pytest

from parlaylab import enrich
from parlaylab.enrich import NOT_FOUND, enrich_leg, enrich_slips, unmatched

KNICKS = {"displayName": "New York Knicks", "name": "Knicks", "abbreviation": "NY", "shortDisplayName": "Knicks"}
CELTICS = {"displayName": "Boston Celtics", "name": "Celtics", "abbreviation": "BOS", "shortDisplayName": "Celtics"}


def game(state="pre", date="2026-10-21T23:30Z", hs=0, as_=0, detail="Tue, October 21th at 7:30 PM EDT"):
    return {"id": "1", "name": "BOS @ NY", "state": state, "detail": detail, "date": date,
            "home": {"team": KNICKS, "score": hs}, "away": {"team": CELTICS, "score": as_}}


@pytest.fixture(autouse=True)
def new_york(monkeypatch):
    monkeypatch.setenv("PARLAYLAB_TZ", "America/New_York")


def test_fills_opponent_date_full_team_name_and_game_line():
    leg = {"sport": "NBA", "type": "spread", "team": "NYK", "line": -4.5, "odds": -110}
    out = enrich_leg(leg, find=lambda l: game())
    assert out["opponent"] == "Boston Celtics"
    assert out["team"] == "New York Knicks"                # abbreviation → full name
    assert out["game_date"] == "2026-10-21"                # 23:30 UTC = 7:30 PM in New York, same day
    assert out["game"] == "BOS @ NY · Wed Oct 21, 7:30 PM"
    assert leg["team"] == "NYK"                            # original not changed


def test_late_game_date_uses_your_timezone_not_utc():
    out = enrich_leg({"sport": "NBA", "type": "moneyline", "team": "Knicks"},
                     find=lambda l: game(date="2026-10-22T02:00Z"))   # 10 PM ET on the 21st
    assert out["game_date"] == "2026-10-21"


def test_keeps_what_the_slip_already_said():
    leg = {"sport": "NBA", "type": "moneyline", "team": "Knicks", "opponent": "Celtics", "game_date": "2026-10-21"}
    out = enrich_leg(leg, find=lambda l: game())
    assert out["opponent"] == "Celtics" and out["game_date"] == "2026-10-21"


def test_live_game_shows_the_score():
    out = enrich_leg({"sport": "NBA", "type": "moneyline", "team": "Knicks"},
                     find=lambda l: game(state="in", hs=61, as_=58, detail="Q3 5:12"))
    assert out["game"] == "BOS 58-61 NY · Q3 5:12"


def test_totals_keep_the_team_as_given():
    out = enrich_leg({"sport": "NBA", "type": "total", "team": "Knicks", "side": "over", "line": 220.5},
                     find=lambda l: game())
    assert out["team"] == "Knicks" and out["game"].startswith("BOS @ NY")


def test_not_found_and_errors_are_flagged_not_hidden():
    assert enrich_leg({"sport": "NBA", "type": "moneyline", "team": "Nobody"}, find=lambda l: None)["game"] == NOT_FOUND

    def boom(l):
        raise RuntimeError("ESPN down")
    assert enrich_leg({"sport": "NBA", "type": "moneyline", "team": "Knicks"}, find=boom)["game"] == NOT_FOUND
    no_team = enrich_leg({"sport": "NBA", "type": "player_prop", "player": "Jalen Brunson"}, find=lambda l: game())
    assert "add the player's team" in no_team["game"]


def test_enrich_slips_matches_every_leg_in_place():
    slips = [{"legs": [{"sport": "NBA", "type": "moneyline", "team": "Knicks"},
                       {"sport": "NBA", "type": "moneyline", "team": "Nobody"}]},
             {"legs": [{"sport": "NBA", "type": "spread", "team": "Celtics", "line": 4.5}]}]
    enrich_slips(slips, find=lambda l: game() if l["team"] != "Nobody" else None)
    assert slips[0]["legs"][0]["opponent"] == "Boston Celtics"
    assert slips[1]["legs"][0]["opponent"] == "New York Knicks"
    assert unmatched(slips[0]) == 1 and unmatched(slips[1]) == 0
