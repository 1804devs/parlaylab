"""Sharing slips from Hard Rock Bet: several screenshots, shared text, shared links."""
import json

import pytest

from parlaylab import nebius_client, slip_reader
from parlaylab.slip_reader import ShareError, merge_slips, read_shared

LEG_A = {"sport": "NFL", "type": "spread", "team": "New York Giants", "line": 3.5, "odds": -110}
LEG_B = {"sport": "NFL", "type": "player_prop", "player": "Malik Nabers", "stat": "receiving_yards",
         "side": "over", "line": 64.5, "odds": -115}
LEG_C = {"sport": "NFL", "type": "total", "team": "New York Giants", "side": "under", "line": 44.5, "odds": -105}


def test_one_long_slip_in_parts_keeps_each_leg_once():
    top = {"book": "Hard Rock Bet", "stake": 10, "total_odds": None, "legs": [LEG_A, LEG_B]}
    bottom = {"book": None, "stake": None, "total_odds": 595, "legs": [dict(LEG_B), LEG_C]}  # B overlaps
    merged = merge_slips([top, bottom])
    assert [l.get("player") or l["type"] for l in merged["legs"]] == ["spread", "Malik Nabers", "total"]
    assert merged["stake"] == 10 and merged["total_odds"] == 595 and merged["book"] == "Hard Rock Bet"


def _fake_text_reader(calls):
    def fake(text, api_key=None, book=None):
        calls.append({"text": text, "book": book})
        return {"book": book, "stake": None, "total_odds": None, "potential_payout": None, "legs": [LEG_A]}
    return fake


def test_shared_text_is_read_with_hard_rock_hint(monkeypatch):
    calls = []
    monkeypatch.setattr(slip_reader, "read_slip_text", _fake_text_reader(calls))
    out = read_shared("My Hard Rock parlay: Giants +3.5 (-110), Nabers o64.5 rec yds (-115)", "key")
    assert calls[0]["book"] == "Hard Rock Bet" and "Giants +3.5" in calls[0]["text"]
    assert out["_shared_from"] == "text"


def test_hard_rock_link_is_opened_when_it_shows_the_bets(monkeypatch):
    calls, fetched = [], []
    monkeypatch.setattr(slip_reader, "read_slip_text", _fake_text_reader(calls))

    def fake_fetch(url):
        fetched.append(url)
        return "Parlay 3 legs New York Giants +3.5 -110 Malik Nabers Over 64.5 Receiving Yards -115"

    out = read_shared("Check out my bet! https://share.hardrock.bet/b/abc123", "key", fetch=fake_fetch)
    assert fetched == ["https://share.hardrock.bet/b/abc123"]
    assert "Malik Nabers Over 64.5" in calls[0]["text"] and out["_shared_from"] == "link"


def test_link_that_needs_sign_in_gives_a_clear_message(monkeypatch):
    monkeypatch.setattr(slip_reader, "read_slip_text", _fake_text_reader([]))
    with pytest.raises(ShareError, match="sign in"):
        read_shared("https://share.hardrock.bet/b/abc123", "key", fetch=lambda url: "Log in to Hard Rock Bet")


def test_other_sites_are_never_fetched(monkeypatch):
    fetched = []
    monkeypatch.setattr(slip_reader, "read_slip_text", _fake_text_reader([]))
    with pytest.raises(ShareError):
        read_shared("http://169.254.169.254/latest https://evil.example/hardrock.bet", "key",
                    fetch=lambda url: fetched.append(url) or "Giants +3.5 -110")
    assert fetched == []


def test_text_without_odds_is_not_sent_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(slip_reader, "read_slip_text", _fake_text_reader(calls))
    with pytest.raises(ShareError, match="doesn't look like a bet slip"):
        read_shared("hey look at this", "key")
    assert calls == []


def test_sportsbook_hint_reaches_the_screenshot_reader(monkeypatch):
    monkeypatch.setattr(nebius_client, "list_models",
                        lambda api_key=None, refresh=False: ["google/gemma-3-27b-it", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"])
    seen = []

    def fake_chat(messages, model, api_key=None, **kw):
        seen.append(json.dumps(messages))
        if model.startswith("google"):
            return "Giants +3.5 -110"
        return json.dumps({"book": None, "legs": [LEG_A]})

    monkeypatch.setattr(slip_reader, "chat", fake_chat)
    out = slip_reader.read_slip_image(b"png", "image/png", "key", book="Hard Rock Bet")
    assert "This slip is from Hard Rock Bet." in seen[0] and "This slip is from Hard Rock Bet." in seen[1]
    assert out["book"] == "Hard Rock Bet"
