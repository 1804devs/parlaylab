import pytest

from parlaylab import grader, odds
from parlaylab.nebius_client import extract_json, strip_thinking
from parlaylab.scores import parse_boxscore, parse_event, player_matches, team_matches

KNICKS = {"displayName": "New York Knicks", "shortDisplayName": "Knicks", "name": "Knicks",
          "location": "New York", "abbreviation": "NY"}
CELTICS = {"displayName": "Boston Celtics", "shortDisplayName": "Celtics", "name": "Celtics",
           "location": "Boston", "abbreviation": "BOS"}
RED_SOX = {"displayName": "Boston Red Sox", "name": "Red Sox", "location": "Boston", "abbreviation": "BOS"}


def game(state, home_score, away_score):
    return {"id": "1", "name": "BOS @ NY", "state": state, "detail": "Q4 2:00",
            "home": {"team": KNICKS, "score": home_score},
            "away": {"team": CELTICS, "score": away_score}}


# ---- name matching ----
@pytest.mark.parametrize("q", ["New York Knicks", "Knicks", "NY Knicks", "ny", "NEW YORK KNICKS"])
def test_team_matches(q):
    assert team_matches(q, KNICKS)


def test_team_no_false_match():
    assert not team_matches("Boston Celtics", KNICKS)
    assert team_matches("Red Sox", RED_SOX)
    assert not team_matches("White Sox", RED_SOX)


def test_player_matches():
    assert player_matches("Jalen Brunson", "Jalen Brunson")
    assert player_matches("J. Brunson", "Jalen Brunson")
    assert player_matches("Ronald Acuña Jr.", "Ronald Acuna Jr.")
    assert not player_matches("Josh Hart", "Jalen Brunson")


# ---- game legs ----
def test_moneyline():
    leg = {"type": "moneyline", "team": "Knicks"}
    assert grader.grade_game_leg(leg, game("post", 110, 100))["result"] == "won"
    assert grader.grade_game_leg(leg, game("post", 99, 100))["result"] == "lost"
    assert grader.grade_game_leg(leg, game("in", 50, 40))["result"] == "winning"
    assert grader.grade_game_leg(leg, game("pre", 0, 0))["result"] == "pending"


def test_spread():
    fav = {"type": "spread", "team": "Knicks", "line": -5.5}
    dog = {"type": "spread", "team": "Celtics", "line": 5.5}
    assert grader.grade_game_leg(fav, game("post", 110, 105))["result"] == "lost"
    assert grader.grade_game_leg(dog, game("post", 110, 105))["result"] == "won"
    assert grader.grade_game_leg(fav, game("post", 110, 104))["result"] == "won"
    push = {"type": "spread", "team": "Knicks", "line": -6}
    assert grader.grade_game_leg(push, game("post", 110, 104))["result"] == "push"


def test_total():
    over = {"type": "total", "side": "over", "line": 210.5}
    under = {"type": "total", "side": "under", "line": 210.5}
    assert grader.grade_game_leg(over, game("in", 110, 101))["result"] == "won"   # cashed early
    assert grader.grade_game_leg(under, game("in", 110, 101))["result"] == "lost"  # dead early
    assert grader.grade_game_leg(over, game("post", 100, 100))["result"] == "lost"
    assert grader.grade_game_leg(under, game("post", 100, 100))["result"] == "won"
    assert grader.grade_game_leg(under, game("in", 50, 50))["result"] == "winning"


# ---- props ----
NBA_SUMMARY = {"boxscore": {"players": [{
    "team": {"displayName": "New York Knicks"},
    "statistics": [{"labels": ["MIN", "FG", "3PT", "REB", "AST", "PTS"],
                    "athletes": [{"athlete": {"displayName": "Jalen Brunson"},
                                  "stats": ["36", "11-22", "4-9", "3", "8", "31"]}]}]}]}}

NFL_SUMMARY = {"boxscore": {"players": [{
    "team": {"displayName": "New York Giants"},
    "statistics": [
        {"name": "rushing", "labels": ["CAR", "YDS", "AVG", "TD", "LONG"],
         "athletes": [{"athlete": {"displayName": "Tyrone Tracy Jr."}, "stats": ["15", "62", "4.1", "0", "18"]}]},
        {"name": "receiving", "labels": ["REC", "YDS", "AVG", "TD", "LONG", "TGTS"],
         "athletes": [{"athlete": {"displayName": "Tyrone Tracy Jr."}, "stats": ["3", "21", "7.0", "1", "11", "4"]}]},
    ]}]}}


def test_nba_props():
    players = parse_boxscore(NBA_SUMMARY)
    g = game("in", 80, 70)
    pts = {"sport": "NBA", "type": "player_prop", "player": "Jalen Brunson", "stat": "points", "side": "over", "line": 27.5}
    assert grader.grade_prop_leg(pts, g, players)["result"] == "won"
    threes = {**pts, "stat": "threes", "line": 4.5}
    r = grader.grade_prop_leg(threes, g, players)
    assert r["result"] == "losing" and r["current"] == 4
    pra = {**pts, "stat": "points_rebounds_assists", "side": "under", "line": 40.5}
    assert grader.grade_prop_leg(pra, g, players)["result"] == "lost"  # 31+3+8 = 42


def test_nfl_props():
    players = parse_boxscore(NFL_SUMMARY)
    g = game("post", 20, 17)
    td = {"sport": "NFL", "type": "player_prop", "player": "Tyrone Tracy", "stat": "anytime_td", "side": "over", "line": 0.5}
    assert grader.grade_prop_leg(td, g, players)["result"] == "won"
    rr = {**td, "stat": "rush_rec_yards", "line": 85.5}
    r = grader.grade_prop_leg(rr, g, players)
    assert r["result"] == "lost" and r["current"] == 83


def test_missing_player():
    leg = {"sport": "NBA", "type": "player_prop", "player": "Nobody", "stat": "points", "side": "over", "line": 10.5}
    assert grader.grade_prop_leg(leg, game("post", 1, 0), parse_boxscore(NBA_SUMMARY))["result"] == "not_found"


# ---- parlay & grade_leg wiring ----
def test_parlay_status():
    assert grader.parlay_status(["won", "lost", "winning"]) == "lost"
    assert grader.parlay_status(["won", "push", "won"]) == "won"
    assert grader.parlay_status(["won", "pending"]) == "live"
    assert grader.parlay_status(["pending", "pending"]) == "pending"


def test_grade_leg_with_fakes():
    leg = {"sport": "NBA", "type": "player_prop", "team": "Knicks", "player": "Jalen Brunson",
           "stat": "assists", "side": "over", "line": 6.5}
    r = grader.grade_leg(leg, find_game=lambda l: game("post", 100, 90),
                         get_box=lambda s, e: parse_boxscore(NBA_SUMMARY))
    assert r["result"] == "won"
    assert grader.grade_leg(leg, find_game=lambda l: None)["result"] == "not_found"


def test_parse_event():
    ev = {"id": "9", "shortName": "BOS @ NY", "status": {"type": {"state": "in", "shortDetail": "Q3"}},
          "competitions": [{"competitors": [
              {"homeAway": "home", "score": "61", "team": KNICKS},
              {"homeAway": "away", "score": "58", "team": CELTICS}]}]}
    e = parse_event(ev)
    assert e["state"] == "in" and e["home"]["score"] == 61 and e["away"]["team"]["abbreviation"] == "BOS"


# ---- odds & json ----
def test_odds():
    assert odds.american_to_decimal(150) == 2.5
    assert round(odds.american_to_decimal(-110), 4) == 1.9091
    s = odds.parlay_summary([-110, -110, 150], 10)
    assert s["payout"] == round(10 * 1.9091 * 1.9091 * 2.5, 1) or abs(s["payout"] - 91.12) < 0.05
    assert round(odds.implied_probability(-110), 4) == 0.5238
    assert round(odds.implied_probability(150), 4) == 0.4
    assert s["leg_probs"] == [0.5238, 0.5238, 0.4]
    assert odds.decimal_to_american(2.5) == 150
    assert odds.decimal_to_american(1.5) == -200


def test_extract_json():
    assert extract_json('<think>hmm {"a":0}</think>Here: ```json\n{"legs": []}\n```') == {"legs": []}
    assert extract_json('Sure! {"x": 1} done') == {"x": 1}
    assert strip_thinking("reasoning...</think>answer") == "answer"
