"""Player-prop stat keys and where to find them in ESPN box scores.

Each stat maps to a list of (group, label) lookups that are summed.
`group` is the ESPN statistics group name ("passing", "batting"...) or
None to search every group. Labels with "made-attempted" values like
"3-8" are read as the made number.
"""

PROP_STATS: dict[str, dict[str, list[tuple[str | None, str]]]] = {
    "NBA": {
        "points": [(None, "PTS")],
        "rebounds": [(None, "REB")],
        "assists": [(None, "AST")],
        "threes": [(None, "3PT")],
        "steals": [(None, "STL")],
        "blocks": [(None, "BLK")],
        "turnovers": [(None, "TO")],
        "points_rebounds_assists": [(None, "PTS"), (None, "REB"), (None, "AST")],
        "points_rebounds": [(None, "PTS"), (None, "REB")],
        "points_assists": [(None, "PTS"), (None, "AST")],
        "rebounds_assists": [(None, "REB"), (None, "AST")],
    },
    "NFL": {
        "passing_yards": [("passing", "YDS")],
        "passing_tds": [("passing", "TD")],
        "interceptions_thrown": [("passing", "INT")],
        "rushing_yards": [("rushing", "YDS")],
        "rushing_attempts": [("rushing", "CAR")],
        "receiving_yards": [("receiving", "YDS")],
        "receptions": [("receiving", "REC")],
        "rush_rec_yards": [("rushing", "YDS"), ("receiving", "YDS")],
        "anytime_td": [("rushing", "TD"), ("receiving", "TD")],
    },
    "MLB": {
        "hits": [("batting", "H")],
        "home_runs": [("batting", "HR")],
        "rbis": [("batting", "RBI")],
        "runs": [("batting", "R")],
        "hits_runs_rbis": [("batting", "H"), ("batting", "R"), ("batting", "RBI")],
        "batter_strikeouts": [("batting", "K")],
        "pitcher_strikeouts": [("pitching", "K")],
        "earned_runs": [("pitching", "ER")],
        "hits_allowed": [("pitching", "H")],
    },
    "NHL": {
        "goals": [(None, "G")],
        "assists": [(None, "A")],
        "points": [(None, "G"), (None, "A")],
        "shots": [(None, "S")],
        "saves": [(None, "SV")],
    },
}

SPORTS = list(PROP_STATS)
BET_TYPES = ["moneyline", "spread", "total", "player_prop"]


def stats_for(sport: str) -> list[str]:
    """Built-in stats plus any the R&D team has added."""
    from .learned import stat_lookups

    return list(dict.fromkeys(list(PROP_STATS.get(sport, {})) + list(stat_lookups(sport))))


def all_stat_keys() -> list[str]:
    keys: list[str] = []
    for sport in SPORTS:
        for k in stats_for(sport):
            if k not in keys:
                keys.append(k)
    return keys
