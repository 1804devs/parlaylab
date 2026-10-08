"""American-odds math for parlays."""
from __future__ import annotations

from math import prod


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds can't be 0")
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def decimal_to_american(dec: float) -> int:
    if dec <= 1:
        raise ValueError("Decimal odds must be above 1")
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def implied_probability(odds: float) -> float:
    return 1 / american_to_decimal(odds)


def parlay_decimal(leg_odds: list[float]) -> float:
    return prod(american_to_decimal(o) for o in leg_odds)


def parlay_summary(leg_odds: list[float], stake: float) -> dict:
    """Combined odds, payout and the book's implied chance of hitting."""
    dec = parlay_decimal(leg_odds)
    return {
        "decimal": round(dec, 3),
        "american": decimal_to_american(dec),
        "payout": round(stake * dec, 2),
        "profit": round(stake * (dec - 1), 2),
        "implied_prob": round(1 / dec, 4),
        "leg_probs": [round(implied_probability(o), 4) for o in leg_odds],
    }
