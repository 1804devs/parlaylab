"""Turn a bet-slip screenshot (or pasted text) into structured legs.

Screenshots go to the best vision model on your Nebius account (an NVIDIA
one if available); pasted text and the final structuring go to Nemotron.
"""
from __future__ import annotations

import base64
from datetime import date

from .nebius_client import chat, extract_json, pick_model
from .learned import slip_rules
from .stats_catalog import BET_TYPES, PROP_STATS, SPORTS, stats_for

LEG_FIELDS = [
    "sport", "type", "team", "opponent", "player", "stat",
    "side", "line", "odds", "game_date", "description",
]


def _instructions() -> str:
    stat_lines = "\n".join(f"  {s}: {', '.join(stats_for(s))}" for s in PROP_STATS)
    learned_rules = slip_rules()
    learned_block = ("\nLessons from past slips (follow these):\n" + "\n".join(f"- {r}" for r in learned_rules)
                     if learned_rules else "")
    return f"""You read sportsbook bet slips and return the parlay as JSON.
Today is {date.today().isoformat()}.

Return ONLY a JSON object, no commentary:
{{
  "book": "sportsbook name or null",
  "stake": number or null,
  "total_odds": American odds of the whole parlay as a number or null,
  "potential_payout": number or null,
  "legs": [
    {{
      "sport": one of {SPORTS},
      "type": one of {BET_TYPES},
      "team": "full team name the bet is on (moneyline/spread), or the player's team for props, or home team for totals",
      "opponent": "other team or null",
      "player": "player full name for player_prop, else null",
      "stat": "stat key for player_prop (list below), else null",
      "side": "over or under for totals and props, else null",
      "line": number (spread like -3.5, total like 47.5, prop like 24.5; null for moneyline),
      "odds": American odds for this leg as a number (e.g. -110, 145) or null,
      "game_date": "YYYY-MM-DD if shown, else null",
      "description": "the leg exactly as written on the slip"
    }}
  ]
}}

Allowed stat keys per sport:
{stat_lines}

Rules:
- "Anytime touchdown" = stat anytime_td, side over, line 0.5.
- "2+ hits" style = side over, line 1.5.
- Use full team names (e.g. "New York Knicks", not "NYK").
- If something is unreadable, use null rather than guessing.
{learned_block}"""


def _normalize(data: dict) -> dict:
    legs = []
    for raw in data.get("legs") or []:
        leg = {k: raw.get(k) for k in LEG_FIELDS}
        if isinstance(leg["sport"], str):
            leg["sport"] = leg["sport"].upper()
        if isinstance(leg["type"], str):
            leg["type"] = leg["type"].lower().replace(" ", "_")
        if isinstance(leg["side"], str):
            leg["side"] = leg["side"].lower()
        for num in ("line", "odds"):
            try:
                leg[num] = float(leg[num]) if leg[num] not in (None, "") else None
            except (TypeError, ValueError):
                leg[num] = None
        legs.append(leg)
    return {
        "book": data.get("book"),
        "stake": data.get("stake"),
        "total_odds": data.get("total_odds"),
        "potential_payout": data.get("potential_payout"),
        "legs": legs,
    }


TRANSCRIBE_PROMPT = (
    "This is a screenshot of a sportsbook bet slip. Copy out every bet on it as plain text, one leg per "
    "line, exactly as written: team or player, the bet (moneyline, spread, total, or player prop and stat), "
    "over/under, the line, the odds, and the game and date if shown. Then copy the sportsbook name, stake, "
    "total parlay odds and potential payout if shown. Write only what you can read; don't guess."
)


def _image_part(image_bytes: bytes, mime: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def read_slip_image(image_bytes: bytes, mime: str = "image/png", api_key: str | None = None) -> dict:
    """Screenshot → legs.

    A Nemotron vision model reads the slip straight to JSON. Any other vision model
    (e.g. NVIDIA Cosmos, Gemma) copies the slip out as text, and Nemotron turns that
    text into legs, so an NVIDIA model always does the understanding.
    """
    vision = pick_model("vision", api_key)
    if "nemotron" in vision.lower():  # Nemotron vision models read straight to JSON
        messages = [
            {"role": "system", "content": _instructions()},
            {"role": "user", "content": [{"type": "text", "text": "Read this bet slip and return the JSON."},
                                         _image_part(image_bytes, mime)]},
        ]
        result = _normalize(extract_json(chat(messages, model=vision, api_key=api_key, temperature=0.0,
                                                 think=False)))
        result["_models"] = {"vision": vision, "parser": vision}
        return result

    transcript = chat(
        [{"role": "user", "content": [{"type": "text", "text": TRANSCRIBE_PROMPT}, _image_part(image_bytes, mime)]}],
        model=vision, api_key=api_key, temperature=0.0, max_tokens=2000,
    )
    result = read_slip_text(transcript, api_key)
    result["_models"] = {"vision": vision, "parser": result["_models"]["parser"]}
    result["_transcript"] = transcript
    return result


def read_slip_text(slip_text: str, api_key: str | None = None) -> dict:
    parser = pick_model("agent", api_key)
    messages = [
        {"role": "system", "content": _instructions()},
        {"role": "user", "content": f"Bet slip text:\n\n{slip_text}"},
    ]
    # Turning text into JSON needs no step-by-step thinking: skip it (faster, cheaper, and
    # small reasoning models otherwise spend their whole budget thinking).
    result = _normalize(extract_json(chat(messages, model="agent", api_key=api_key,
                                          temperature=0.0, max_tokens=4000, think=False)))
    result["_models"] = {"vision": None, "parser": parser}
    return result
