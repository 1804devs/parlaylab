"""Turn a bet-slip screenshot (or pasted text) into structured legs.

Screenshots go to the best vision model on your Nebius account (an NVIDIA
one if available); pasted text and the final structuring go to Nemotron.
"""
from __future__ import annotations

import base64
import re
from datetime import date
from urllib.parse import urlparse

import requests

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


def _book_hint(book: str | None) -> str:
    return f" This slip is from {book}." if book else ""


def read_slip_image(image_bytes: bytes, mime: str = "image/png", api_key: str | None = None,
                    book: str | None = None) -> dict:
    """Screenshot → legs.

    A Nemotron vision model reads the slip straight to JSON. Any other vision model
    (e.g. NVIDIA Cosmos, Gemma) copies the slip out as text, and Nemotron turns that
    text into legs, so an NVIDIA model always does the understanding.
    """
    vision = pick_model("vision", api_key)
    if "nemotron" in vision.lower():  # Nemotron vision models read straight to JSON
        messages = [
            {"role": "system", "content": _instructions()},
            {"role": "user", "content": [{"type": "text",
                                          "text": "Read this bet slip and return the JSON." + _book_hint(book)},
                                         _image_part(image_bytes, mime)]},
        ]
        result = _normalize(extract_json(chat(messages, model=vision, api_key=api_key, temperature=0.0,
                                                 think=False)))
        result["_models"] = {"vision": vision, "parser": vision}
        return _with_book(result, book)

    transcript = chat(
        [{"role": "user", "content": [{"type": "text", "text": TRANSCRIBE_PROMPT + _book_hint(book)},
                                      _image_part(image_bytes, mime)]}],
        model=vision, api_key=api_key, temperature=0.0, max_tokens=2000,
    )
    result = read_slip_text(transcript, api_key, book=book)
    result["_models"] = {"vision": vision, "parser": result["_models"]["parser"]}
    result["_transcript"] = transcript
    return result


def read_slip_text(slip_text: str, api_key: str | None = None, book: str | None = None) -> dict:
    parser = pick_model("agent", api_key)
    messages = [
        {"role": "system", "content": _instructions()},
        {"role": "user", "content": f"Bet slip text:{_book_hint(book)}\n\n{slip_text}"},
    ]
    # Turning text into JSON needs no step-by-step thinking: skip it (faster, cheaper, and
    # small reasoning models otherwise spend their whole budget thinking).
    result = _normalize(extract_json(chat(messages, model="agent", api_key=api_key,
                                          temperature=0.0, max_tokens=4000, think=False)))
    result["_models"] = {"vision": None, "parser": parser}
    return _with_book(result, book)


def _with_book(result: dict, book: str | None) -> dict:
    if book and not result.get("book"):
        result["book"] = book
    return result


# ---------------- several screenshots of one slip ----------------

def _leg_key(leg: dict) -> tuple:
    def n(v):
        return str(v).strip().lower() if v is not None else ""
    return tuple(n(leg.get(f)) for f in ("type", "team", "player", "stat", "side", "line"))


def merge_slips(results: list[dict]) -> dict:
    """One long slip screenshotted in parts → one slip. Legs that appear in two
    overlapping screenshots are kept once."""
    merged = {"book": None, "stake": None, "total_odds": None, "potential_payout": None, "legs": []}
    seen = set()
    for r in results:
        for f in ("book", "stake", "total_odds", "potential_payout"):
            if merged[f] is None and r.get(f) is not None:
                merged[f] = r[f]
        for leg in r.get("legs") or []:
            k = _leg_key(leg)
            if k not in seen:
                seen.add(k)
                merged["legs"].append(leg)
    merged["_models"] = (results[0].get("_models") if results else None)
    return merged


# ---------------- what a sportsbook's Share button gives you ----------------

# Only these sites are fetched. Anything else is treated as plain text.
SHARE_DOMAINS = ("hardrock.bet", "hardrocksportsbook.com", "hardrockbet.com")
_URL = re.compile(r"https?://[^\s<>\"']+")


class ShareError(RuntimeError):
    pass


def _allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in SHARE_DOMAINS)


def fetch_share_page(url: str, timeout: int = 10) -> str:
    """Visible text of a shared-bet page, or '' if it can't be read without signing in."""
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 ParlayLab"})
    resp.raise_for_status()
    html = resp.text
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    return " ".join(text.split())


def _looks_like_bets(text: str) -> bool:
    """Real slip text has American odds (+150, -110) or lines (o47.5, +3.5)."""
    return len(re.findall(r"(?<![\w.])[+-]\d{3,4}(?!\d)|\b[ou]\d+(\.5)?\b|[+-]\d+\.5", text, flags=re.I)) >= 1


# ---------------- Google Photos / Google Drive share links ----------------
# Sharing a screenshot "via Google" gives a link to the picture, not a sportsbook page.
GOOGLE_PAGE_HOSTS = ("photos.app.goo.gl", "goo.gl", "photos.google.com", "drive.google.com")
GOOGLE_IMAGE_HOSTS = ("googleusercontent.com", "drive.google.com", "drive.usercontent.google.com")
MAX_IMAGE_BYTES = 15 * 1024 * 1024


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _host_in(url: str, hosts) -> bool:
    h = _host(url)
    return any(h == d or h.endswith("." + d) for d in hosts)


def is_google_link(url: str) -> bool:
    return _host_in(url, GOOGLE_PAGE_HOSTS)


def _get(url: str, timeout: int = 15):
    return requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 ParlayLab"}, allow_redirects=True)


def _image_from_response(resp) -> tuple[bytes, str] | None:
    mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if resp.ok and mime.startswith("image/") and 0 < len(resp.content) <= MAX_IMAGE_BYTES:
        return resp.content, mime
    return None


def fetch_google_image(url: str, get=_get) -> tuple[bytes, str]:
    """Download the picture behind a Google Photos or Google Drive share link."""
    # Google Drive: /file/d/<id>/... or ?id=<id>
    m = re.search(r"/file/d/([\w-]{10,})", url) or re.search(r"[?&]id=([\w-]{10,})", url)
    if _host_in(url, ("drive.google.com",)) and m:
        resp = get(f"https://drive.google.com/uc?export=download&id={m.group(1)}")
        img = _image_from_response(resp)
        if img:
            return img
        raise ShareError("Google Drive didn't hand over the picture. In Drive, set the file's sharing to "
                         "'Anyone with the link', or download the screenshot and upload it here.")

    # Google Photos: the share page names the picture in its og:image tag.
    resp = get(url)
    if not _host_in(resp.url or url, GOOGLE_PAGE_HOSTS):
        raise ShareError("That Google link went somewhere other than Google Photos or Drive, so it wasn't opened.")
    page = resp.text or ""
    found = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', page) \
        or re.search(r'(https://lh\d\.googleusercontent\.com/[\w\-./=%]+)', page)
    if not found:
        raise ShareError("That Google Photos link didn't show a picture. Make sure it's shared with a link "
                         "(not only with people), or save the screenshot and upload it here.")
    img_url = found.group(1).replace("&amp;", "&")
    if not _host_in(img_url, GOOGLE_IMAGE_HOSTS):
        raise ShareError("The picture in that Google link isn't on a Google address, so it wasn't downloaded.")
    img_url = re.sub(r"=[wsh]\d+[^/?]*$", "", img_url) + "=w2048"  # ask for a sharp full-width copy
    img = _image_from_response(get(img_url))
    if not img:
        raise ShareError("Couldn't download the picture from Google Photos. Save the screenshot and upload it here.")
    return img


def hardrock_betslip_ids(url: str) -> list[str] | None:
    """IDs in a Hard Rock share link like share.hardrock.bet/...?deep_link_value=hardrock://betslip/1,2,3."""
    if not _allowed(url):
        return None
    m = re.search(r"betslip(?:/|%2F)([\d,%2C]+)", url, flags=re.I)
    if not m:
        return None
    return [x for x in re.split(r",|%2C", m.group(1), flags=re.I) if x]


def read_shared(shared: str, api_key: str | None = None, book: str | None = "Hard Rock Bet",
                fetch=fetch_share_page, fetch_image=fetch_google_image) -> dict:
    """Whatever the Share button gave you: text, a sportsbook link, or a Google Photos/Drive link."""
    shared = (shared or "").strip()
    urls = _URL.findall(shared)
    text_part = _URL.sub(" ", shared).strip()

    # 1) A Google Photos / Drive link → it's a picture of the slip: read it like an upload.
    google = [u for u in urls if is_google_link(u)]
    if google:
        image, mime = fetch_image(google[0])
        result = read_slip_image(image, mime, api_key, book=book)
        result["_shared_from"] = "google"
        return result

    # 2) A Hard Rock "share betslip" link only carries Hard Rock's internal bet IDs
    #    (hardrock://betslip/<id>,<id>,...), and the page redirects to the app's home page.
    #    The bets can't be read from it, so say so plainly instead of guessing.
    for url in urls:
        ids = hardrock_betslip_ids(url)
        if ids is not None:
            raise ShareError(f"This Hard Rock share link only carries {len(ids)} internal bet ID"
                             f"{'' if len(ids) == 1 else 's'} ({len(ids)}-leg slip), not the teams, lines or odds, "
                             "and Hard Rock's page just opens their app. Take a screenshot of the slip and upload "
                             "it, or share the screenshot as a Google Photos link and paste that.")

    # 3) A sportsbook page.
    page_text, opened, blocked = "", [], []
    for url in urls:
        if not _allowed(url):
            blocked.append(url)
            continue
        opened.append(url)
        try:
            page_text = fetch(url)
        except Exception:
            page_text = ""
        if page_text and _looks_like_bets(page_text):
            break
        page_text = ""

    # 4) Plain text (with or without a link).
    combined = "\n".join(x for x in (text_part, page_text) if x)
    if not combined or not _looks_like_bets(combined):
        if opened:
            raise ShareError("The Hard Rock link opened, but the page didn't show the bets (it may need you to "
                             "sign in, or only show them inside the app). Upload a screenshot instead.")
        if blocked:
            raise ShareError(f"That link goes to {_host(blocked[0]) or 'an unknown site'}, which the app doesn't "
                             "open for safety. It reads Hard Rock links and Google Photos/Drive links. "
                             "Upload a screenshot instead.")
        raise ShareError("That doesn't look like a bet slip: no odds or lines found. Paste the full slip text, "
                         "or upload a screenshot.")
    result = read_slip_text(combined[:8000], api_key, book=book)
    result["_shared_from"] = "link" if page_text else "text"
    return result
