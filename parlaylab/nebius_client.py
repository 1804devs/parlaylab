"""Thin wrapper around Nebius Token Factory's OpenAI-compatible API.

Model choice is automatic: the app asks Nebius which models YOUR key can
use (`/v1/models`) and picks from those, so a model that isn't offered on
your account (e.g. one that is dedicated-only) can't break the app.

Cheap mode (default) uses Nemotron 3 Nano for most calls and Nemotron 3
Super only for the R&D Engineer and Reviewer. Best mode uses Super
everywhere. See MODE_PATTERNS below.

You can force a choice with NEBIUS_AGENT_MODEL / NEBIUS_BUILDER_MODEL /
NEBIUS_VISION_MODEL or from the app's sidebar.

Nemotron models are *reasoning* models: the answer can arrive in
`message.content` or only in `reasoning_content`, and may include
<think> blocks. `chat()` handles all of that.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import openai
from openai import OpenAI

BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

DEFAULT_AGENT = "nvidia/nemotron-3-super-120b-a12b"

# Roles:
#   agent   - research team, slip parsing, R&D analysts (most calls → cheapest model that works)
#   builder - R&D Engineer + Reviewer (writes code; rare, so quality beats price)
#   vision  - reads screenshots
#
# Modes (PARLAYLAB_MODE): "cheap" (default) or "best". Patterns are tried in order
# against the models your key can see. Approx. Nebius prices per 1M tokens (in/out):
#   Nemotron 3 Nano 30B $0.06/$0.24 · Cosmos 3 Super Reasoner $0.10/$0.30
#   Gemma 3 27B $0.10/$0.30 · Nemotron 3 Super $0.30/$0.90 · Qwen2.5-VL-72B $0.25/$0.75
_NOT_OMNI = r"(?!.*omni)"
_ANY_VISION = r"-vl|vl-|_vl|vision|omni|pixtral|llava|gemma-3|cosmos|minicpm-v"
MODE_PATTERNS = {
    "cheap": {
        "agent": [r"nemotron-3-nano-30b", r"nemotron.*nano" + _NOT_OMNI, r"nemotron-3-super",
                  r"nvidia/.*nemotron" + _NOT_OMNI],
        "builder": [r"nemotron-3-super", r"nemotron.*super", r"nvidia/.*nemotron" + _NOT_OMNI],
        "vision": [r"nvidia/.*nemotron.*(omni|vl)", r"nvidia/.*cosmos", r"gemma-3", _ANY_VISION],
    },
    "best": {
        "agent": [r"nemotron-3-super", r"nemotron.*super", r"nvidia/.*nemotron" + _NOT_OMNI],
        "builder": [r"nemotron-3-super", r"nemotron.*super", r"nvidia/.*nemotron" + _NOT_OMNI],
        "vision": [r"nvidia/.*nemotron.*(omni|vl)", r"qwen.*vl", r"gemma-3", r"nvidia/.*cosmos", _ANY_VISION],
    },
}
ROLES = ("agent", "builder", "vision")
ENV_FOR = {"agent": "NEBIUS_AGENT_MODEL", "builder": "NEBIUS_BUILDER_MODEL", "vision": "NEBIUS_VISION_MODEL"}


def mode() -> str:
    m = (os.getenv("PARLAYLAB_MODE") or "cheap").lower()
    return m if m in MODE_PATTERNS else "cheap"

# Kept for older imports; the real choice happens in pick_model().
AGENT_MODEL = os.getenv("NEBIUS_AGENT_MODEL") or DEFAULT_AGENT
VISION_MODEL = os.getenv("NEBIUS_VISION_MODEL") or ""

_MODELS_CACHE: dict[str, tuple[float, list[str]]] = {}
CACHE_SECONDS = 600


class NebiusError(RuntimeError):
    pass


def _key(api_key: str | None) -> str:
    key = api_key or os.getenv("NEBIUS_API_KEY")
    if not key:
        raise NebiusError("No Nebius API key. Paste it in the sidebar, or set NEBIUS_API_KEY.")
    return key


def get_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=_key(api_key))


# ---------------- model discovery ----------------

def list_models(api_key: str | None = None, refresh: bool = False) -> list[str]:
    """Model IDs this key can call. Empty list if Nebius can't be reached."""
    key = _key(api_key)
    hit = _MODELS_CACHE.get(key)
    if hit and not refresh and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        ids = sorted(m.id for m in get_client(key).models.list())
    except openai.AuthenticationError as e:
        raise NebiusError("Nebius rejected the API key. Check it was copied in full.") from e
    except Exception:
        ids = []
    _MODELS_CACHE[key] = (time.time(), ids)
    return ids


def _match(name: str | None, models: list[str]) -> str | None:
    """Case-insensitive exact match."""
    if not name:
        return None
    low = name.lower()
    return next((m for m in models if m.lower() == low), None)


def choose(role: str, models: list[str], cost_mode: str | None = None) -> str | None:
    """Pure choice logic (unit-tested): override first, then the mode's patterns in order."""
    override = os.getenv(ENV_FOR[role])
    if override and _match(override, models):
        return _match(override, models)
    patterns = MODE_PATTERNS[cost_mode or mode()][role]
    for pat in patterns:
        found = [m for m in models if re.search(pat, m, flags=re.IGNORECASE)]
        if found:
            # Prefer instruct/chat variants over base models when both exist.
            found.sort(key=lambda m: (("base" in m.lower()), len(m)))
            return found[0]
    return None


def pick_model(role: str, api_key: str | None = None) -> str:
    """The model ID to call for a role ("agent" or "vision")."""
    models = list_models(api_key)
    if not models:  # couldn't list: fall back to configured names
        fallback = os.getenv(ENV_FOR[role])
        if fallback:
            return fallback
        if role in ("agent", "builder"):
            return DEFAULT_AGENT
        raise NebiusError("Couldn't get the model list from Nebius, so no image model is set. "
                          "Use 'Paste the slip text' for now, or pick a model in the sidebar.")
    chosen = choose(role, models)
    if not chosen:
        if role == "vision":
            raise NebiusError("Your Nebius account has no model that can read images. "
                              "Use 'Paste the slip text' instead (it uses Nemotron).")
        raise NebiusError("No Nemotron model is available on your Nebius account.")
    return chosen


def model_name(role: str, api_key: str | None = None) -> str:
    """Best-effort name for labels and reports; never raises."""
    try:
        return pick_model(role, api_key)
    except Exception:
        return os.getenv(ENV_FOR[role]) or (DEFAULT_AGENT if role != "vision" else "auto")


def is_nvidia(model_id: str | None) -> bool:
    return bool(model_id) and ("nvidia/" in model_id.lower() or "nemotron" in model_id.lower())


def model_report(api_key: str | None = None) -> dict:
    """What the sidebar shows."""
    models = list_models(api_key, refresh=True)
    out = {"available": models, "agent": None, "builder": None, "vision": None, "errors": [], "mode": mode()}
    for role in ROLES:
        try:
            out[role] = pick_model(role, api_key)
        except NebiusError as e:
            out["errors"].append(str(e))
    return out


# ---------------- chat ----------------

def chat(
    messages: list[dict[str, Any]],
    model: str = "agent",
    api_key: str | None = None,
    max_tokens: int = 6000,
    temperature: float = 0.3,
) -> str:
    """Send a chat request and return the final answer text.

    `model` can be a role ("agent", "builder", "vision") or an exact model ID.
    """
    if model in ROLES:
        model = pick_model(model, api_key)
    client = get_client(api_key)
    try:
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
            )
        except openai.BadRequestError as e:
            # Small models can have a lower output cap; retry once within it.
            if max_tokens > 4000 and "token" in str(e).lower():
                resp = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=4000, temperature=temperature,
                )
            else:
                raise
    except openai.NotFoundError as e:
        _MODELS_CACHE.clear()
        raise NebiusError(f"Nebius doesn't offer `{model}` on your account. Open 'Models' in the "
                          "sidebar and click 'Check my models' to pick one that works.") from e
    except openai.AuthenticationError as e:
        raise NebiusError("Nebius rejected the API key. Check it was copied in full.") from e
    except openai.RateLimitError as e:
        raise NebiusError("Nebius says you're out of credits or sending too fast. "
                          "Check your balance, wait a minute, and try again.") from e
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        # Some reasoning endpoints only fill reasoning_content.
        extra = getattr(msg, "reasoning_content", None) or ""
        if not extra and getattr(msg, "model_extra", None):
            extra = msg.model_extra.get("reasoning_content") or ""
        text = extra.strip()
    if not text:
        raise NebiusError(
            f"{model} returned an empty answer (finish_reason="
            f"{resp.choices[0].finish_reason}). Try again or raise max_tokens."
        )
    return strip_thinking(text)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit inline."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:  # opening tag cut off
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model reply."""
    text = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise NebiusError("Could not find valid JSON in the model's reply.")
