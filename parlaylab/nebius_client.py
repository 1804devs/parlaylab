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
import threading
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

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
MAX_OUTPUT = 16000

# ---------------- usage + cost tracking ----------------
# Approximate Nebius prices, $ per 1M tokens (input, output). Used only for
# ESTIMATES in reports; your Nebius billing page is the real number.
PRICES = {
    "nemotron-3-nano-30b": (0.06, 0.24),
    "nemotron-3-nano-omni": (0.06, 0.24),
    "nemotron-3-super": (0.30, 0.90),
    "nemotron-3-ultra": (1.00, 3.00),
    "cosmos3-super-reasoner": (0.10, 0.30),
    "gemma-3-27b": (0.10, 0.30),
    "qwen2.5-vl-72b": (0.25, 0.75),
}
_USAGE: dict[str, dict] = {}
_USAGE_LOCK = threading.Lock()


def _record_usage(model: str, resp) -> None:
    u = getattr(resp, "usage", None)
    if not u:
        return
    with _USAGE_LOCK:
        row = _USAGE.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        row["calls"] += 1
        row["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        row["output_tokens"] += getattr(u, "completion_tokens", 0) or 0


def usage_reset() -> None:
    with _USAGE_LOCK:
        _USAGE.clear()


def usage_snapshot() -> dict[str, dict]:
    """{model: {calls, input_tokens, output_tokens, est_cost (None if price unknown)}}."""
    with _USAGE_LOCK:
        out = {m: dict(v) for m, v in _USAGE.items()}
    for m, v in out.items():
        price = next((p for k, p in PRICES.items() if k in m.lower()), None)
        v["est_cost"] = (None if price is None else
                         round(v["input_tokens"] / 1e6 * price[0] + v["output_tokens"] / 1e6 * price[1], 4))
    return out


def _create(client, model: str, messages, max_tokens: int, temperature: float, think: bool):
    """One API call (usage recorded). `think=False` asks Nemotron to skip its reasoning step."""
    resp = _create_raw(client, model, messages, max_tokens, temperature, think)
    _record_usage(model, resp)
    return resp


def _create_raw(client, model: str, messages, max_tokens: int, temperature: float, think: bool):
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    no_think = not think and "nemotron" in model.lower()
    try:
        return client.chat.completions.create(**kwargs, **({"extra_body": NO_THINK} if no_think else {}))
    except openai.BadRequestError as e:
        err = str(e).lower()
        if no_think and ("chat_template" in err or "enable_thinking" in err):
            return client.chat.completions.create(**kwargs)  # endpoint ignores the switch: plain call
        if max_tokens > 4000 and "token" in err:
            # Small models can have a lower output cap; retry once within it.
            return client.chat.completions.create(**{**kwargs, "max_tokens": 4000},
                                                  **({"extra_body": NO_THINK} if no_think else {}))
        raise


def _answer(resp) -> tuple[str, str | None]:
    """(answer text, finish_reason). Thinking that was cut off is not an answer."""
    choice = resp.choices[0]
    msg = choice.message
    text = (msg.content or "").strip()
    if not text and choice.finish_reason == "stop":
        # Some reasoning endpoints put a finished answer only in reasoning_content.
        extra = getattr(msg, "reasoning_content", None) or ""
        if not extra and getattr(msg, "model_extra", None):
            extra = msg.model_extra.get("reasoning_content") or ""
        text = extra.strip()
    return strip_thinking(text), choice.finish_reason


def chat(
    messages: list[dict[str, Any]],
    model: str = "agent",
    api_key: str | None = None,
    max_tokens: int = 6000,
    temperature: float = 0.3,
    think: bool = True,
) -> str:
    """Send a chat request and return the final answer text.

    `model` can be a role ("agent", "builder", "vision") or an exact model ID.
    `think=False` skips Nemotron's reasoning step: faster and cheaper for simple
    jobs like turning a slip into JSON.

    If a reasoning model runs out of room before answering, this retries with
    thinking off and more room, then (for the cheap "agent" role) hands the job
    to the stronger "builder" model.
    """
    role = model if model in ROLES else None
    if role:
        model = pick_model(role, api_key)
    client = get_client(api_key)
    try:
        resp = _create(client, model, messages, max_tokens, temperature, think)
        text, finish = _answer(resp)
        if not text and finish == "length":
            resp = _create(client, model, messages, MAX_OUTPUT, temperature, think=False)
            text, finish = _answer(resp)
        if not text and role == "agent":
            stronger = pick_model("builder", api_key)
            if stronger != model:
                model = stronger
                resp = _create(client, model, messages, MAX_OUTPUT, temperature, think=False)
                text, finish = _answer(resp)
    except openai.NotFoundError as e:
        _MODELS_CACHE.clear()
        raise NebiusError(f"Nebius doesn't offer `{model}` on your account. Open 'Models' in the "
                          "sidebar and click 'Check my models' to pick one that works.") from e
    except openai.AuthenticationError as e:
        raise NebiusError("Nebius rejected the API key. Check it was copied in full.") from e
    except openai.RateLimitError as e:
        raise NebiusError("Nebius says you're out of credits or sending too fast. "
                          "Check your balance, wait a minute, and try again.") from e
    if not text:
        raise NebiusError(f"{model} didn't return an answer (finish_reason={finish}). "
                          "Try again, or switch Model cost to Best in the sidebar.")
    return text


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
