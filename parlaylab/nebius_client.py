"""Thin wrapper around Nebius Token Factory's OpenAI-compatible API.

Both Nemotron models used here are *reasoning* models. Depending on the
endpoint, the final answer can arrive in `message.content` while the
thinking lands in `reasoning_content`, or `content` can come back empty
when the token budget runs out mid-thought. `chat()` handles both.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

# Vision model: reads bet-slip screenshots (text + image input).
VISION_MODEL = os.getenv("NEBIUS_VISION_MODEL", "nvidia/Nemotron-3-Nano-Omni")
# Agent model: research team (text in, text out, built for multi-agent work).
AGENT_MODEL = os.getenv("NEBIUS_AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b")


class NebiusError(RuntimeError):
    pass


def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("NEBIUS_API_KEY")
    if not key:
        raise NebiusError(
            "No Nebius API key. Set NEBIUS_API_KEY in your .env file "
            "or paste it in the sidebar."
        )
    return OpenAI(base_url=BASE_URL, api_key=key)


def chat(
    messages: list[dict[str, Any]],
    model: str = AGENT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 6000,
    temperature: float = 0.3,
) -> str:
    """Send a chat request and return the final answer text."""
    client = get_client(api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
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
