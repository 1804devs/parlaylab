"""Model auto-selection and the two-step screenshot reader."""
import json

import pytest

from parlaylab import nebius_client, slip_reader
from parlaylab.nebius_client import NebiusError, choose, pick_model

# What a Nebius account without Nemotron Omni might list.
ACCOUNT = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
]


def test_agent_prefers_nemotron_super():
    assert choose("agent", ACCOUNT) == "nvidia/nemotron-3-super-120b-a12b"
    no_super = [m for m in ACCOUNT if "super" not in m]
    assert choose("agent", no_super).startswith("nvidia/")


def test_vision_prefers_nvidia_then_any():
    assert choose("vision", ACCOUNT + ["nvidia/Nemotron-3-Nano-Omni"]) == "nvidia/Nemotron-3-Nano-Omni"
    assert choose("vision", ACCOUNT) == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert choose("vision", ["nvidia/nemotron-3-super-120b-a12b"]) is None


def test_override_is_case_insensitive_and_must_exist(monkeypatch):
    monkeypatch.setenv("NEBIUS_AGENT_MODEL", "NVIDIA/NEMOTRON-3-ULTRA-550B-A55B")
    assert choose("agent", ACCOUNT) == "nvidia/Nemotron-3-Ultra-550b-a55b"
    monkeypatch.setenv("NEBIUS_AGENT_MODEL", "nvidia/does-not-exist")
    assert choose("agent", ACCOUNT) == "nvidia/nemotron-3-super-120b-a12b"


def test_no_vision_model_gives_clear_error(monkeypatch):
    monkeypatch.setattr(nebius_client, "list_models", lambda api_key=None, refresh=False: ["nvidia/nemotron-3-super-120b-a12b"])
    with pytest.raises(NebiusError, match="Paste the slip text"):
        pick_model("vision", "key")


LEGS = {"book": "FanDuel", "stake": 10, "total_odds": None, "potential_payout": None,
        "legs": [{"sport": "NBA", "type": "moneyline", "team": "New York Knicks", "odds": -150}]}


def test_screenshot_two_step_when_no_nvidia_vision(monkeypatch):
    monkeypatch.setattr(nebius_client, "list_models", lambda api_key=None, refresh=False: ACCOUNT)
    calls = []

    def fake_chat(messages, model, api_key=None, **kw):
        calls.append(model)
        if model.startswith("Qwen"):
            return "Knicks moneyline -150\nFanDuel, stake $10"
        assert "Knicks moneyline -150" in messages[-1]["content"]  # Nemotron got the transcript
        return json.dumps(LEGS)

    monkeypatch.setattr(slip_reader, "chat", fake_chat)
    out = slip_reader.read_slip_image(b"png", "image/png", "key")
    assert calls == ["Qwen/Qwen2.5-VL-72B-Instruct", "nvidia/nemotron-3-super-120b-a12b"]
    assert out["legs"][0]["team"] == "New York Knicks"
    assert out["_models"] == {"vision": "Qwen/Qwen2.5-VL-72B-Instruct",
                              "parser": "nvidia/nemotron-3-super-120b-a12b"}


def test_screenshot_one_step_with_nvidia_vision(monkeypatch):
    monkeypatch.setattr(nebius_client, "list_models",
                        lambda api_key=None, refresh=False: ACCOUNT + ["nvidia/Nemotron-3-Nano-Omni"])
    calls = []

    def fake_chat(messages, model, api_key=None, **kw):
        calls.append(model)
        return json.dumps(LEGS)

    monkeypatch.setattr(slip_reader, "chat", fake_chat)
    out = slip_reader.read_slip_image(b"png", "image/png", "key")
    assert calls == ["nvidia/Nemotron-3-Nano-Omni"]
    assert out["_models"]["parser"] == "nvidia/Nemotron-3-Nano-Omni"


def test_model_not_found_becomes_plain_message(monkeypatch):
    from types import SimpleNamespace

    class NotFound(Exception):
        pass

    class Other(Exception):
        pass

    monkeypatch.setattr(nebius_client, "openai",
                        SimpleNamespace(NotFoundError=NotFound, AuthenticationError=Other, RateLimitError=Other))

    class FakeCompletions:
        def create(self, **kw):
            raise NotFound("The model `nvidia/x` does not exist.")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(nebius_client, "get_client", lambda api_key=None: FakeClient())
    with pytest.raises(NebiusError, match="Check my models"):
        nebius_client.chat([{"role": "user", "content": "hi"}], model="nvidia/x", api_key="key")
