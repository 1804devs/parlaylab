"""Model auto-selection and the two-step screenshot reader."""
import json

import pytest

from parlaylab import nebius_client, slip_reader
from parlaylab.nebius_client import NebiusError, choose, pick_model

# What a Nebius account without Nemotron Omni (dedicated-only) might list.
ACCOUNT = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
]
WITH_CHEAP_VISION = ACCOUNT + ["google/gemma-3-27b-it", "nvidia/Cosmos3-Super-Reasoner"]


def test_cheap_mode_uses_nano_for_agents_and_super_for_engineer():
    assert choose("agent", ACCOUNT, "cheap") == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert choose("builder", ACCOUNT, "cheap") == "nvidia/nemotron-3-super-120b-a12b"
    # Omni is never picked as a text agent (it's dedicated-only on Nebius).
    assert choose("agent", ["nvidia/Nemotron-3-Nano-Omni", "nvidia/nemotron-3-super-120b-a12b"], "cheap") \
        == "nvidia/nemotron-3-super-120b-a12b"


def test_best_mode_uses_super():
    assert choose("agent", ACCOUNT, "best") == "nvidia/nemotron-3-super-120b-a12b"
    assert choose("builder", ACCOUNT, "best") == "nvidia/nemotron-3-super-120b-a12b"


def test_cheap_is_the_default(monkeypatch):
    assert choose("agent", ACCOUNT) == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    monkeypatch.setenv("PARLAYLAB_MODE", "best")
    assert choose("agent", ACCOUNT) == "nvidia/nemotron-3-super-120b-a12b"


def test_vision_choice_by_mode():
    assert choose("vision", ACCOUNT + ["nvidia/Nemotron-3-Nano-Omni"], "cheap") == "nvidia/Nemotron-3-Nano-Omni"
    assert choose("vision", WITH_CHEAP_VISION, "cheap") == "nvidia/Cosmos3-Super-Reasoner"
    assert choose("vision", WITH_CHEAP_VISION, "best") == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert choose("vision", ACCOUNT, "cheap") == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert choose("vision", ["nvidia/nemotron-3-super-120b-a12b"]) is None


def test_override_is_case_insensitive_and_must_exist(monkeypatch):
    monkeypatch.setenv("NEBIUS_AGENT_MODEL", "NVIDIA/NEMOTRON-3-ULTRA-550B-A55B")
    assert choose("agent", ACCOUNT) == "nvidia/Nemotron-3-Ultra-550b-a55b"
    monkeypatch.setenv("NEBIUS_AGENT_MODEL", "nvidia/does-not-exist")
    assert choose("agent", ACCOUNT, "best") == "nvidia/nemotron-3-super-120b-a12b"


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
        if model == "agent":
            assert kw.get("think") is False  # parsing skips Nemotron's thinking
        if model.startswith("Qwen"):
            return "Knicks moneyline -150\nFanDuel, stake $10"
        assert "Knicks moneyline -150" in messages[-1]["content"]  # Nemotron got the transcript
        return json.dumps(LEGS)

    monkeypatch.setattr(slip_reader, "chat", fake_chat)
    out = slip_reader.read_slip_image(b"png", "image/png", "key")
    assert calls == ["Qwen/Qwen2.5-VL-72B-Instruct", "agent"]
    assert out["legs"][0]["team"] == "New York Knicks"
    assert out["_models"] == {"vision": "Qwen/Qwen2.5-VL-72B-Instruct",
                              "parser": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"}


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
                        SimpleNamespace(NotFoundError=NotFound, AuthenticationError=Other, RateLimitError=Other,
                                        BadRequestError=Other))

    class FakeCompletions:
        def create(self, **kw):
            raise NotFound("The model `nvidia/x` does not exist.")

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(nebius_client, "get_client", lambda api_key=None: FakeClient())
    with pytest.raises(NebiusError, match="Check my models"):
        nebius_client.chat([{"role": "user", "content": "hi"}], model="nvidia/x", api_key="key")


# ---- the real failure: Nemotron Nano thinks until it runs out of room ----
from types import SimpleNamespace as NS


def _resp(content, finish):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content, model_extra={}))])


class ScriptedClient:
    """Plays back responses and records each request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def _wire(monkeypatch, client):
    monkeypatch.setattr(nebius_client, "get_client", lambda api_key=None: client)
    monkeypatch.setattr(nebius_client, "list_models", lambda api_key=None, refresh=False: ACCOUNT)


def test_think_false_sends_the_switch_to_nemotron(monkeypatch):
    client = ScriptedClient([_resp('{"ok": 1}', "stop")])
    _wire(monkeypatch, client)
    assert nebius_client.chat([{"role": "user", "content": "x"}], model="agent", api_key="k", think=False) == '{"ok": 1}'
    assert client.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert client.requests[0]["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def test_ran_out_of_room_retries_then_uses_super(monkeypatch):
    client = ScriptedClient([_resp("", "length"), _resp("", "length"), _resp("answer", "stop")])
    _wire(monkeypatch, client)
    out = nebius_client.chat([{"role": "user", "content": "x"}], model="agent", api_key="k")
    assert out == "answer"
    models = [r["model"] for r in client.requests]
    assert models == ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
                      "nvidia/nemotron-3-super-120b-a12b"]
    assert client.requests[1]["max_tokens"] == nebius_client.MAX_OUTPUT
    assert client.requests[1]["extra_body"] == nebius_client.NO_THINK


def test_cut_off_thinking_is_not_returned_as_the_answer(monkeypatch):
    cut = NS(choices=[NS(finish_reason="length",
                         message=NS(content=None, reasoning_content="Let me think about leg 1...", model_extra={}))])
    client = ScriptedClient([cut, cut, cut])
    _wire(monkeypatch, client)
    with pytest.raises(NebiusError, match="didn't return an answer"):
        nebius_client.chat([{"role": "user", "content": "x"}], model="agent", api_key="k")


def test_short_retry_succeeds_without_escalating(monkeypatch):
    client = ScriptedClient([_resp("", "length"), _resp("legs", "stop")])
    _wire(monkeypatch, client)
    assert nebius_client.chat([{"role": "user", "content": "x"}], model="agent", api_key="k") == "legs"
    assert len(client.requests) == 2
