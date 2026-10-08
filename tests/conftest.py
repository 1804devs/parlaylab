import pytest

from parlaylab import nebius_client


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call Nebius, even if a real key is set in the Codespace."""
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_AGENT_MODEL", raising=False)
    monkeypatch.delenv("NEBIUS_VISION_MODEL", raising=False)
    nebius_client._MODELS_CACHE.clear()
