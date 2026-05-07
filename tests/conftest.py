"""Pytest config — clear env so tests run in fallback (no-API-keys) mode."""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _clean_signbridge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with no API keys / providers configured.

    Tests that want a specific provider can `monkeypatch.setenv(...)` themselves.
    """
    for var in (
        "AMD_DEV_CLOUD_BASE_URL",
        "AMD_DEV_CLOUD_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "amd")
    monkeypatch.setenv("SIGNBRIDGE_RECOGNIZER_MODE", "vlm")
