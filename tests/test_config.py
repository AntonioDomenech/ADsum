"""Tests for configuration helpers exposed to the UI."""

from __future__ import annotations

import os

import pytest

from adsum import config


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch, tmp_path):
    """Ensure each test starts with a clean configuration environment."""

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config, "_ENV_PATH", env_path)
    monkeypatch.setattr(config, "_settings", None)

    for key in list(os.environ):
        if key.startswith("ADSUM_"):
            monkeypatch.delenv(key, raising=False)

    yield


def test_list_environment_settings_reflects_defaults():
    entries = list(config.list_environment_settings())
    env_names = {entry.env_name for entry in entries}

    assert "ADSUM_SAMPLE_RATE" in env_names
    assert "ADSUM_OPENAI_NOTES_MODEL" in env_names
    assert "ADSUM_DEFAULT_MIC_DEVICE" in env_names
    assert "ADSUM_DEFAULT_SYSTEM_DEVICE" in env_names


def test_update_environment_setting_persists_and_reloads():
    updated = config.update_environment_setting("sample_rate", "22050")

    assert updated.sample_rate == 22050
    assert config.get_settings().sample_rate == 22050
    assert os.environ["ADSUM_SAMPLE_RATE"] == "22050"

    env_contents = config._ENV_PATH.read_text().strip().splitlines()  # type: ignore[attr-defined]
    assert "ADSUM_SAMPLE_RATE=22050" in env_contents


def test_clear_environment_setting_removes_override():
    config.update_environment_setting("sample_rate", "24000")
    cleared = config.clear_environment_setting("sample_rate")

    assert cleared.sample_rate == config.Settings().sample_rate
    assert "ADSUM_SAMPLE_RATE" not in os.environ
    assert not config._ENV_PATH.exists()  # type: ignore[attr-defined]
