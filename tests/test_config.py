from __future__ import annotations

import pytest

from llm_judge.config import Settings, get_settings
from llm_judge.exceptions import ConfigurationError


def test_require_api_key_raises_when_missing():
    settings = Settings(_env_file=None, groq_api_key=None)
    with pytest.raises(ConfigurationError):
        settings.require_api_key()


def test_require_api_key_returns_value():
    settings = Settings(_env_file=None, groq_api_key="abc123")
    assert settings.require_api_key() == "abc123"


def test_log_level_is_uppercased():
    settings = Settings(_env_file=None, log_level="debug")
    assert settings.log_level == "DEBUG"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_reads_from_environment(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "some-other-model")
    settings = Settings(_env_file=None)
    assert settings.groq_model == "some-other-model"
