"""Application configuration.

Configuration is loaded from environment variables (and an optional ``.env``
file) using ``pydantic-settings``. Unlike the previous implementation, importing
this module never raises: validation of required secrets happens lazily, when a
component actually needs them, via :meth:`Settings.require_api_key`. This keeps
the package importable for testing, documentation builds, and the ``list-criteria``
CLI command, none of which need a live API key.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError


class Settings(BaseSettings):
    """Runtime settings sourced from the environment and ``.env``.

    All values have sensible defaults except ``groq_api_key``, which is optional
    at construction time but required before making a real API call.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Generation / request behaviour.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="JUDGE_TEMPERATURE")
    request_timeout: float = Field(default=60.0, gt=0, alias="JUDGE_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, ge=0, le=10, alias="JUDGE_MAX_RETRIES")

    # Evaluation behaviour.
    mitigate_position_bias: bool = Field(default=False, alias="JUDGE_MITIGATE_POSITION_BIAS")

    # Batch behaviour.
    batch_max_workers: int = Field(default=4, ge=1, le=64, alias="JUDGE_BATCH_MAX_WORKERS")

    # Safety limit to avoid sending pathologically large prompts.
    max_input_chars: int = Field(default=50_000, gt=0, alias="JUDGE_MAX_INPUT_CHARS")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.upper()

    def require_api_key(self) -> str:
        """Return the Groq API key or raise :class:`ConfigurationError`.

        Call this at the point of use so that importing the package (and running
        offline tooling) never requires a configured key.
        """
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
                "or export GROQ_API_KEY in your environment."
            )
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
