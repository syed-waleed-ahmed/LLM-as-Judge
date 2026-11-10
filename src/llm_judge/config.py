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

    # Which judge backend to use: groq | ollama | openai_compatible | mock | <plugin>.
    provider: str = Field(default="groq", alias="JUDGE_PROVIDER")

    # Secrets use repr=False so they never appear in logs or tracebacks of Settings.
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY", repr=False)
    # Groq retires models over time; `llm-judge health` checks the configured one.
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")

    # Local inference with Ollama (https://ollama.com).
    ollama_base_url: str = Field(default="http://localhost:11434", alias="JUDGE_OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="JUDGE_OLLAMA_MODEL")

    # Any OpenAI-compatible endpoint. Deliberately prefixed, so a paid
    # OPENAI_API_KEY in the environment is never picked up by accident.
    openai_base_url: str | None = Field(default=None, alias="JUDGE_OPENAI_BASE_URL")
    openai_api_key: str | None = Field(default=None, alias="JUDGE_OPENAI_API_KEY", repr=False)
    openai_model: str | None = Field(default=None, alias="JUDGE_OPENAI_MODEL")
    openai_json_mode: bool = Field(default=True, alias="JUDGE_OPENAI_JSON_MODE")

    # Generation / request behaviour.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="JUDGE_TEMPERATURE")
    request_timeout: float = Field(default=60.0, gt=0, alias="JUDGE_REQUEST_TIMEOUT")
    max_retries: int = Field(default=3, ge=0, le=10, alias="JUDGE_MAX_RETRIES")

    # Malformed or schema-violating replies are retried this many times, each
    # time telling the judge what was wrong (a "repair" retry).
    max_parse_retries: int = Field(default=1, ge=0, le=5, alias="JUDGE_MAX_PARSE_RETRIES")

    # Evaluation behaviour.
    mitigate_position_bias: bool = Field(default=False, alias="JUDGE_MITIGATE_POSITION_BIAS")
    # Run the two bias-mitigation orderings concurrently (requires a
    # thread-safe provider; all bundled providers are).
    parallel_bias_runs: bool = Field(default=True, alias="JUDGE_PARALLEL_BIAS_RUNS")
    # Normalised (0-1) score difference treated as a tie. 0.06 is just over
    # half a point on a 1-10 scale, so whole-point differences pick a winner.
    tie_margin: float = Field(default=0.06, ge=0.0, le=0.5, alias="JUDGE_TIE_MARGIN")
    # Warn when the winner is at least this many times longer (in words).
    verbosity_ratio_threshold: float = Field(default=2.0, ge=1.0, alias="JUDGE_VERBOSITY_RATIO")
    max_criteria: int = Field(default=20, ge=1, le=100, alias="JUDGE_MAX_CRITERIA")
    # Round-robin ranking makes n(n-1)/2 comparisons, so keep n modest on free tiers.
    max_candidates: int = Field(default=8, ge=2, le=32, alias="JUDGE_MAX_CANDIDATES")

    # Batch behaviour.
    batch_max_workers: int = Field(default=4, ge=1, le=64, alias="JUDGE_BATCH_MAX_WORKERS")

    # REST API (llm-judge serve). Secrets use repr=False.
    api_keys: str | None = Field(default=None, alias="JUDGE_API_KEYS", repr=False)
    api_admin_key: str | None = Field(default=None, alias="JUDGE_API_ADMIN_KEY", repr=False)
    api_rate_limit_per_minute: int = Field(default=60, ge=0, alias="JUDGE_API_RATE_LIMIT")
    api_max_body_bytes: int = Field(default=1_000_000, gt=0, alias="JUDGE_API_MAX_BODY_BYTES")
    api_max_concurrent: int = Field(default=4, ge=1, le=64, alias="JUDGE_API_MAX_CONCURRENT")
    api_max_batch_rows: int = Field(default=50, ge=1, alias="JUDGE_API_MAX_BATCH_ROWS")
    api_max_job_rows: int = Field(default=1_000, ge=1, alias="JUDGE_API_MAX_JOB_ROWS")
    api_max_queued_jobs: int = Field(default=10, ge=1, alias="JUDGE_API_MAX_QUEUED_JOBS")
    api_cors_origins: str = Field(default="", alias="JUDGE_API_CORS_ORIGINS")

    # Public demo safeguards for the Streamlit dashboard (JUDGE_DEMO_MODE=true on a
    # public deployment): live-call budgets, input caps, and session-only storage.
    demo_mode: bool = Field(default=False, alias="JUDGE_DEMO_MODE")
    demo_session_calls: int = Field(default=25, ge=0, alias="JUDGE_DEMO_SESSION_CALLS")
    demo_daily_calls: int = Field(default=500, ge=0, alias="JUDGE_DEMO_DAILY_CALLS")
    demo_max_input_chars: int = Field(default=4_000, gt=0, alias="JUDGE_DEMO_MAX_INPUT_CHARS")
    demo_max_rows: int = Field(default=25, ge=1, alias="JUDGE_DEMO_MAX_ROWS")
    demo_max_candidates: int = Field(default=5, ge=2, alias="JUDGE_DEMO_MAX_CANDIDATES")

    # Experiment tracking database (SQLite file).
    db_path: str = Field(default=".llm_judge/experiments.db", alias="JUDGE_DB_PATH")

    # Dataset limits: files are rejected before parsing if they exceed these.
    max_dataset_bytes: int = Field(default=50_000_000, gt=0, alias="JUDGE_MAX_DATASET_BYTES")
    max_dataset_rows: int = Field(default=100_000, gt=0, alias="JUDGE_MAX_DATASET_ROWS")

    # Safety limit to avoid sending pathologically large prompts. Longer inputs
    # are truncated and the result carries an "input_truncated" warning.
    max_input_chars: int = Field(default=50_000, gt=0, alias="JUDGE_MAX_INPUT_CHARS")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("provider")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")

    def require_api_key(self) -> str:
        """Return the Groq API key or raise :class:`ConfigurationError`.

        Call this at the point of use so that importing the package (and running
        offline tooling) never requires a configured key.
        """
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. Create a .env file (see .env.example) or "
                "export GROQ_API_KEY. To try the tool without a key, use the offline mock "
                "judge: --provider mock (or JUDGE_PROVIDER=mock)."
            )
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
