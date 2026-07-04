"""llm_judge — a production-grade LLM-as-Judge evaluation framework.

Quick start::

    from llm_judge import Judge

    judge = Judge()
    result = judge.evaluate(
        task_description="Write a friendly tweet announcing our AI tool.",
        output_a="...",
        output_b="...",
        criteria=["Creativity", "Brand Tone Adherence"],
    )
    print(result.overall_winner, result.model_dump())
"""

from __future__ import annotations

from .config import Settings, get_settings
from .criteria import CRITERIA_PRESETS, resolve_criterion
from .exceptions import (
    ConfigurationError,
    LLMJudgeError,
    ProviderError,
    ResponseParsingError,
    ValidationError,
)
from .judge import Judge
from .models import Criterion, CriterionScore, JudgeResult, Winner

__version__ = "1.0.0"

__all__ = [
    "Judge",
    "Criterion",
    "CriterionScore",
    "JudgeResult",
    "Winner",
    "Settings",
    "get_settings",
    "CRITERIA_PRESETS",
    "resolve_criterion",
    "LLMJudgeError",
    "ConfigurationError",
    "ProviderError",
    "ResponseParsingError",
    "ValidationError",
    "__version__",
]
