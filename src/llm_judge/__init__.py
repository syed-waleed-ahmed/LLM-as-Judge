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

from .agents import AgentResult, AgentStep, AgentTrace, ToolSpec
from .config import Settings, get_settings
from .criteria import (
    AGENT_CRITERIA_PRESETS,
    CRITERIA_PRESETS,
    REFERENCE_CRITERIA_PRESETS,
    resolve_criterion,
)
from .exceptions import (
    ConfigurationError,
    LLMJudgeError,
    ProviderError,
    ResponseParsingError,
    ValidationError,
)
from .judge import Judge
from .models import (
    RESULT_SCHEMA_VERSION,
    Criterion,
    CriterionScore,
    EvaluationWarning,
    JudgeResult,
    PairwiseMatch,
    PointwiseResult,
    PointwiseScore,
    RankedCandidate,
    RankingResult,
    ReferenceMetrics,
    Winner,
)
from .prompts import PROMPT_VERSION
from .rag import RagResult
from .reliability import ConsistencyReport, JudgeAgreement

# Single source of truth for the package version (read by setuptools).
__version__ = "2.3.0"


__all__ = [
    "Judge",
    "Criterion",
    "CriterionScore",
    "JudgeResult",
    "Winner",
    "EvaluationWarning",
    "PointwiseResult",
    "PointwiseScore",
    "ReferenceMetrics",
    "RankingResult",
    "RankedCandidate",
    "PairwiseMatch",
    "REFERENCE_CRITERIA_PRESETS",
    "AGENT_CRITERIA_PRESETS",
    "RagResult",
    "AgentResult",
    "AgentStep",
    "AgentTrace",
    "ToolSpec",
    "ConsistencyReport",
    "JudgeAgreement",
    "RESULT_SCHEMA_VERSION",
    "PROMPT_VERSION",
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
