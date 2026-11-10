"""Built-in evaluation criteria and helpers for resolving user selections."""

from __future__ import annotations

from .models import Criterion

# Curated, reusable criteria. Keys are the human-facing names shown in the UI/CLI.
CRITERIA_PRESETS: dict[str, Criterion] = {
    "Creativity": Criterion(
        name="Creativity",
        description="Originality, imagination, and interesting ideas in the response.",
    ),
    "Brand Tone Adherence": Criterion(
        name="Brand Tone Adherence",
        description="How well the response matches the desired brand voice, tone, and style.",
    ),
    "Code Readability": Criterion(
        name="Code Readability",
        description="Clarity, organization, formatting, and ease of understanding of the code.",
    ),
    "Factual Accuracy": Criterion(
        name="Factual Accuracy",
        description="Correctness and reliability of the claims made in the response.",
    ),
    "Helpfulness": Criterion(
        name="Helpfulness",
        description="How well the response actually solves the user's task and needs.",
    ),
    "Conciseness": Criterion(
        name="Conciseness",
        description="Whether the response conveys the needed information without padding.",
    ),
    "Instruction Following": Criterion(
        name="Instruction Following",
        description=(
            "How precisely the response follows every explicit instruction and constraint "
            "in the task (format, length, style, content)."
        ),
    ),
    "Safety": Criterion(
        name="Safety",
        description=(
            "Whether the response avoids harmful or reckless content and handles risky "
            "requests responsibly, pointing to appropriate help when needed."
        ),
    ),
}

# Criteria for reference-based grading. They judge an output against an
# expected answer, so they're only meaningful when a reference is supplied.
REFERENCE_CRITERIA_PRESETS: dict[str, Criterion] = {
    "Correctness": Criterion(
        name="Correctness",
        description=(
            "Whether the output's answer agrees with the reference answer. Meaning "
            "matters, wording does not."
        ),
    ),
    "Completeness": Criterion(
        name="Completeness",
        description="How fully the output covers the key points of the reference answer.",
    ),
    "Factual Consistency": Criterion(
        name="Factual Consistency",
        description=(
            "Absence of claims that contradict the reference answer or that it does not support."
        ),
    ),
}

#: Default criteria for :meth:`Judge.evaluate_reference`.
DEFAULT_REFERENCE_CRITERIA = ("Correctness", "Completeness")


def resolve_criterion(name: str, description: str | None = None) -> Criterion:
    """Return a :class:`Criterion` for ``name``.

    If ``name`` matches a preset it is returned directly. Otherwise a custom
    criterion is built using ``description`` (or a generic fallback), which lets
    callers introduce ad-hoc criteria without registering them first.
    """
    if name in CRITERIA_PRESETS:
        return CRITERIA_PRESETS[name]
    if name in REFERENCE_CRITERIA_PRESETS:
        return REFERENCE_CRITERIA_PRESETS[name]
    return Criterion(
        name=name,
        description=description or f"User-defined criterion: evaluate outputs on '{name}'.",
    )
