"""Reliability diagnostics: heuristic signals that a judgment may be biased.

These checks never change scores. They emit :class:`EvaluationWarning` objects
that point a human at judgments worth a second look. They are heuristics, not
guarantees: a clean result doesn't prove the absence of bias or injection, and
a warning doesn't prove its presence.
"""

from __future__ import annotations

import re

from .models import EvaluationWarning, Winner

# --- prompt injection ---------------------------------------------------------

_INJECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,40}?"
            r"\b(previous|prior|above|earlier|all|any|your)\b.{0,40}?"
            r"\b(instructions?|prompts?|rules?|directions?|guidelines?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "asks the judge to ignore its instructions",
    ),
    (
        re.compile(
            r"\b(you are|you're|act as)\b.{0,30}?\b(judge|evaluator|grader|reviewer)\b",
            re.IGNORECASE,
        ),
        "addresses the judge directly",
    ),
    (
        re.compile(
            r"\b(give|assign|award|rate|score|grade)\b.{0,40}?"
            r"\b(10/10|100/100|full marks|perfect score|maximum score|highest score)",
            re.IGNORECASE,
        ),
        "requests a particular score",
    ),
    (
        re.compile(r'"(overall_winner|output_[ab]_score|winner)"\s*:', re.IGNORECASE),
        "contains text shaped like the judge's JSON output",
    ),
    (
        re.compile(r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
        "contains a fake chat-role header",
    ),
    (
        re.compile(r"<<<\s*(end\s+)?(task|output)\b", re.IGNORECASE),
        "contains text that imitates the prompt's boundary markers",
    ),
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return human-readable reasons why ``text`` looks like a prompt injection."""
    return [reason for pattern, reason in _INJECTION_PATTERNS if pattern.search(text)]


def injection_warnings(outputs: dict[str, str]) -> list[EvaluationWarning]:
    """Build one warning per candidate output that shows injection signals."""
    warnings: list[EvaluationWarning] = []
    for label, text in outputs.items():
        reasons = detect_prompt_injection(text)
        if reasons:
            warnings.append(
                EvaluationWarning(
                    code="possible_prompt_injection",
                    candidate=label,
                    message=(
                        f"Output {label} {', '.join(reasons)}. It was passed to the judge "
                        "as untrusted data, but review this judgment manually."
                    ),
                )
            )
    return warnings


# --- verbosity bias -------------------------------------------------------------


def verbosity_warning(
    output_a: str, output_b: str, winner: Winner, ratio_threshold: float
) -> EvaluationWarning | None:
    """Flag a win by an output that is much longer than its competitor.

    LLM judges are known to favour longer answers. A long winner is often
    legitimate (the criterion may reward completeness), so this is a prompt to
    double-check, not a verdict.
    """
    if winner is Winner.TIE:
        return None
    words_a, words_b = len(output_a.split()), len(output_b.split())
    if words_a == words_b or min(words_a, words_b) == 0:
        return None
    longer = Winner.A if words_a > words_b else Winner.B
    ratio = max(words_a, words_b) / min(words_a, words_b)
    if winner is not longer or ratio < ratio_threshold:
        return None
    return EvaluationWarning(
        code="possible_verbosity_bias",
        candidate=longer.value,
        message=(
            f"Output {longer.value} won and is {ratio:.1f}x longer "
            f"({max(words_a, words_b)} vs {min(words_a, words_b)} words). LLM judges "
            "tend to favour longer answers; check that the extra length adds value."
        ),
    )


# --- self-preference bias -----------------------------------------------------

_FAMILY_RE = re.compile(r"[a-z]+")


def model_family(model: str) -> str | None:
    """Best-effort model family from a model identifier.

    ``"llama-3.3-70b-versatile"`` gives ``"llama"``, and
    ``"meta-llama/Llama-4-Scout"`` also gives ``"llama"`` (only the part after
    the last ``/`` is used). Returns ``None`` when no family can be inferred.
    """
    name = model.strip().lower().rsplit("/", 1)[-1]
    match = _FAMILY_RE.match(name)
    return match.group(0) if match else None


def self_preference_warnings(
    judge_model: str, candidate_models: dict[str, str | None]
) -> list[EvaluationWarning]:
    """Warn when a candidate comes from the same model family as the judge."""
    judge_family = model_family(judge_model)
    if judge_family is None:
        return []
    warnings: list[EvaluationWarning] = []
    for label, candidate in candidate_models.items():
        if candidate and model_family(candidate) == judge_family:
            warnings.append(
                EvaluationWarning(
                    code="self_preference_risk",
                    candidate=label,
                    message=(
                        f"Candidate {label} ({candidate}) is from the same model family as "
                        f"the judge ({judge_model}). LLM judges tend to favour outputs "
                        "from their own family; consider an independent judge model."
                    ),
                )
            )
    return warnings
