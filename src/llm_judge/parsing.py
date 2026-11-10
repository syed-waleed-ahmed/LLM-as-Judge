"""Defensive parsing of raw LLM output into validated models."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from .exceptions import ResponseParsingError
from .models import JudgeResult

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Upper bound on candidate start positions tried during recovery. Judge replies
# are short, so this only matters for pathological input full of "{".
_MAX_DECODE_ATTEMPTS = 200


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    return stripped


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first complete JSON object embedded in ``text``, if any.

    Each ``{`` is tried as a start position and decoded with
    :meth:`json.JSONDecoder.raw_decode`. That respects string literals and
    nesting, so braces inside strings or stray braces in surrounding prose
    can't produce a wrong slice.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    attempts = 0
    while start != -1 and attempts < _MAX_DECODE_ATTEMPTS:
        attempts += 1
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj
        start = text.find("{", start + 1)
    return None


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse ``raw`` model output into a JSON object (a ``dict``).

    Fast path: the whole reply (minus an optional code fence) is the object.
    Recovery path: find the first complete object embedded in surrounding text.
    Raises :class:`ResponseParsingError` if neither works.
    """
    candidate = _strip_code_fences(raw)
    try:
        data: Any = json.loads(candidate)
    except json.JSONDecodeError:
        data = _first_json_object(candidate)
        if data is None:
            raise ResponseParsingError(
                "Judge response did not contain a valid JSON object.", raw_response=raw
            ) from None
    if not isinstance(data, dict):
        raise ResponseParsingError(
            f"Judge response must be a JSON object, got {type(data).__name__}.",
            raw_response=raw,
        )
    return data


def parse_judge_result(raw: str) -> JudgeResult:
    """Parse and validate raw pairwise-judge output into a :class:`JudgeResult`.

    Only the fields the judge is asked to produce are read. Library-computed
    fields (``overall_score_a``, ``warnings``, ``metadata``, ...) are never
    taken from model output. ``overall_winner`` defaults to ``"tie"`` when it
    is missing, because the judge recomputes it from the scores anyway.

    Raises :class:`ResponseParsingError` if the text is not a JSON object or
    doesn't satisfy the schema.
    """
    data = extract_json_object(raw)
    payload = {
        "criteria": data.get("criteria"),
        "overall_winner": data.get("overall_winner", "tie"),
        "overall_comment": data.get("overall_comment", ""),
    }
    try:
        return JudgeResult.model_validate(payload)
    except PydanticValidationError as exc:
        raise ResponseParsingError(
            f"Judge response did not match the expected schema: {exc}",
            raw_response=raw,
        ) from exc
