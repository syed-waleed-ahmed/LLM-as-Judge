"""Defensive parsing of raw LLM output into a validated :class:`JudgeResult`."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError as PydanticValidationError

from .exceptions import ResponseParsingError
from .models import JudgeResult

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence if the model added one."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    """Return the outermost ``{...}`` block from ``text``.

    Even with JSON mode enabled, models occasionally prepend a stray token; this
    extracts the first balanced object so we can still recover a valid result.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ResponseParsingError("No JSON object found in judge response.", raw_response=text)
    return text[start : end + 1]


def parse_judge_result(raw: str) -> JudgeResult:
    """Parse and validate raw model output into a :class:`JudgeResult`.

    Raises :class:`ResponseParsingError` if the text is not valid JSON or does
    not satisfy the schema.
    """
    candidate = _strip_code_fences(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to extracting the first balanced object.
        try:
            data = json.loads(_extract_json_object(candidate))
        except json.JSONDecodeError as exc:
            raise ResponseParsingError(
                f"Judge response was not valid JSON: {exc}", raw_response=raw
            ) from exc

    try:
        return JudgeResult.model_validate(data)
    except PydanticValidationError as exc:
        raise ResponseParsingError(
            f"Judge response did not match the expected schema: {exc}",
            raw_response=raw,
        ) from exc
