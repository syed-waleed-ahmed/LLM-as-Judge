"""A deterministic, offline *mock* judge for demos, tests and keyless use.

**This is not an LLM.** It scores outputs with simple, transparent text
features: coverage of the task's key terms, length adequacy, lexical diversity
and (with a reference) word overlap. Every explanation and comment says so,
and results record ``provider: "mock"``. Use it to try the platform without
credentials or network access, never to draw conclusions about model quality.

It understands the prompts built by this package: it reads the criteria and
scales from the schema block of the system prompt, and the task, outputs and
reference from the boundary-marked blocks of the user message. Because it only
looks at content, never position, it is free of position bias by construction.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from ..diagnostics import detect_prompt_injection
from ..exceptions import ProviderError
from ..metrics import token_f1, tokenize
from ..scoring import criterion_key
from .base import ChatMessage, ProviderHealth, UsageTracker

MOCK_NOTICE = "Mock judge (deterministic heuristic, not an LLM)"

_SCHEMA_ENTRY_RE = re.compile(
    r'"criterion": ("(?:[^"\\]|\\.)*"),\s*'
    r'"(?:output_a_score|score)": <integer from (\d+) \(worst\) to (\d+) \(best\)>'
)
_BLOCK_RE = re.compile(
    r"<<<(TASK|OUTPUT A|OUTPUT B|OUTPUT|REFERENCE) (\w+)>>>\n(.*?)\n<<<END \1 \2>>>", re.S
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with your you our we they their them i me my".split()
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS]


def _overlap(part: Iterable[str], whole: Iterable[str]) -> float:
    """Share of the distinct tokens in ``part`` that also occur in ``whole``."""
    part_set, whole_set = set(part), set(whole)
    return len(part_set & whole_set) / len(part_set) if part_set else 0.0


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


class MockProvider:
    """Offline heuristic stand-in for an LLM judge (see the module docstring)."""

    name = "mock"

    def __init__(self, model: str = "mock-heuristic-v1") -> None:
        self.model = model
        self.usage = UsageTracker()

    def health(self) -> ProviderHealth:
        return ProviderHealth(True, self.name, self.model, "offline heuristic judge, always ready")

    # -- protocol ---------------------------------------------------------------

    def complete_json(self, messages: list[ChatMessage]) -> str:
        # Repair turns append messages; the original prompt is always first.
        system, user = messages[0]["content"], messages[1]["content"]
        criteria = [
            (json.loads(name), int(lo), int(hi))
            for name, lo, hi in _SCHEMA_ENTRY_RE.findall(system)
        ]
        blocks = {label: text for label, _token, text in _BLOCK_RE.findall(user)}
        if not criteria or "TASK" not in blocks:
            raise ProviderError(
                "The mock provider only understands prompts built by llm_judge.", retryable=False
            )
        self.usage.record()
        if "OUTPUT A" in blocks and "OUTPUT B" in blocks:
            payload = self._pairwise(
                criteria, blocks["TASK"], blocks["OUTPUT A"], blocks["OUTPUT B"]
            )
        elif "OUTPUT" in blocks:
            payload = self._pointwise(
                criteria, blocks["TASK"], blocks["OUTPUT"], blocks.get("REFERENCE")
            )
        else:
            raise ProviderError("The mock provider found no output to judge.", retryable=False)
        return json.dumps(payload)

    # -- scoring ----------------------------------------------------------------

    @staticmethod
    def _quality(
        criterion: str, output: str, task: str, reference: str | None
    ) -> tuple[float, str]:
        """Return a quality in [0, 1] and a short description of the features used."""
        words = len(output.split())
        tokens = _content_tokens(output)
        coverage = _overlap(_content_tokens(task), tokens)
        length = min(1.0, words / 30) * (0.8 if words > 400 else 1.0)
        diversity = len(set(tokens)) / len(tokens) if tokens else 0.0
        key = criterion_key(criterion)

        if reference is not None and key == "correctness":
            quality, detail = token_f1(output, reference), "word-overlap F1 with the reference"
        elif reference is not None and key == "completeness":
            quality = _overlap(_content_tokens(reference), tokens)
            detail = "share of the reference's key terms present"
        elif reference is not None and key == "factual consistency":
            quality = _overlap(tokens, _content_tokens(reference))
            detail = "share of the output's terms supported by the reference"
        elif "concis" in key:
            quality = max(0.0, 1.0 - max(0, words - 25) / 200) * (0.5 if words < 3 else 1.0)
            detail = "brevity"
        elif "creativ" in key:
            quality, detail = 0.6 * diversity + 0.4 * length, "lexical variety and length"
        elif "code" in key:
            lines = [ln for ln in output.splitlines() if ln.strip()] or [output]
            short = sum(len(ln) <= 80 for ln in lines) / len(lines)
            quality, detail = 0.4 + 0.6 * short * min(1.0, words / 10), "line length and size"
        else:
            quality, detail = 0.6 * coverage + 0.3 * length + 0.1 * diversity, "task-term coverage"

        if detect_prompt_injection(output):
            quality *= 0.5
            detail += "; penalised for instructions aimed at the judge"
        quality = max(0.0, min(1.0, quality))
        summary = (
            f"{MOCK_NOTICE}: {detail} -> {quality:.0%} "
            f"(task-term coverage {coverage:.0%}, {words} words)."
        )
        return quality, summary

    @staticmethod
    def _to_scale(quality: float, low: int, high: int) -> int:
        return round(low + quality * (high - low))

    def _pairwise(
        self, criteria: list[tuple[str, int, int]], task: str, a: str, b: str
    ) -> dict[str, object]:
        entries = []
        total_a = total_b = 0.0
        for name, low, high in criteria:
            qa, why_a = self._quality(name, a, task, None)
            qb, why_b = self._quality(name, b, task, None)
            sa, sb = self._to_scale(qa, low, high), self._to_scale(qb, low, high)
            total_a, total_b = total_a + qa, total_b + qb
            entries.append(
                {
                    "criterion": name,
                    "output_a_score": sa,
                    "output_b_score": sb,
                    "output_a_explanation": why_a,
                    "output_b_explanation": why_b,
                    "winner": "A" if sa > sb else "B" if sb > sa else "tie",
                }
            )
        overall = "A" if total_a > total_b else "B" if total_b > total_a else "tie"
        return {
            "criteria": entries,
            "overall_winner": overall,
            "overall_comment": (
                f"{MOCK_NOTICE}. Scores come from simple text features, so treat this "
                "as a demonstration, not a real evaluation."
            ),
        }

    def _pointwise(
        self,
        criteria: list[tuple[str, int, int]],
        task: str,
        output: str,
        reference: str | None,
    ) -> dict[str, object]:
        entries = []
        for name, low, high in criteria:
            quality, why = self._quality(name, output, task, reference)
            entries.append(
                {"criterion": name, "score": self._to_scale(quality, low, high), "explanation": why}
            )
        coverage = _overlap(_content_tokens(task), _content_tokens(output))
        if coverage < 0.5:
            tip = "address the task's key terms more directly"
        elif len(output.split()) < 15:
            tip = "add more detail"
        else:
            tip = "tighten the wording"
        payload: dict[str, object] = {
            "criteria": entries,
            "feedback": f"{MOCK_NOTICE} suggestion: {tip}.",
        }
        if reference is not None:
            out_tokens = _content_tokens(output)
            ref_tokens = _content_tokens(reference)
            payload["missing_information"] = [
                s for s in _sentences(reference) if _overlap(_content_tokens(s), out_tokens) < 0.5
            ][:5]
            payload["unsupported_claims"] = [
                s
                for s in _sentences(output)
                if len(_content_tokens(s)) >= 3 and _overlap(_content_tokens(s), ref_tokens) < 0.3
            ][:5]
        return payload
