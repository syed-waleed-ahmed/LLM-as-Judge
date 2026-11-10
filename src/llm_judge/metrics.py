"""Deterministic, LLM-free text metrics for reference-based evaluation.

These are cheap, reproducible baselines to read alongside LLM judgments. They
measure surface overlap, not meaning.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import ReferenceMetrics

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (punctuation dropped)."""
    return _TOKEN_RE.findall(text.casefold())


def exact_match(prediction: str, reference: str) -> bool:
    """True when both texts have identical tokens (ignoring case and punctuation)."""
    return tokenize(prediction) == tokenize(reference)


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style bag-of-words F1 between ``prediction`` and ``reference``.

    Returns 1.0 when both are empty of tokens and 0.0 when only one is.
    """
    pred, ref = Counter(tokenize(prediction)), Counter(tokenize(reference))
    if not pred and not ref:
        return 1.0
    common = sum((pred & ref).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def reference_metrics(prediction: str, reference: str) -> ReferenceMetrics:
    """Compute all reference metrics for one prediction."""
    return ReferenceMetrics(
        exact_match=exact_match(prediction, reference),
        token_f1=round(token_f1(prediction, reference), 4),
    )
