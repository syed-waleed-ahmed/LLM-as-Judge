from __future__ import annotations

import json

import pytest

from llm_judge.exceptions import ResponseParsingError
from llm_judge.parsing import parse_judge_result

VALID = {
    "criteria": [
        {
            "criterion": "Helpfulness",
            "output_a_score": 7,
            "output_b_score": 3,
            "output_a_explanation": "a",
            "output_b_explanation": "b",
            "winner": "A",
        }
    ],
    "overall_winner": "A",
    "overall_comment": "ok",
}


def test_parses_plain_json():
    result = parse_judge_result(json.dumps(VALID))
    assert result.overall_winner.value == "A"
    assert result.criteria[0].output_a_score == 7


def test_parses_json_wrapped_in_code_fence():
    fenced = "```json\n" + json.dumps(VALID) + "\n```"
    result = parse_judge_result(fenced)
    assert result.criteria[0].criterion == "Helpfulness"


def test_parses_json_with_leading_prose():
    noisy = "Sure, here is the judgment:\n" + json.dumps(VALID)
    result = parse_judge_result(noisy)
    assert result.overall_winner.value == "A"


def test_raises_on_non_json():
    with pytest.raises(ResponseParsingError):
        parse_judge_result("this is not json at all")


def test_raises_on_schema_mismatch():
    bad = json.dumps({"criteria": [], "overall_winner": "A"})
    with pytest.raises(ResponseParsingError) as exc:
        parse_judge_result(bad)
    assert exc.value.raw_response == bad
