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


def test_recovers_object_despite_stray_braces_and_braces_in_strings():
    # v1 sliced from the first "{" to the last "}", which breaks on input like this.
    obj = dict(VALID, overall_comment="uses } and { inside")
    noisy = "Prefix {not json} then " + json.dumps(obj) + " trailing }"
    result = parse_judge_result(noisy)
    assert result.overall_comment == "uses } and { inside"


def test_rejects_non_finite_scores():
    bad = json.dumps(VALID).replace('"output_a_score": 7', '"output_a_score": NaN')
    with pytest.raises(ResponseParsingError):
        parse_judge_result(bad)


def test_rejects_top_level_array():
    with pytest.raises(ResponseParsingError, match="must be a JSON object"):
        parse_judge_result(json.dumps([VALID]))


def test_accepts_lenient_winner_spellings():
    data = dict(VALID, overall_winner="Tie", criteria=[dict(VALID["criteria"][0], winner=" a ")])
    result = parse_judge_result(json.dumps(data))
    assert result.overall_winner.value == "tie"
    assert result.criteria[0].winner.value == "A"


def test_missing_overall_winner_defaults_to_tie():
    data = {k: v for k, v in VALID.items() if k != "overall_winner"}
    assert parse_judge_result(json.dumps(data)).overall_winner.value == "tie"


def test_library_fields_are_never_read_from_model_output():
    data = dict(
        VALID,
        overall_score_a=0.99,
        warnings=[{"code": "x", "message": "y"}],
        metadata={"model": "spoofed"},
    )
    result = parse_judge_result(json.dumps(data))
    assert result.overall_score_a is None
    assert result.warnings == []
    assert result.metadata == {}
