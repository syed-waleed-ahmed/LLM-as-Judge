from __future__ import annotations

import json

import pytest

import llm_judge.config as config_module
import llm_judge.judge as judge_module
from llm_judge.cli import main
from llm_judge.config import Settings
from llm_judge.models import CriterionScore, JudgeResult, Winner


class _FakeJudge:
    """Stand-in for Judge that records inputs and returns a fixed result."""

    last_kwargs = None
    last_settings = None

    def __init__(self, *args, settings=None, **kwargs):
        self.settings = settings
        _FakeJudge.last_settings = settings

    def evaluate(self, task_description, output_a, output_b, criteria, **kwargs):
        _FakeJudge.last_kwargs = {
            "task": task_description,
            "output_a": output_a,
            "output_b": output_b,
            "criteria": criteria,
            **kwargs,
        }
        return JudgeResult(
            criteria=[
                CriterionScore(
                    criterion="Helpfulness",
                    output_a_score=8,
                    output_b_score=3,
                    winner=Winner.A,
                )
            ],
            overall_winner=Winner.A,
            overall_comment="A wins.",
            metadata={"model": "fake", "latency_seconds": 0.01},
        )


@pytest.fixture
def patched_judge(monkeypatch):
    monkeypatch.setattr(judge_module, "Judge", _FakeJudge)
    monkeypatch.setattr(
        config_module, "get_settings", lambda: Settings(_env_file=None, groq_api_key="k")
    )
    return _FakeJudge


def test_list_criteria(capsys):
    assert main(["list-criteria"]) == 0
    out = capsys.readouterr().out
    assert "Creativity" in out
    assert "Helpfulness" in out


def test_evaluate_summary(patched_judge, capsys):
    code = main(
        [
            "evaluate",
            "--task",
            "t",
            "--output-a",
            "a",
            "--output-b",
            "b",
            "--criteria",
            "Helpfulness",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Overall winner: A" in out
    assert "Helpfulness" in out


def test_evaluate_json_output(patched_judge, capsys):
    code = main(["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_winner"] == "A"


def test_evaluate_reads_from_files(patched_judge, tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("from file a", encoding="utf-8")
    b.write_text("from file b", encoding="utf-8")
    code = main(["evaluate", "--task", "t", "--output-a-file", str(a), "--output-b-file", str(b)])
    assert code == 0
    assert patched_judge.last_kwargs["output_a"] == "from file a"


def test_evaluate_missing_input_errors(patched_judge):
    with pytest.raises(SystemExit):
        main(["evaluate", "--output-a", "a", "--output-b", "b"])


def test_mitigate_bias_flag_propagates(patched_judge):
    main(["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b"])
    assert patched_judge.last_settings.mitigate_position_bias is False
    main(["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b", "--mitigate-bias"])
    assert patched_judge.last_settings.mitigate_position_bias is True


def test_weights_and_candidate_models_are_passed_through(patched_judge):
    base = ["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b"]
    code = main(
        [
            *base,
            "--criteria",
            "Helpfulness,Conciseness",
            "--weights",
            "Conciseness=3",
            "--model-a",
            "gpt-4o",
            "--model-b",
            "llama3.2",
        ]
    )
    assert code == 0
    kwargs = patched_judge.last_kwargs
    assert [c.weight for c in kwargs["criteria"]] == [1.0, 3.0]
    assert kwargs["candidate_a_model"] == "gpt-4o"
    assert kwargs["candidate_b_model"] == "llama3.2"


@pytest.mark.parametrize(
    "weights",
    ["Conciseness", "Conciseness=heavy", "=2", "Creativity=2", "Helpfulness=0"],
)
def test_invalid_weights_exit_with_an_error(patched_judge, weights):
    base = ["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b"]
    with pytest.raises(SystemExit):
        main([*base, "--criteria", "Helpfulness,Conciseness", "--weights", weights])


def test_summary_shows_overall_scores_and_warnings(capsys):
    from llm_judge.cli import _print_summary
    from llm_judge.models import EvaluationWarning

    result = JudgeResult(
        criteria=[CriterionScore(criterion="C", output_a_score=8, output_b_score=3, winner="A")],
        overall_winner=Winner.A,
        overall_score_a=0.78,
        overall_score_b=0.22,
        warnings=[EvaluationWarning(code="possible_verbosity_bias", message="long")],
    )
    _print_summary(result)
    out = capsys.readouterr().out
    assert "A=0.780 B=0.220" in out
    assert "possible_verbosity_bias: long" in out


def test_batch_command(patched_judge, tmp_path, capsys):
    inp = tmp_path / "in.csv"
    inp.write_text("id,task_description,output_a,output_b\n1,t,a,b\n2,t,a,b\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    code = main(["batch", "--input", str(inp), "--output", str(out), "--criteria", "Helpfulness"])
    assert code == 0
    assert out.exists()
    assert "overall_winner" in out.read_text(encoding="utf-8")
