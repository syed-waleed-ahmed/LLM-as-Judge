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


# --- grade and rank (real Judge, fake provider) -----------------------------------


@pytest.fixture
def real_judge_with(monkeypatch):
    """Run the CLI against the real Judge, with ``provider`` injected as the backend."""

    def install(provider):
        monkeypatch.setattr(
            config_module, "get_settings", lambda: Settings(_env_file=None, groq_api_key="k")
        )
        monkeypatch.setattr(
            judge_module.Judge, "_build_default_provider", staticmethod(lambda _s: provider)
        )
        return provider

    return install


def test_list_criteria_includes_reference_presets(capsys):
    main(["list-criteria"])
    out = capsys.readouterr().out
    assert "Reference criteria" in out
    assert "Correctness" in out


def test_grade_command_pointwise(real_judge_with, capsys):
    from .conftest import FakeProvider, make_pointwise_payload

    real_judge_with(FakeProvider(make_pointwise_payload(("Helpfulness", 8), feedback="Shorter.")))
    code = main(["grade", "--task", "t", "--output", "o"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Pointwise grading: overall score 0.778" in out
    assert "Feedback: Shorter." in out


def test_grade_command_with_reference_uses_reference_criteria(real_judge_with, capsys, tmp_path):
    from .conftest import FakeProvider, make_pointwise_payload

    provider = real_judge_with(
        FakeProvider(
            make_pointwise_payload(
                ("Correctness", 9), ("Completeness", 4), missing=["the year"], unsupported=[]
            )
        )
    )
    ref = tmp_path / "ref.txt"
    ref.write_text("It happened in 1969.", encoding="utf-8")
    code = main(["grade", "--task", "t", "--output", "It happened.", "--reference-file", str(ref)])
    assert code == 0
    out = capsys.readouterr().out
    assert "Reference grading" in out
    assert "Missing information:\n  - the year" in out
    assert "token F1" in out
    assert "Correctness" in provider.messages[0][0]["content"]


def test_grade_command_json(real_judge_with, capsys):
    from .conftest import FakeProvider, make_pointwise_payload

    real_judge_with(FakeProvider(make_pointwise_payload(("Helpfulness", 8))))
    main(["grade", "--task", "t", "--output", "o", "--json"])
    assert json.loads(capsys.readouterr().out)["mode"] == "pointwise"


def test_rank_command(real_judge_with, capsys, tmp_path):
    from .conftest import SlotAwareProvider
    from .test_ranking import by_quality

    real_judge_with(SlotAwareProvider(by_quality))
    candidates = tmp_path / "c.json"
    candidates.write_text(
        json.dumps(
            [
                {"id": "small", "output": "weak", "model": "llama3.2:1b"},
                {"id": "large", "output": "best"},
                {"id": "mid", "output": "good"},
            ]
        ),
        encoding="utf-8",
    )
    code = main(["rank", "--task", "t", "--candidates-file", str(candidates)])
    assert code == 0
    out = capsys.readouterr().out
    assert out.index("1. large") < out.index("2. mid") < out.index("3. small")
    assert "W-L-T 2-0-0" in out


def test_rank_command_accepts_a_mapping_and_json_output(real_judge_with, capsys, tmp_path):
    from .conftest import SlotAwareProvider
    from .test_ranking import by_quality

    real_judge_with(SlotAwareProvider(by_quality))
    candidates = tmp_path / "c.json"
    candidates.write_text(json.dumps({"a": "best", "b": "weak"}), encoding="utf-8")
    main(["rank", "--task", "t", "--candidates-file", str(candidates), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["leaderboard"][0]["candidate"] == "a"


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps([1, 2]), json.dumps([{"id": "x"}, {"id": "x"}])],
)
def test_rank_command_rejects_bad_candidate_files(real_judge_with, tmp_path, content):
    from .conftest import FakeProvider

    real_judge_with(FakeProvider("{}"))
    path = tmp_path / "c.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["rank", "--task", "t", "--candidates-file", str(path)])


# --- providers, health, version, --provider ---------------------------------------


@pytest.fixture
def keyless_settings(monkeypatch):
    monkeypatch.setattr(config_module, "get_settings", lambda: Settings(_env_file=None))


def test_mock_provider_runs_end_to_end_without_a_key(keyless_settings, capsys):
    code = main(
        [
            "--provider",
            "mock",
            "evaluate",
            "--task",
            "Explain what an API is",
            "--output-a",
            "An API lets two programs talk to each other through a defined contract.",
            "--output-b",
            "Interface.",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Overall winner: A" in out
    assert "not an LLM" in out


def test_missing_key_error_suggests_the_mock_provider(keyless_settings, capsys):
    code = main(["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b"])
    assert code == 2
    assert "--provider mock" in capsys.readouterr().err


def test_providers_command_lists_status(keyless_settings, capsys):
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "* groq" in out and "not configured" in out
    assert "mock" in out and "NOT an LLM" in out


def test_health_command(keyless_settings, capsys, monkeypatch):
    assert main(["--provider", "mock", "health"]) == 0
    assert "OK" in capsys.readouterr().out

    from llm_judge.providers.base import ProviderHealth

    class Unhealthy:
        model = "m"

        def health(self):
            return ProviderHealth(False, "x", "m", "down")

    class NoHealth:
        model = "m"

    import llm_judge.providers.registry as registry

    monkeypatch.setattr(registry, "build_provider", lambda s: Unhealthy())
    assert main(["health"]) == 1
    monkeypatch.setattr(registry, "build_provider", lambda s: NoHealth())
    assert main(["health"]) == 0
    assert "no health check" in capsys.readouterr().out


def test_unknown_provider_exits_with_an_error(keyless_settings, capsys):
    assert main(["--provider", "nope", "health"]) == 2
    assert "Unknown provider" in capsys.readouterr().err


def test_version_command(capsys):
    from llm_judge import __version__

    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out
