"""Regression tests for defects found by the independent code review."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_judge.agents import AgentTrace, render_trace
from llm_judge.config import Settings
from llm_judge.criteria import resolve_criterion
from llm_judge.datasets import (
    DatasetError,
    DatasetRunner,
    PairwiseRecord,
    iter_rows,
    load_dataset,
)
from llm_judge.datasets.records import canonical_columns
from llm_judge.exceptions import ResponseParsingError, ValidationError
from llm_judge.experiments import ExperimentStore, compare_runs, record_dataset_run
from llm_judge.judge import Judge
from llm_judge.models import Criterion, swap_labels
from llm_judge.providers import CallBudget, GroqProvider, MockProvider, QuotaProvider

from .conftest import FakeProvider, ScriptedProvider, make_pointwise_payload


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- datasets --------------------------------------------------------------------------------


def test_yaml_aliases_are_rejected(tmp_path):
    bomb = 'a: &a ["x","x","x","x","x","x","x","x","x","x"]\nb: &b [*a,*a,*a,*a,*a,*a,*a,*a]\n'
    path = tmp_path / "bomb.yaml"
    path.write_text(bomb + "records:\n  - {task_description: t, output: *b}\n", "utf-8")
    with pytest.raises(DatasetError, match="aliases are not allowed"):
        list(iter_rows(path))


def test_nested_field_values_are_rejected_not_stringified():
    records, report = load_dataset(
        [{"task_description": "t", "output": ["x"] * 3}, {"task_description": "t", "output": "ok"}],
        settings=_settings(),
    )
    assert [r.id for r in records] == ["1"]
    assert "must be text or numbers, not list" in report.issues[0].message


def test_padded_column_names_are_recognised(tmp_path):
    path = tmp_path / "padded.csv"
    path.write_text("id, task_description , output\n1, Explain DNS, It maps names.\n", "utf-8")
    records, report = load_dataset(path, settings=_settings())
    assert report.ok and records[0].task_description == " Explain DNS"


def test_two_aliases_for_one_field_do_not_overwrite_each_other():
    mapping = canonical_columns(["prompt", "question", "response"])
    assert mapping == {"prompt": "task_description", "question": "question", "response": "output"}
    records, report = load_dataset(
        [{"prompt": "Explain DNS", "question": "", "output": "It maps names."}],
        settings=_settings(),
    )
    assert report.ok and records[0].task_description == "Explain DNS"
    assert report.unknown_columns == ["question"]


def test_non_string_yaml_keys_do_not_crash(tmp_path):
    path = tmp_path / "keys.yaml"
    path.write_text("- {task_description: t, output: o, 7: extra}\n", "utf-8")
    records, report = load_dataset(path, settings=_settings())
    assert report.ok and report.unknown_columns == ["7"]


def test_encoding_and_csv_size_errors_are_dataset_errors(tmp_path):
    latin = tmp_path / "latin.csv"
    latin.write_bytes("task_description,output\ncaf\xe9,ok\n".encode("cp1252"))
    with pytest.raises(DatasetError, match="not valid UTF-8"):
        list(iter_rows(latin))
    big = tmp_path / "big.csv"
    big.write_text("task_description,output\nt," + "x" * 200_000 + "\n", "utf-8")
    assert len(list(iter_rows(big))[0]["output"]) == 200_000


def test_duplicate_id_issues_report_the_original_row():
    rows = [
        {"id": "a", "task_description": "t", "output": "o"},
        {"task_description": "t"},  # invalid, so it is skipped
        {"id": "b", "task_description": "t", "output": "o"},
        {"id": "b", "task_description": "t", "output": "o2"},
    ]
    _, report = load_dataset(rows, settings=_settings())
    duplicate = next(i for i in report.issues if i.field == "id")
    assert duplicate.row == 4


# --- judge and providers -------------------------------------------------------------------------


def test_label_swapping_handles_paired_labels():
    assert swap_labels("Output A and B are close.") == "Output B and A are close."
    assert swap_labels("Outputs A vs B differ") == "Outputs B vs A differ"
    assert swap_labels(swap_labels("Candidates A, B")) == "Candidates A, B"


def test_presets_match_case_insensitively_and_bad_names_raise_library_errors():
    assert (
        resolve_criterion("helpfulness").description == resolve_criterion("Helpfulness").description
    )
    assert resolve_criterion("  safety ").name == "Safety"
    for name in ("", "x" * 121):
        with pytest.raises(ValidationError, match="Invalid criterion"):
            resolve_criterion(name)


def test_groq_sdk_retries_are_disabled():
    provider = GroqProvider(api_key="k", model="m")
    assert provider._client.max_retries == 0


def test_groq_json_mode_rejections_get_a_repair_retry():
    provider = GroqProvider(api_key="k", model="m")
    calls = {"n": 0, "always_fail": False}

    class JSONValidateFailed(Exception):
        status_code = 400
        body = {"error": {"code": "json_validate_failed", "failed_generation": "{broken"}}

    def create(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1 or calls["always_fail"]:
            raise JSONValidateFailed("Error code: 400 - json_validate_failed")
        content = json.dumps(make_pointwise_payload(("Helpfulness", 7)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    judge = Judge(provider=provider, settings=_settings(max_retries=0))
    result = judge.evaluate_pointwise("t", "o")
    assert result.criteria[0].score == 7 and calls["n"] == 2

    calls["always_fail"] = True
    with pytest.raises(ResponseParsingError) as exc:
        Judge(provider=provider, settings=_settings(max_parse_retries=0)).evaluate_pointwise(
            "t", "o"
        )
    assert exc.value.raw_response == "{broken"


def test_repair_conversation_echoes_the_rejected_reply():
    provider = ScriptedProvider(["not json", make_pointwise_payload(("Helpfulness", 5))])
    Judge(provider=provider, settings=_settings()).evaluate_pointwise("t", "o")
    assert provider.messages[1][2] == {"role": "assistant", "content": "not json"}


# --- runner, experiments, agents ------------------------------------------------------------------


def _pairs(n: int) -> list[PairwiseRecord]:
    return [
        PairwiseRecord(id=f"r{i}", task_description="Explain DNS", output_a=f"a{i}", output_b="b")
        for i in range(n)
    ]


def test_an_exhausted_budget_stops_the_run_with_a_reason():
    budget = CallBudget(2, name="session budget")
    judge = Judge(provider=QuotaProvider(MockProvider(), [budget]), settings=_settings())
    report = DatasetRunner(judge, max_workers=1).run(_pairs(6))
    assert report.interrupted and report.stop_reason and "session budget" in report.stop_reason
    assert report.succeeded == 2 and len(report.outcomes) < 6


def test_run_comparisons_count_failed_rows_and_refuse_thin_evidence():
    from llm_judge.exceptions import ProviderError

    from .conftest import FnProvider, slots

    store = ExperimentStore(":memory:")
    judge = Judge(provider=MockProvider(), settings=_settings())
    base = record_dataset_run(store, DatasetRunner(judge).run(_pairs(10)), judge)

    def mostly_down(messages):
        if slots(messages)[0] not in ("a0", "a1"):
            raise ProviderError("judge down", retryable=False)
        return json.loads(MockProvider().complete_json(messages))

    failing = Judge(provider=FnProvider(mostly_down), settings=_settings())
    cand = record_dataset_run(store, DatasetRunner(failing).run(_pairs(10)), failing)
    comparison = compare_runs(store, base.id, cand.id)
    assert comparison.verdict == "insufficient_data"
    assert comparison.candidate_failed_rows == 8 and comparison.matched_rows == 2
    assert any("failed in the candidate run" in w for w in comparison.warnings)
    assert any("too few for a reliable verdict" in w for w in comparison.warnings)

    partial = DatasetRunner(judge).run(_pairs(4))
    short = record_dataset_run(store, partial, judge)
    warnings = compare_runs(store, base.id, short.id).warnings
    assert any("missing from the candidate run" in w for w in warnings)


def test_comparisons_notice_changed_criterion_weights():
    store = ExperimentStore(":memory:")
    judge = Judge(provider=MockProvider(), settings=_settings())
    light = Criterion(name="Helpfulness", description="d", weight=1)
    heavy = Criterion(name="Helpfulness", description="d", weight=5)
    a = record_dataset_run(store, DatasetRunner(judge, [light]).run(_pairs(3)), judge)
    b = record_dataset_run(store, DatasetRunner(judge, [heavy]).run(_pairs(3)), judge)
    assert a.config["criteria"][0]["weight"] == 1
    warnings = compare_runs(store, a.id, b.id).warnings
    assert any("criterion definitions" in w for w in warnings)


def test_run_lookup_treats_wildcards_literally():
    store = ExperimentStore(":memory:")
    judge = Judge(provider=MockProvider(), settings=_settings())
    record_dataset_run(store, DatasetRunner(judge).run(_pairs(1)), judge)
    assert store.get_run("%") is None and store.get_run("_") is None


def test_long_agent_traces_keep_the_final_answer():
    steps = [{"type": "message", "content": "x" * 1500} for _ in range(5)]
    trace = AgentTrace(task="Do it", steps=steps, final_answer="The answer is 42.")
    text = render_trace(trace, max_chars=4000)
    assert len(text) <= 4000 and text.endswith("Final answer: The answer is 42.")
    assert "later step(s) omitted" in text

    provider = FakeProvider(make_pointwise_payload(("Task Completion", 6)))
    judge = Judge(provider=provider, settings=_settings(max_input_chars=4000))
    result = judge.evaluate_agent(trace, "Task Completion")
    assert "trace_truncated" in [c.code for c in result.checks]
    assert "The answer is 42." in provider.messages[0][1]["content"]


# --- CLI ---------------------------------------------------------------------------------------


@pytest.fixture
def mock_cli(monkeypatch, tmp_path):
    import llm_judge.config as config_module

    settings = Settings(_env_file=None, provider="mock", db_path=str(tmp_path / "exp.db"))
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    return tmp_path


def test_unsupported_output_is_rejected_before_any_evaluation(mock_cli, capsys, monkeypatch):
    from llm_judge.cli import main
    from llm_judge.providers import mock_provider

    def never(*args, **kwargs):
        raise AssertionError("no evaluation should run")

    monkeypatch.setattr(mock_provider.MockProvider, "complete_json", never)
    data = mock_cli / "qa.jsonl"
    main(["datasets", "export", "general_qa", "--output", str(data)])
    assert main(["batch", "--input", str(data), "--output", str(mock_cli / "r.txt")]) == 2
    assert "unsupported output format" in capsys.readouterr().err
    assert main(["benchmark", "safety", "--output", str(mock_cli / "r.txt")]) == 2


def test_batch_json_output_is_pure_json(mock_cli, capsys):
    from llm_judge.cli import main

    data = mock_cli / "qa.jsonl"
    main(["datasets", "export", "general_qa", "--output", str(data)])
    capsys.readouterr()
    out = mock_cli / "r.csv"
    assert (
        main(["batch", "--input", str(data), "--output", str(out), "--json", "--experiment", "x"])
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["total"] == 6  # the whole of stdout parses
    assert "Recorded run" in captured.err and "Evaluating" in captured.err


def test_consistency_exit_codes_and_temperature_range(mock_cli, monkeypatch, capsys):
    from llm_judge.cli import main
    from llm_judge.exceptions import ProviderError
    from llm_judge.providers import mock_provider

    with pytest.raises(SystemExit) as exc:
        main(["consistency", "--task", "t", "--output", "o", "--temperature", "9"])
    assert exc.value.code == 2

    def down(self, messages):
        raise ProviderError("unreachable", retryable=False)

    monkeypatch.setattr(mock_provider.MockProvider, "complete_json", down)
    assert main(["consistency", "--task", "t", "--output", "o", "--samples", "2"]) == 1


def test_cli_usage_errors_exit_2_and_respect_log_level(mock_cli, capsys, monkeypatch):
    import logging

    from llm_judge.cli import main

    long_name = "x" * 200
    args = ["evaluate", "--task", "t", "--output-a", "a", "--output-b", "b"]
    assert main([*args, "--criteria", long_name]) == 2
    assert "Invalid criterion" in capsys.readouterr().err

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    main(["version"])
    assert logging.getLogger("llm_judge").level == logging.DEBUG


# --- REST API ------------------------------------------------------------------------------------


def _api(**overrides):
    pytest.importorskip("fastapi")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from llm_judge.api import create_app

    settings = Settings(_env_file=None, provider="mock", db_path=":memory:", **overrides)
    return TestClient(create_app(settings, store=ExperimentStore(":memory:")))


def test_random_api_keys_cannot_bypass_the_rate_limit():
    client = _api(api_rate_limit_per_minute=2)
    codes = [
        client.get("/api/v1/criteria", headers={"X-API-Key": f"random-{i}"}).status_code
        for i in range(4)
    ]
    assert codes == [200, 200, 429, 429]


def test_failed_authentication_is_rate_limited_and_precedes_body_parsing():
    client = _api(api_keys="secret", api_rate_limit_per_minute=3)
    bad = client.post(
        "/api/v1/evaluate/pairwise",
        content=b"{not json",
        headers={"Content-Type": "application/json"},
    )
    assert bad.status_code == 401  # not a 422 about the body
    codes = [
        client.get("/api/v1/criteria", headers={"X-API-Key": "guess"}).status_code for _ in range(3)
    ]
    assert codes[-1] == 429


def test_api_rejects_overlong_criteria_and_literal_wildcards():
    client = _api(api_rate_limit_per_minute=0)
    body = {"task_description": "t", "output_a": "a", "output_b": "b", "criteria": ["x" * 200]}
    response = client.post("/api/v1/evaluate/pairwise", json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    assert client.get("/api/v1/runs/%25").status_code == 404
    assert client.get("/api/v1/runs/____").status_code == 404


# --- dashboard ------------------------------------------------------------------------------------


def test_dataset_criteria_follow_the_selected_sample(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JUDGE_PROVIDER", "mock")
    monkeypatch.setenv("JUDGE_DB_PATH", str(tmp_path / "e.db"))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    at.run()
    at.switch_page("dashboard/views/datasets.py")
    at.run()
    assert at.multiselect(key="ds:general_qa:criteria").value == ["Correctness", "Completeness"]
    at.selectbox(key="ds:sample").select("summarization").run()
    assert at.multiselect(key="ds:summarization:criteria").value == [
        "Factual Accuracy",
        "Conciseness",
    ]
    at.multiselect(key="ds:summarization:criteria").set_value([]).run()
    at.button(key="ds:run").click().run()
    assert any("Choose at least one criterion" in e.value for e in at.error)
    assert not at.exception
