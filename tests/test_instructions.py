"""Optional evaluator instructions (custom judging guidance) across every interface."""

from __future__ import annotations

import json

import pytest

from llm_judge.cli import main
from llm_judge.config import Settings
from llm_judge.criteria import resolve_criterion
from llm_judge.datasets import DatasetError, DatasetRunner, PairwiseRecord, PointwiseRecord
from llm_judge.experiments import ExperimentStore, compare_runs, record_dataset_run
from llm_judge.judge import Judge
from llm_judge.pointwise import build_pointwise_system_prompt
from llm_judge.prompts import PROMPT_VERSION, build_system_prompt, guidance_block
from llm_judge.providers import MockProvider

from .conftest import FnProvider, make_entry, make_payload, make_pointwise_payload

GUIDE = "Penalise answers that state facts without a source."


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="test-key", **overrides)


def _pairwise_provider() -> FnProvider:
    return FnProvider(lambda _: make_payload(make_entry("Helpfulness", 8, 4)))


def _system(provider: FnProvider, call: int = 0) -> str:
    return provider.messages[call][0]["content"]


# --- prompts ------------------------------------------------------------------------------


def test_guidance_block_is_empty_without_instructions():
    assert guidance_block(None) == ""
    assert guidance_block("   \n ") == ""


def test_guidance_sits_in_the_system_prompt_outside_the_data_markers():
    crit = [resolve_criterion("Helpfulness")]
    prompt = build_system_prompt(crit, boundary="TOKEN", instructions=GUIDE)
    assert "Evaluator guidance:\n" + GUIDE in prompt
    # Guidance comes before the untrusted-content section and never inside a marker.
    assert prompt.index(GUIDE) < prompt.index("Untrusted content:")
    assert "never changes the response format" in prompt
    assert build_system_prompt(crit, boundary="TOKEN") == build_system_prompt(
        crit, boundary="TOKEN", instructions="  "
    )


def test_pointwise_prompt_includes_guidance():
    crit = [resolve_criterion("Helpfulness")]
    prompt = build_pointwise_system_prompt(
        crit, boundary="T", with_reference=True, instructions=GUIDE
    )
    assert GUIDE in prompt and prompt.index(GUIDE) < prompt.index("Untrusted content:")


def test_prompt_version_was_bumped_for_the_new_section():
    assert PROMPT_VERSION == "2.1"


# --- judge --------------------------------------------------------------------------------


def test_pairwise_instructions_reach_the_judge_and_metadata():
    provider = _pairwise_provider()
    result = Judge(provider=provider, settings=_settings()).evaluate(
        "task", "a", "b", "Helpfulness", instructions=f"  {GUIDE}  "
    )
    assert GUIDE in _system(provider)
    assert result.metadata["instructions"] == GUIDE  # stripped


def test_no_instructions_means_no_guidance_and_no_metadata_key():
    provider = _pairwise_provider()
    result = Judge(provider=provider, settings=_settings()).evaluate(
        "task", "a", "b", "Helpfulness", instructions="   "
    )
    assert "Evaluator guidance" not in _system(provider)
    assert "instructions" not in result.metadata


def test_both_bias_mitigation_runs_get_the_guidance():
    provider = _pairwise_provider()
    Judge(provider=provider, settings=_settings(mitigate_position_bias=True)).evaluate(
        "task", "a", "b", "Helpfulness", instructions=GUIDE
    )
    assert provider.calls == 2
    assert all(GUIDE in m[0]["content"] for m in provider.messages)


def test_oversized_instructions_are_truncated_with_a_warning():
    provider = _pairwise_provider()
    judge = Judge(provider=provider, settings=_settings(max_instructions_chars=20))
    result = judge.evaluate("task", "a", "b", "Helpfulness", instructions="x" * 50)
    assert result.metadata["instructions"] == "x" * 20
    assert any(w.code == "input_truncated" and "instructions" in w.message for w in result.warnings)


def test_non_string_instructions_are_rejected():
    judge = Judge(provider=_pairwise_provider(), settings=_settings())
    with pytest.raises(Exception, match="instructions"):
        judge.evaluate("task", "a", "b", "Helpfulness", instructions=42)  # type: ignore[arg-type]


def test_pointwise_and_reference_pass_guidance():
    provider = FnProvider(
        lambda _: make_pointwise_payload(("Correctness", 7), ("Completeness", 6), missing=[])
    )
    judge = Judge(provider=provider, settings=_settings())
    graded = judge.evaluate_reference("task", "out", "ref", instructions=GUIDE)
    assert GUIDE in _system(provider)
    assert graded.metadata["instructions"] == GUIDE
    judge.evaluate_pointwise("task", "out", ["Correctness", "Completeness"], instructions=GUIDE)
    assert GUIDE in _system(provider, 1)


@pytest.mark.parametrize("strategy", ["round_robin", "pointwise"])
def test_ranking_passes_guidance_to_every_comparison(strategy):
    def reply(messages):
        if "Output A" in messages[0]["content"]:
            return make_payload(make_entry("Helpfulness", 8, 4))
        return make_pointwise_payload(("Helpfulness", 7))

    provider = FnProvider(reply)
    result = Judge(provider=provider, settings=_settings()).rank(
        "task", ["one", "two", "three"], strategy=strategy, instructions=GUIDE
    )
    assert provider.calls >= 3
    assert all(GUIDE in m[0]["content"] for m in provider.messages)
    assert result.metadata["instructions"] == GUIDE


# --- datasets and experiments -------------------------------------------------------------


def _rows() -> list[PointwiseRecord]:
    return [PointwiseRecord(id=str(i), task_description="t", output=f"out {i}") for i in range(3)]


def test_dataset_runner_applies_and_reports_instructions():
    provider = FnProvider(lambda _: make_pointwise_payload(("Helpfulness", 7)))
    report = DatasetRunner(Judge(provider=provider, settings=_settings()), instructions=GUIDE).run(
        _rows()
    )
    assert report.succeeded == 3
    assert report.instructions == GUIDE
    assert all(GUIDE in m[0]["content"] for m in provider.messages)


def test_resume_with_different_instructions_is_refused(tmp_path):
    checkpoint = tmp_path / "run.checkpoint.jsonl"
    judge = Judge(provider=MockProvider(), settings=_settings(provider="mock"))
    DatasetRunner(judge, checkpoint=checkpoint, instructions=GUIDE).run(_rows()[:1])
    with pytest.raises(DatasetError, match="different judge configuration"):
        DatasetRunner(judge, checkpoint=checkpoint, resume=True, instructions="Be lenient.").run(
            _rows()
        )


def test_recorded_runs_keep_instructions_and_comparisons_flag_differences():
    store = ExperimentStore(":memory:")
    judge = Judge(provider=MockProvider(), settings=_settings(provider="mock"))
    rows = [
        PairwiseRecord(id=str(i), task_description="t", output_a="a long answer", output_b="b")
        for i in range(3)
    ]
    base = record_dataset_run(store, DatasetRunner(judge).run(rows), judge, experiment="e")
    guided = record_dataset_run(
        store, DatasetRunner(judge, instructions=GUIDE).run(rows), judge, experiment="e"
    )
    assert store.get_run(guided.id).config["instructions"] == GUIDE
    assert store.get_run(base.id).config["instructions"] is None
    comparison = compare_runs(store, base.id, guided.id)
    assert any("judge instructions" in w for w in comparison.warnings)


# --- CLI ----------------------------------------------------------------------------------


def test_cli_reads_instructions_from_a_file(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    guide_file = tmp_path / "guide.txt"
    guide_file.write_text(GUIDE, encoding="utf-8")
    code = main(
        [
            "--provider",
            "mock",
            "evaluate",
            "--task",
            "Explain recursion",
            "--output-a",
            "A function that calls itself, with a base case.",
            "--output-b",
            "Stuff repeats.",
            "--instructions-file",
            str(guide_file),
            "--json",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["metadata"]["instructions"] == GUIDE


# --- REST API -----------------------------------------------------------------------------


def test_api_accepts_instructions_and_bounds_their_size():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from llm_judge.api import create_app

    settings = Settings(
        _env_file=None, provider="mock", db_path=":memory:", api_rate_limit_per_minute=0
    )
    client = TestClient(
        create_app(settings, store=ExperimentStore(":memory:")), raise_server_exceptions=False
    )
    body = {
        "task_description": "Explain what an API is",
        "output_a": "An API lets two programs talk through a defined contract.",
        "output_b": "Interface.",
        "instructions": GUIDE,
    }
    ok = client.post("/api/v1/evaluate/pairwise", json=body)
    assert ok.status_code == 200, ok.text
    assert ok.json()["metadata"]["instructions"] == GUIDE
    graded = client.post(
        "/api/v1/evaluate/pointwise",
        json={"task_description": "t", "output": "o", "instructions": GUIDE},
    )
    assert graded.json()["metadata"]["instructions"] == GUIDE
    too_long = client.post("/api/v1/evaluate/pairwise", json={**body, "instructions": "x" * 20_001})
    assert too_long.status_code == 422
