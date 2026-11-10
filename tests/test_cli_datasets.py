"""CLI: dataset-aware `batch`, `datasets` and `benchmark`, run with the offline mock judge."""

from __future__ import annotations

import json

import pytest

import llm_judge.config as config_module
from llm_judge.cli import main
from llm_judge.config import Settings


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setattr(
        config_module, "get_settings", lambda: Settings(_env_file=None, provider="mock")
    )


def _export(tmp_path, name: str):
    path = tmp_path / f"{name}.jsonl"
    assert main(["datasets", "export", name, "--output", str(path)]) == 0
    return path


def test_batch_runs_a_reference_dataset_and_cleans_up_its_checkpoint(tmp_path, capsys):
    data = _export(tmp_path, "general_qa")
    out = tmp_path / "out.csv"
    assert main(["batch", "--input", str(data), "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "reference run: 6/6 succeeded" in printed
    assert "mean token F1" in printed
    assert out.exists()
    assert not (tmp_path / "out.csv.checkpoint.jsonl").exists()


def test_batch_keeps_the_checkpoint_on_failure_and_resumes(tmp_path, capsys, monkeypatch):
    import llm_judge.providers.mock_provider as mock_module
    from llm_judge.exceptions import ProviderError

    data = _export(tmp_path, "summarization")
    out = tmp_path / "out.jsonl"
    original = mock_module.MockProvider.complete_json

    def flaky(self, messages):
        if "library" in messages[1]["content"]:
            raise ProviderError("temporarily down", retryable=False)
        return original(self, messages)

    monkeypatch.setattr(mock_module.MockProvider, "complete_json", flaky)
    assert main(["batch", "--input", str(data), "--output", str(out)]) == 1
    checkpoint = tmp_path / "out.jsonl.checkpoint.jsonl"
    assert checkpoint.exists()
    assert "--resume" in capsys.readouterr().err

    # Re-running without --resume refuses to mix runs.
    assert main(["batch", "--input", str(data), "--output", str(out)]) == 2
    assert "already exists" in capsys.readouterr().err

    monkeypatch.setattr(mock_module.MockProvider, "complete_json", original)
    assert main(["batch", "--input", str(data), "--output", str(out), "--resume"]) == 0
    assert "4 resumed" in capsys.readouterr().out
    assert not checkpoint.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5


def test_batch_rejects_invalid_data_unless_skipping(tmp_path, capsys):
    data = tmp_path / "bad.jsonl"
    data.write_text(
        json.dumps({"task_description": "t", "output_a": "a", "output_b": "b"})
        + "\n"
        + json.dumps({"task_description": "t", "output_a": "a"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    assert main(["batch", "--input", str(data), "--output", str(out), "--no-checkpoint"]) == 2
    captured = capsys.readouterr()
    assert "--skip-invalid" in captured.err
    assert "missing required field(s): output_b" in captured.out
    assert (
        main(
            [
                "batch",
                "--input",
                str(data),
                "--output",
                str(out),
                "--no-checkpoint",
                "--skip-invalid",
            ]
        )
        == 0
    )


def test_batch_sample_limit_criteria_and_json_summary(tmp_path, capsys):
    data = _export(tmp_path, "code_generation")
    out = tmp_path / "out.csv"
    code = main(
        [
            "batch",
            "--input",
            str(data),
            "--output",
            str(out),
            "--sample",
            "4",
            "--limit",
            "2",
            "--criteria",
            "Code Readability",
            "--json",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{") : printed.rindex("}") + 1])
    assert summary["total"] == 2 and summary["criteria"] == ["Code Readability"]


def test_batch_with_no_rows_errors(tmp_path, capsys):
    data = _export(tmp_path, "rag")
    assert (
        main(["batch", "--input", str(data), "--output", str(tmp_path / "o.csv"), "--limit", "0"])
        == 2
    )
    assert "no rows" in capsys.readouterr().err


def test_datasets_validate_and_samples(tmp_path, capsys):
    data = _export(tmp_path, "safety")
    assert main(["datasets", "validate", str(data)]) == 0
    assert "kind pairwise, 5 row(s)" in capsys.readouterr().out
    assert main(["datasets", "validate", str(data), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["valid_rows"] == 5

    bad = tmp_path / "bad.csv"
    bad.write_text("text\nhello\n", encoding="utf-8")
    assert main(["datasets", "validate", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().out

    assert main(["datasets", "samples"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in ("general_qa", "rag", "safety"))


def test_benchmark_a_sample_and_a_file(tmp_path, capsys):
    out = tmp_path / "rows.csv"
    assert main(["benchmark", "safety", "--output", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Benchmark 'safety' with provider mock" in printed
    assert "system-alpha" in printed and "win rate" in printed
    assert out.exists()

    assert main(["benchmark", "general_qa", "--limit", "2", "--json"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed[printed.index("{") : printed.rindex("}") + 1])["total"] == 2

    data = _export(tmp_path, "instruction_following")
    assert main(["benchmark", str(data), "--sample", "3", "--seed", "7"]) == 0
    assert "pointwise run: 3/3 succeeded" in capsys.readouterr().out
