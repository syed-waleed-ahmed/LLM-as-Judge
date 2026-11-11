"""Phase 9 hardening: hostile inputs, secrets hygiene, and concurrency under load."""

from __future__ import annotations

import csv
import json
import logging
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm_judge.api.security import RateLimiter
from llm_judge.config import Settings
from llm_judge.csv_safety import safe_cell, safe_rows
from llm_judge.datasets import (
    DatasetError,
    DatasetRunner,
    PairwiseRecord,
    iter_rows,
    load_dataset,
    read_checkpoint,
    write_outcomes,
)
from llm_judge.exceptions import ProviderError, ResponseParsingError
from llm_judge.judge import Judge
from llm_judge.parsing import parse_judge_result
from llm_judge.providers import CallBudget, GroqProvider, MockProvider, QuotaProvider

from .conftest import FakeProvider, FnProvider, make_entry, make_payload

ROOT = Path(__file__).resolve().parents[1]


# --- CSV injection ------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["=1+1", "+cmd", "-2+3", "@SUM(A1)", "\tx", "\rx"])
def test_formula_like_text_is_neutralised(value):
    assert safe_cell(value) == "'" + value


@pytest.mark.parametrize("value", ["plain text", "", -0.5, 3, None, True])
def test_other_values_are_untouched(value):
    assert safe_cell(value) == value


def test_result_csv_never_starts_a_cell_with_a_formula(tmp_path):
    payload = make_payload(
        make_entry("Helpfulness", 8, 4, '=HYPERLINK("http://evil.example","x")', "+1"),
        comment="@cmd",
    )
    judge = Judge(provider=FakeProvider(payload), settings=Settings(_env_file=None))
    record = PairwiseRecord(id="=1", task_description="t", output_a="a", output_b="b")
    report = DatasetRunner(judge).run([record])
    path = tmp_path / "out.csv"
    write_outcomes(report.outcomes, path)
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        for cell in row:
            assert not cell.startswith(("=", "+", "@")), cell
    assert rows[1][0] == "'=1"
    assert safe_rows([{"a": "=x", "b": 1}]) == [{"a": "'=x", "b": 1}]


# --- hostile files and replies ----------------------------------------------------------------


def test_deeply_nested_json_is_a_clean_error(tmp_path):
    deep = "[" * 100_000 + "]" * 100_000
    # Up to Python 3.13 the parser hits its recursion limit ("invalid JSON"); 3.14 can
    # parse this depth, and the result is then rejected for not being an object.
    # Either way the caller gets a clean DatasetError, never a RecursionError.
    rejected = "invalid JSON|expected a JSON object|is not an object"
    with pytest.raises(DatasetError, match=rejected):
        list(iter_rows(_write(tmp_path / "d.jsonl", deep + "\n")))
    with pytest.raises(DatasetError, match=rejected):
        list(iter_rows(_write(tmp_path / "d.json", deep)))
    with pytest.raises(ResponseParsingError):
        parse_judge_result("{" * 50_000)


def test_yaml_cannot_construct_python_objects(tmp_path):
    payload = "- !!python/object/apply:os.system ['echo pwned']\n"
    with pytest.raises(DatasetError, match="invalid YAML"):
        list(iter_rows(_write(tmp_path / "evil.yaml", payload)))


def test_oversized_and_overlong_datasets_are_refused_before_parsing(tmp_path):
    path = _write(tmp_path / "big.jsonl", json.dumps({"x": "y" * 1000}) + "\n")
    with pytest.raises(DatasetError, match="JUDGE_MAX_DATASET_BYTES"):
        load_dataset(path, settings=Settings(_env_file=None, max_dataset_bytes=100))


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --- secrets hygiene -----------------------------------------------------------------------------


def test_provider_errors_and_logs_never_contain_the_api_key(caplog):
    secret = "gsk_SECRET_DO_NOT_LEAK"
    provider = GroqProvider(api_key=secret, model="m")

    def boom(**kwargs):
        error = RuntimeError("upstream failure")
        error.status_code = 500  # type: ignore[attr-defined]
        raise error

    from types import SimpleNamespace

    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=boom))
    )
    judge = Judge(provider=provider, settings=Settings(_env_file=None, max_retries=0))
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ProviderError) as exc:
        judge.evaluate("t", "a", "b", "Helpfulness")
    assert secret not in str(exc.value)
    assert secret not in caplog.text


def test_api_responses_never_contain_secrets():
    pytest.importorskip("fastapi")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from llm_judge.api import create_app
    from llm_judge.experiments import ExperimentStore

    settings = Settings(
        _env_file=None,
        provider="mock",
        groq_api_key="gsk_SECRET_GROQ",
        openai_api_key="sk_SECRET_OPENAI",
        api_admin_key="ADMIN_SECRET",
        api_rate_limit_per_minute=0,
    )
    client = TestClient(create_app(settings, store=ExperimentStore(":memory:")))
    for path in ("/api/v1/system/status", "/api/v1/providers", "/openapi.json", "/health"):
        text = client.get(path).text
        assert "SECRET" not in text, path


# --- concurrency under load --------------------------------------------------------------------


def test_quota_is_never_overspent_under_contention():
    budget = CallBudget(100)
    provider = QuotaProvider(MockProvider(), [budget])
    judge = Judge(provider=provider, settings=Settings(_env_file=None, max_retries=0))

    def attempt(_: int) -> bool:
        try:
            judge.evaluate_pointwise("Explain DNS", "DNS maps names to IP addresses.")
            return True
        except ProviderError:
            return False

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(attempt, range(400)))
    assert sum(results) == 100
    assert budget.used == 100 and budget.remaining == 0


def test_rate_limiter_is_exact_under_contention():
    limiter = RateLimiter(10, clock=lambda: 1000.0)  # frozen clock: no refill
    barrier = threading.Barrier(64)

    def hit(_: int) -> bool:
        barrier.wait()
        return limiter.check("same-client") == 0

    with ThreadPoolExecutor(max_workers=64) as pool:
        allowed = sum(pool.map(hit, range(64)))
    assert allowed == 10


def test_runner_stress_keeps_order_and_checkpoint_integrity(tmp_path):
    payload = make_payload(make_entry("Helpfulness", 7, 5))
    rng = random.Random(0)
    lock = threading.Lock()

    def jittery(messages):
        with lock:
            delay = rng.random() / 500
        time.sleep(delay)  # completion order differs from submission order
        return payload

    judge = Judge(provider=FnProvider(jittery), settings=Settings(_env_file=None))
    records = [
        PairwiseRecord(id=f"r{i:03d}", task_description="t", output_a=f"a{i}", output_b="b")
        for i in range(400)
    ]
    checkpoint = tmp_path / "ckpt.jsonl"
    report = DatasetRunner(judge, max_workers=16, checkpoint=checkpoint).run(records)
    assert [o.id for o in report.outcomes] == [r.id for r in records]
    assert report.succeeded == 400 and report.complete
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 400 and all(json.loads(line)["ok"] for line in lines)
    assert len(read_checkpoint(checkpoint, "pairwise")) == 400

    calls_before = judge.provider.calls
    resumed = DatasetRunner(judge, checkpoint=checkpoint, resume=True).run(records)
    assert judge.provider.calls == calls_before and resumed.resumed == 400


# --- static analysis ----------------------------------------------------------------------------


def test_bandit_reports_no_unreviewed_issues():
    pytest.importorskip("bandit")
    result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        [
            sys.executable,
            "-m",
            "bandit",
            "-c",
            "pyproject.toml",
            "-q",
            "-r",
            "src",
            "app.py",
            "dashboard",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
