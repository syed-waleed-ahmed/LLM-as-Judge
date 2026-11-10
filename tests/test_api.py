"""REST API (Phase 6): endpoints, security, error envelope, jobs and the serve command."""

from __future__ import annotations

import threading
import time
import warnings

import pytest

pytest.importorskip("fastapi")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

from llm_judge.api import create_app
from llm_judge.api.jobs import IdempotencyConflictError, JobManager, QueueFullError
from llm_judge.api.security import RateLimiter, request_id_from
from llm_judge.config import Settings
from llm_judge.datasets import PairwiseRecord, RunReport
from llm_judge.exceptions import ProviderError
from llm_judge.experiments import ExperimentStore
from llm_judge.judge import Judge
from llm_judge.providers import MockProvider

from .conftest import FakeProvider, FnProvider

PAIR = {
    "task_description": "Explain what an API is",
    "output_a": "An API lets two programs talk through a defined contract.",
    "output_b": "Interface.",
}


def _settings(**overrides) -> Settings:
    defaults = {"provider": "mock", "db_path": ":memory:", "api_rate_limit_per_minute": 0}
    return Settings(_env_file=None, **{**defaults, **overrides})


def _client(*, judge: Judge | None = None, store=None, **overrides) -> TestClient:
    app = create_app(
        _settings(**overrides), judge=judge, store=store or ExperimentStore(":memory:")
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client() -> TestClient:
    return _client()


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


# --- service endpoints --------------------------------------------------------------------


def test_health_and_openapi(client):
    assert client.get("/health").json()["status"] == "ok"
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/evaluate/pairwise" in paths and "/api/v1/jobs" in paths


def test_status_providers_criteria_and_benchmarks(client):
    status = client.get("/api/v1/system/status").json()
    assert status["provider"] == "mock" and status["model"] == "mock-heuristic-v1"
    assert status["provider_health"]["ok"] and status["auth_required"] is False
    assert status["limits"]["max_batch_rows"] == 50

    providers = {p["name"]: p for p in client.get("/api/v1/providers").json()}
    assert providers["mock"]["active"] and not providers["groq"]["configured"]

    criteria = client.get("/api/v1/criteria").json()
    assert "Helpfulness" in [c["name"] for c in criteria["presets"]]
    assert "Correctness" in [c["name"] for c in criteria["reference_presets"]]

    names = [b["name"] for b in client.get("/api/v1/benchmarks").json()]
    assert "general_qa" in names


def test_status_reports_an_unconfigured_provider():
    status = _client(provider="groq").get("/api/v1/system/status").json()
    assert status["model"] is None and status["provider_health"]["ok"] is False
    assert "GROQ_API_KEY" in status["provider_health"]["detail"]


# --- evaluation --------------------------------------------------------------------------------


def test_pairwise_pointwise_reference_and_rank(client):
    pairwise = client.post("/api/v1/evaluate/pairwise", json={**PAIR, "candidate_a_model": "m1"})
    assert pairwise.status_code == 200
    assert pairwise.json()["overall_winner"] == "A"
    assert pairwise.json()["metadata"]["candidates"]["A"] == {"model": "m1"}

    pointwise = client.post(
        "/api/v1/evaluate/pointwise",
        json={"task_description": "t", "output": "an answer", "criteria": ["Conciseness"]},
    )
    assert pointwise.json()["mode"] == "pointwise"

    reference = client.post(
        "/api/v1/evaluate/reference",
        json={"task_description": "Capital of France?", "output": "Paris", "reference": "Paris"},
    )
    body = reference.json()
    assert body["mode"] == "reference" and body["reference_metrics"]["exact_match"] is True

    rank = client.post(
        "/api/v1/evaluate/rank",
        json={
            "task_description": PAIR["task_description"],
            "candidates": {"good": PAIR["output_a"], "bad": PAIR["output_b"]},
            "strategy": "pointwise",
        },
    )
    assert rank.json()["leaderboard"][0]["candidate"] == "good"


def test_inline_criteria_with_scale_and_weight(client):
    response = client.post(
        "/api/v1/evaluate/pairwise",
        json={
            **PAIR,
            "criteria": [
                "Helpfulness",
                {"name": "Coverage", "description": "Covers it.", "scale_max": 100, "weight": 2},
            ],
            "mitigate_position_bias": True,
        },
    )
    body = response.json()
    assert [c["criterion"] for c in body["criteria"]] == ["Helpfulness", "Coverage"]
    assert body["metadata"]["position_bias_mitigated"] is True

    bad = client.post(
        "/api/v1/evaluate/pairwise",
        json={**PAIR, "criteria": [{"name": "X", "scale_min": 5, "scale_max": 5}]},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_input"


def test_error_envelope_and_request_ids(client):
    missing = client.post(
        "/api/v1/evaluate/pairwise",
        json={"task_description": "t"},
        headers={"X-Request-ID": "abc-1"},
    )
    assert missing.status_code == 422
    error = missing.json()["error"]
    assert error["code"] == "invalid_request" and error["request_id"] == "abc-1"
    assert missing.headers["X-Request-ID"] == "abc-1"
    assert {tuple(d["loc"]) for d in error["details"]} >= {("body", "output_a")}

    extra = client.post("/api/v1/evaluate/pairwise", json={**PAIR, "surprise": 1})
    assert extra.status_code == 422  # unknown fields are rejected

    blank = client.post("/api/v1/evaluate/pairwise", json={**PAIR, "output_b": "   "})
    assert blank.json()["error"]["code"] == "invalid_input"

    unknown = client.get("/api/v1/nope")
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "not_found"
    assert client.put("/health").json()["error"]["code"] == "method_not_allowed"

    assert len(request_id_from("bad id with spaces")) == 32
    assert request_id_from("ok_id.1") == "ok_id.1"


@pytest.mark.parametrize(
    "provider,status,code",
    [
        (
            FnProvider(lambda m: (_ for _ in ()).throw(ProviderError("busy", retryable=True))),
            503,
            "provider_unavailable",
        ),
        (
            FnProvider(lambda m: (_ for _ in ()).throw(ProviderError("denied", retryable=False))),
            502,
            "provider_error",
        ),
        (FakeProvider("not json"), 502, "judge_output_invalid"),
        (
            FnProvider(lambda m: (_ for _ in ()).throw(RuntimeError("secret internals"))),
            500,
            "internal_error",
        ),
    ],
)
def test_provider_failures_map_to_http_errors(provider, status, code):
    settings = _settings(max_retries=0, max_parse_retries=0)
    client = _client(judge=Judge(provider=provider, settings=settings), max_retries=0)
    response = client.post("/api/v1/evaluate/pairwise", json=PAIR)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "secret internals" not in response.text
    if status == 503:
        assert response.headers["Retry-After"] == "10"


def test_unconfigured_provider_is_503():
    response = _client(provider="groq").post("/api/v1/evaluate/pairwise", json=PAIR)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_not_configured"


# --- security ------------------------------------------------------------------------------------


def test_api_keys_are_enforced():
    client = _client(api_keys="k1, k2")
    assert client.get("/health").status_code == 200  # liveness stays open
    denied = client.get("/api/v1/criteria")
    assert denied.status_code == 401 and denied.headers["WWW-Authenticate"] == "Bearer"
    assert client.get("/api/v1/criteria", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/criteria", headers={"X-API-Key": "k2"}).status_code == 200
    assert client.get("/api/v1/criteria", headers={"Authorization": "Bearer k1"}).status_code == 200
    assert client.get("/api/v1/system/status", headers={"X-API-Key": "k1"}).json()["auth_required"]


def test_admin_routes_are_disabled_or_protected():
    assert _client().get("/api/v1/jobs").json()["error"]["code"] == "admin_disabled"
    client = _client(api_admin_key="root")
    assert client.get("/api/v1/jobs").status_code == 401
    assert client.get("/api/v1/jobs", headers={"X-Admin-Key": "root"}).status_code == 200


def test_rate_limiting():
    client = _client(api_rate_limit_per_minute=2)
    assert [client.get("/api/v1/criteria").status_code for _ in range(2)] == [200, 200]
    limited = client.get("/api/v1/criteria")
    assert limited.status_code == 429 and int(limited.headers["Retry-After"]) >= 1
    assert client.get("/health").status_code == 200  # never rate limited


def test_rate_limiter_refills_per_identity():
    now = [0.0]
    limiter = RateLimiter(60, clock=lambda: now[0])
    assert all(limiter.check("a") == 0 for _ in range(60))
    assert limiter.check("a") > 0 and limiter.check("b") == 0
    now[0] += 1.0  # one token per second at 60/minute
    assert limiter.check("a") == 0
    assert RateLimiter(0).check("x") == 0  # disabled


def test_body_size_limit_with_and_without_content_length():
    client = _client(api_max_body_bytes=300)
    big = {**PAIR, "output_a": "x" * 1000}
    declared = client.post("/api/v1/evaluate/pairwise", json=big)
    assert declared.status_code == 413
    assert declared.json()["error"]["code"] == "payload_too_large"

    def chunks():
        yield b'{"task_description": "t", "output_a": "'
        yield b"x" * 1000
        yield b'", "output_b": "b"}'

    streamed = client.post(
        "/api/v1/evaluate/pairwise", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert streamed.status_code == 413
    assert client.post("/api/v1/evaluate/pairwise", json=PAIR).status_code == 200


def test_concurrency_cap_sheds_load(client):
    slots = client.app.state.llm_judge.slots
    acquired = [slots.acquire(blocking=False) for _ in range(4)]
    try:
        busy = client.post("/api/v1/evaluate/pairwise", json=PAIR)
        assert busy.status_code == 503 and busy.headers["Retry-After"] == "5"
        assert busy.json()["error"]["code"] == "server_busy"
    finally:
        for ok in acquired:
            if ok:
                slots.release()
    assert client.post("/api/v1/evaluate/pairwise", json=PAIR).status_code == 200


def test_cors_is_opt_in():
    assert (
        "access-control-allow-origin"
        not in _client().get("/health", headers={"Origin": "https://x.example"}).headers
    )
    allowed = _client(api_cors_origins="https://x.example").get(
        "/health", headers={"Origin": "https://x.example"}
    )
    assert allowed.headers["access-control-allow-origin"] == "https://x.example"


# --- batch, datasets, jobs ---------------------------------------------------------------------


def test_synchronous_batch(client):
    rows = [{"id": str(i), **PAIR} for i in range(3)]
    body = client.post("/api/v1/evaluate/batch", json={"rows": rows}).json()
    assert body["kind"] == "pairwise" and body["summary"]["succeeded"] == 3
    assert [r["id"] for r in body["rows"]] == ["0", "1", "2"]

    invalid = client.post("/api/v1/evaluate/batch", json={"rows": [{"task_description": "t"}]})
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "invalid_dataset"

    too_many = _client(api_max_batch_rows=1).post("/api/v1/evaluate/batch", json={"rows": rows})
    assert too_many.status_code == 413 and "jobs" in too_many.json()["error"]["message"]


def test_dataset_validation(client):
    rows = client.post("/api/v1/datasets/validate", json={"rows": [PAIR, PAIR]}).json()
    assert rows["kind"] == "pairwise" and rows["duplicate_content"] == [["0", "1"]]

    csv_text = "task_description,output\nt,o\n,missing\n"
    report = client.post(
        "/api/v1/datasets/validate", json={"content": csv_text, "format": "csv"}
    ).json()
    assert report["source"] == "<upload.csv>" and report["valid_rows"] == 1

    assert client.post("/api/v1/datasets/validate", json={}).status_code == 422
    no_format = client.post("/api/v1/datasets/validate", json={"content": "x"})
    assert "'format' is required" in no_format.json()["error"]["message"]


def test_job_lifecycle_idempotency_and_experiment_recording(client):
    rows = [{"id": str(i), **PAIR} for i in range(4)]
    payload = {"rows": rows, "experiment": "api-exp", "run_name": "first"}
    created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "k1"})
    assert created.status_code == 202
    job_id = created.json()["id"]

    replay = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "k1"})
    assert replay.status_code == 200 and replay.json()["id"] == job_id
    conflict = client.post(
        "/api/v1/jobs", json={**payload, "run_name": "other"}, headers={"Idempotency-Key": "k1"}
    )
    assert conflict.status_code == 409

    job = _wait_for_job(client, job_id)
    assert job["status"] == "completed" and job["done"] == 4 and job["run_id"]
    assert job["summary"]["succeeded"] == 4
    page = client.get(f"/api/v1/jobs/{job_id}/rows", params={"offset": 1, "limit": 2}).json()
    assert page["total"] == 4 and [r["id"] for r in page["rows"]] == ["1", "2"]

    runs = client.get("/api/v1/runs", params={"experiment": "api-exp"}).json()
    assert runs["total"] == 1 and runs["rows"][0]["id"] == job["run_id"]
    assert client.get("/api/v1/jobs/missing").status_code == 404


def test_benchmark_job(client):
    created = client.post("/api/v1/benchmarks/rag/jobs", json={"experiment": "bench"})
    job = _wait_for_job(client, created.json()["id"])
    assert job["status"] == "completed" and job["kind"] == "reference" and job["total"] == 5
    run = client.get(f"/api/v1/runs/{job['run_id']}").json()
    assert run["dataset_source"] == "sample:rag" and run["criteria"] == [
        "Correctness",
        "Factual Consistency",
    ]
    assert client.post("/api/v1/benchmarks/nope/jobs", json={}).status_code == 404


def test_cancelling_a_running_job():
    release = threading.Event()

    class SlowMock(MockProvider):
        def complete_json(self, messages):
            release.wait(5)
            return super().complete_json(messages)

    client = _client(judge=Judge(provider=SlowMock(), settings=_settings()))
    rows = [{"id": str(i), **PAIR} for i in range(20)]
    job_id = client.post("/api/v1/jobs", json={"rows": rows}).json()["id"]
    cancelled = client.delete(f"/api/v1/jobs/{job_id}")
    assert cancelled.status_code == 200
    release.set()
    job = _wait_for_job(client, job_id)
    assert job["status"] == "cancelled" and job["done"] < 20


def test_job_manager_queue_failure_and_eviction():
    gate = threading.Event()
    record = PairwiseRecord(id="1", **PAIR)

    def blocking(records, stop, progress):
        gate.wait(5)
        return RunReport(kind="pairwise", outcomes=[], total=len(records))

    manager = JobManager(max_running=1, max_queued=1, max_retained=2)
    running, _ = manager.submit([record], "pairwise", blocking, payload=1)
    deadline = time.monotonic() + 5
    while running.status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    queued, _ = manager.submit([record], "pairwise", blocking, payload=2)
    with pytest.raises(QueueFullError):
        manager.submit([record], "pairwise", blocking, payload=3)
    assert manager.cancel(queued.id).status == "cancelled"
    assert manager.cancel("nope") is None
    gate.set()

    def failing(records, stop, progress):
        raise RuntimeError("boom")

    failed, _ = manager.submit([record], "pairwise", failing, payload=4, idempotency_key="x")
    with pytest.raises(IdempotencyConflictError):
        manager.submit([record], "pairwise", failing, payload=5, idempotency_key="x")
    deadline = time.monotonic() + 5
    while failed.status != "failed" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert failed.status == "failed" and failed.error == "boom"
    assert running.status == "completed"

    gate.clear()
    busy, _ = manager.submit([record], "pairwise", blocking, payload=6)
    deadline = time.monotonic() + 5
    while busy.status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)
    waiting, _ = manager.submit([record], "pairwise", blocking, payload=7)
    threading.Timer(0.1, gate.set).start()
    manager.shutdown()
    assert waiting.status == "cancelled"  # queued jobs don't linger as "queued"
    assert len(manager.list()) <= 4  # finished jobs are evicted past max_retained


# --- experiments ---------------------------------------------------------------------------------


def test_experiment_endpoints():
    store = ExperimentStore(":memory:")
    client = _client(store=store, api_admin_key="root")
    ids = []
    for name in ("v1", "v2"):
        job = client.post(
            "/api/v1/benchmarks/general_qa/jobs", json={"experiment": "qa", "run_name": name}
        ).json()
        ids.append(_wait_for_job(client, job["id"])["run_id"])

    assert [e["name"] for e in client.get("/api/v1/experiments").json()] == ["qa"]
    stats = client.get("/api/v1/statistics").json()
    assert stats == {
        "experiments": 1,
        "runs": 2,
        "rows_evaluated": 12,
        "runs_by_kind": {"reference": 2},
        "runs_by_status": {"completed": 2},
    }
    rows = client.get(f"/api/v1/runs/{ids[0]}/rows", params={"limit": 500}).json()
    assert rows["limit"] == 200 and rows["total"] == 6

    comparison = client.get(
        "/api/v1/runs/compare", params={"baseline": ids[0], "candidate": ids[1]}
    ).json()
    assert comparison["verdict"] == "no_significant_change"
    assert client.get("/api/v1/runs/compare", params={"baseline": ids[0]}).status_code == 422

    assert client.get("/api/v1/runs/missing").status_code == 404
    assert client.delete(f"/api/v1/runs/{ids[0]}").status_code == 401
    assert (
        client.delete(f"/api/v1/runs/{ids[0]}", headers={"X-Admin-Key": "root"}).status_code == 204
    )
    assert client.get(f"/api/v1/runs/{ids[0]}").status_code == 404


# --- serve command -----------------------------------------------------------------------------


def test_serve_command(monkeypatch, capsys):
    import uvicorn

    import llm_judge.config as config_module
    from llm_judge.cli import main

    calls = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.update(kw, app=app))
    monkeypatch.setattr(config_module, "get_settings", lambda: _settings())
    assert main(["serve"]) == 0
    assert calls["host"] == "127.0.0.1" and calls["port"] == 8000
    assert "warning" not in capsys.readouterr().err

    assert main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert "without JUDGE_API_KEYS" in capsys.readouterr().err
