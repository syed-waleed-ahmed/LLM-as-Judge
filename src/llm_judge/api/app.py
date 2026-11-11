"""FastAPI application factory for the LLM-as-Judge REST API.

Run it with ``llm-judge serve`` (``pip install "llm-as-judge-platform[api]"``), then open
``/docs`` for the interactive OpenAPI documentation.

Security model:

* ``JUDGE_API_KEYS`` (comma-separated) set: every ``/api/v1`` route needs
  ``X-API-Key: <key>`` or ``Authorization: Bearer <key>``. Unset: the API is
  open, which is only appropriate on localhost or behind your own gateway.
* Admin routes (deleting runs, listing all jobs) are disabled unless
  ``JUDGE_API_ADMIN_KEY`` is set, and they always require it.
* Per-client rate limiting, a request-size limit and a concurrency cap protect
  free-tier provider quotas; excess load gets 429 or 503 with ``Retry-After``.
* Errors share one JSON envelope with a request id; internals are never leaked.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__
from ..agents import AgentResult
from ..config import Settings, get_settings
from ..criteria import CRITERIA_PRESETS, REFERENCE_CRITERIA_PRESETS, resolve_criterion
from ..datasets import (
    SAMPLE_DATASETS,
    DatasetError,
    DatasetReport,
    DatasetRunner,
    Record,
    RunReport,
    kind_of,
    load_dataset,
    load_sample,
    summarize,
)
from ..exceptions import (
    ConfigurationError,
    LLMJudgeError,
    ProviderError,
    ResponseParsingError,
    ValidationError,
)
from ..experiments import (
    Experiment,
    ExperimentError,
    ExperimentStore,
    RunComparison,
    RunInfo,
    compare_runs,
    record_dataset_run,
    run_report,
)
from ..judge import Judge
from ..models import Criterion, JudgeResult, PointwiseResult, RankingResult
from ..rag import RagResult
from ..reliability import ConsistencyReport, JudgeAgreement, judge_agreement, measure_consistency
from .jobs import IdempotencyConflictError, Job, JobManager, QueueFullError
from .schemas import (
    AgentRequest,
    BatchRequest,
    BatchResponse,
    BenchmarkJobRequest,
    ConsistencyRequest,
    CriteriaField,
    CriteriaOut,
    CriterionInput,
    ErrorResponse,
    ExperimentDetail,
    JobInfo,
    JobRequest,
    PairwiseRequest,
    PointwiseRequest,
    ProviderHealthOut,
    ProviderOut,
    RagRequest,
    RankRequest,
    ReferenceRequest,
    RowResult,
    RowsPage,
    SampleOut,
    StatisticsOut,
    StatusOut,
    ValidateRequest,
)
from .security import (
    ApiError,
    BodySizeLimitMiddleware,
    RateLimiter,
    client_identity,
    key_matches,
    parse_keys,
    presented_key,
    request_id_from,
)

access_log = logging.getLogger("llm_judge.api.access")

_MAX_PAGE = 200
_HEALTH_TTL_SECONDS = 30.0


class _State:
    """Lazily-built shared resources for one app instance."""

    def __init__(self, settings: Settings, judge: Judge | None, store: ExperimentStore | None):
        self.settings = settings
        self._judge = judge
        self._store = store
        self._lock = threading.Lock()
        self.api_keys = parse_keys(settings.api_keys)
        self.limiter = RateLimiter(settings.api_rate_limit_per_minute)
        self.slots = threading.BoundedSemaphore(settings.api_max_concurrent)
        self.jobs = JobManager(max_running=1, max_queued=settings.api_max_queued_jobs)
        self._health: tuple[float, ProviderHealthOut] | None = None

    def judge(self, mitigate: bool | None = None) -> Judge:
        with self._lock:
            if self._judge is None:
                self._judge = Judge(settings=self.settings)  # may raise ConfigurationError
            base = self._judge
        if mitigate is None or mitigate == base.settings.mitigate_position_bias:
            return base
        return Judge(
            provider=base.provider,
            settings=base.settings.model_copy(update={"mitigate_position_bias": mitigate}),
        )

    def store(self) -> ExperimentStore:
        with self._lock:
            if self._store is None:
                self._store = ExperimentStore(self.settings.db_path)
            return self._store

    def provider_health(self) -> ProviderHealthOut:
        now = time.monotonic()
        if self._health and now - self._health[0] < _HEALTH_TTL_SECONDS:
            return self._health[1]
        try:
            provider = self.judge().provider
            check = getattr(provider, "health", None)
            if check is None:
                health = ProviderHealthOut(ok=True, detail="no health check available")
            else:
                result = check()
                health = ProviderHealthOut(ok=result.ok, detail=result.detail)
        except LLMJudgeError as exc:
            health = ProviderHealthOut(ok=False, detail=str(exc))
        self._health = (now, health)
        return health

    @contextmanager
    def evaluation_slot(self) -> Iterator[None]:
        """Cap concurrent evaluations; shed load with 503 instead of queueing silently."""
        if not self.slots.acquire(blocking=False):
            raise ApiError(
                503,
                "server_busy",
                "Too many evaluations in progress; retry shortly.",
                headers={"Retry-After": "5"},
            )
        try:
            yield
        finally:
            self.slots.release()


def _criteria(items: CriteriaField | None) -> list[Criterion | str] | None:
    if items is None:
        return None
    out: list[Criterion | str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        assert isinstance(item, CriterionInput)
        base = resolve_criterion(item.name, item.description)
        try:
            out.append(
                Criterion(
                    name=item.name,
                    description=item.description or base.description,
                    scale_min=item.scale_min,
                    scale_max=item.scale_max,
                    weight=item.weight,
                )
            )
        except PydanticValidationError as exc:
            raise ApiError(
                422,
                "invalid_input",
                f"Invalid criterion '{item.name}'.",
                details=json.loads(exc.json()),
            ) from None
    return out


def _error_response(
    request: Request,
    status: int,
    code: str,
    message: str,
    *,
    headers: Any = None,
    details: Any = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    body = {
        "error": {"code": code, "message": message, "request_id": request_id, "details": details}
    }
    response = JSONResponse(body, status_code=status, headers=headers)
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


def _map_library_error(exc: LLMJudgeError) -> tuple[int, str, dict[str, str] | None]:
    if isinstance(exc, DatasetError):
        return 422, "invalid_dataset", None
    if isinstance(exc, ValidationError):
        return 422, "invalid_input", None
    if isinstance(exc, ConfigurationError):
        return 503, "provider_not_configured", None
    if isinstance(exc, ResponseParsingError):
        return 502, "judge_output_invalid", None
    if isinstance(exc, ProviderError):
        if exc.retryable:
            return 503, "provider_unavailable", {"Retry-After": "10"}
        return 502, "provider_error", None
    if isinstance(exc, ExperimentError):
        return 400, "experiment_error", None
    return 400, "bad_request", None


def _page(offset: int, limit: int) -> tuple[int, int]:
    return max(0, offset), max(1, min(limit, _MAX_PAGE))


def _row_result(outcome: Any) -> RowResult:
    return RowResult(id=outcome.id, ok=outcome.ok, error=outcome.error, result=outcome.result)


def create_app(
    settings: Settings | None = None,
    *,
    judge: Judge | None = None,
    store: ExperimentStore | None = None,
) -> FastAPI:
    """Build the API. ``judge`` and ``store`` can be injected (e.g. for tests)."""
    settings = settings or get_settings()
    state = _State(settings, judge, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        state.jobs.shutdown()

    app = FastAPI(
        title="LLM-as-Judge API",
        version=__version__,
        description=(
            "Evaluate and compare LLM outputs: pairwise, pointwise, reference-based and "
            "ranking evaluation, dataset jobs, and experiment tracking."
        ),
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    app.state.llm_judge = state

    # Middleware: later additions wrap earlier ones, so the request-context middleware
    # (added last) runs first and its request id is available to the size limiter.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.api_max_body_bytes)
    origins = [o.strip() for o in settings.api_cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
        )

    def guard(request: Request) -> JSONResponse | None:
        """Authentication and rate limiting for /api/v1, applied before body parsing.

        The rate-limit identity is the authenticated key, or the client address;
        an unverified header is never trusted as an identity. Failed
        authentication attempts are rate-limited by address too.
        """
        if not request.url.path.startswith("/api/v1"):
            return None
        host = request.client.host if request.client else None
        key = presented_key(request.headers, "x-api-key")
        authenticated = bool(state.api_keys) and key_matches(key, state.api_keys)
        wait = state.limiter.check(client_identity(key if authenticated else None, host))
        if wait:
            return _error_response(
                request,
                429,
                "rate_limited",
                "Rate limit exceeded; slow down.",
                headers={"Retry-After": str(max(1, round(wait)))},
            )
        if state.api_keys and not authenticated:
            return _error_response(
                request,
                401,
                "unauthorized",
                "A valid API key is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        request_id = request_id_from(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        started = time.perf_counter()
        response: Response = guard(request) or await call_next(request)
        response.headers["X-Request-ID"] = request_id
        access_log.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                }
            )
        )
        return response

    # -- error handlers ----------------------------------------------------------------

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            request, exc.status, exc.code, exc.message, headers=exc.headers, details=exc.details
        )

    @app.exception_handler(LLMJudgeError)
    async def _library_error(request: Request, exc: LLMJudgeError) -> JSONResponse:
        status, code, headers = _map_library_error(exc)
        return _error_response(request, status, code, str(exc), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        return _error_response(
            request, 422, "invalid_request", "Request validation failed.", details=details
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error_response(
            request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("llm_judge.api").exception("Unhandled error")
        return _error_response(request, 500, "internal_error", "An unexpected error occurred.")

    # -- dependencies ------------------------------------------------------------------

    def require_admin(request: Request) -> None:
        admin = settings.api_admin_key
        if not admin:
            raise ApiError(403, "admin_disabled", "Admin endpoints are disabled on this server.")
        if not key_matches(presented_key(request.headers, "x-admin-key"), [admin]):
            raise ApiError(401, "unauthorized", "A valid admin key is required.")

    api = APIRouter(prefix="/api/v1")  # authentication runs in the middleware
    admin = [Depends(require_admin)]

    # -- service ---------------------------------------------------------------------

    @app.get("/health", tags=["service"])
    def health() -> dict[str, str]:
        """Liveness probe (no authentication, no rate limit)."""
        return {"status": "ok", "version": __version__}

    @api.get("/system/status", response_model=StatusOut, tags=["service"])
    def system_status() -> StatusOut:
        try:
            model: str | None = state.judge().provider.model
        except LLMJudgeError:
            model = None
        return StatusOut(
            version=__version__,
            provider=settings.provider,
            model=model,
            provider_health=state.provider_health(),
            auth_required=bool(state.api_keys),
            limits={
                "rate_limit_per_minute": settings.api_rate_limit_per_minute,
                "max_body_bytes": settings.api_max_body_bytes,
                "max_concurrent_evaluations": settings.api_max_concurrent,
                "max_batch_rows": settings.api_max_batch_rows,
                "max_job_rows": settings.api_max_job_rows,
                "max_queued_jobs": settings.api_max_queued_jobs,
                "max_criteria": settings.max_criteria,
                "max_candidates": settings.max_candidates,
                "max_input_chars": settings.max_input_chars,
            },
        )

    @api.get("/providers", response_model=list[ProviderOut], tags=["service"])
    def providers() -> list[ProviderOut]:
        from ..providers.registry import configuration_status, provider_info, provider_names

        out = []
        for name in provider_names():
            info = provider_info(name)
            configured, detail = configuration_status(name, settings)
            out.append(
                ProviderOut(
                    name=name,
                    description=info.description,
                    requires_api_key=info.requires_api_key,
                    local=info.local,
                    cost=info.cost,
                    configured=configured,
                    detail=detail,
                    active=name == settings.provider,
                )
            )
        return out

    @api.get("/criteria", response_model=CriteriaOut, tags=["service"])
    def criteria() -> CriteriaOut:
        return CriteriaOut(
            presets=list(CRITERIA_PRESETS.values()),
            reference_presets=list(REFERENCE_CRITERIA_PRESETS.values()),
        )

    # -- evaluation ------------------------------------------------------------------

    @api.post("/evaluate/pairwise", response_model=JudgeResult, tags=["evaluate"])
    def evaluate_pairwise(body: PairwiseRequest) -> JudgeResult:
        with state.evaluation_slot():
            return state.judge(body.mitigate_position_bias).evaluate(
                body.task_description,
                body.output_a,
                body.output_b,
                _criteria(body.criteria) or ["Helpfulness"],
                candidate_a_model=body.candidate_a_model,
                candidate_b_model=body.candidate_b_model,
                instructions=body.instructions,
            )

    @api.post("/evaluate/pointwise", response_model=PointwiseResult, tags=["evaluate"])
    def evaluate_pointwise(body: PointwiseRequest) -> PointwiseResult:
        with state.evaluation_slot():
            return state.judge().evaluate_pointwise(
                body.task_description,
                body.output,
                _criteria(body.criteria) or ["Helpfulness"],
                candidate_model=body.candidate_model,
                instructions=body.instructions,
            )

    @api.post("/evaluate/reference", response_model=PointwiseResult, tags=["evaluate"])
    def evaluate_reference(body: ReferenceRequest) -> PointwiseResult:
        with state.evaluation_slot():
            return state.judge().evaluate_reference(
                body.task_description,
                body.output,
                body.reference,
                _criteria(body.criteria),
                candidate_model=body.candidate_model,
                instructions=body.instructions,
            )

    @api.post("/evaluate/rank", response_model=RankingResult, tags=["evaluate"])
    def evaluate_rank(body: RankRequest) -> RankingResult:
        with state.evaluation_slot():
            return state.judge(body.mitigate_position_bias).rank(
                body.task_description,
                body.candidates,
                _criteria(body.criteria) or ["Helpfulness"],
                strategy=body.strategy,
                candidate_models=body.candidate_models,
                instructions=body.instructions,
            )

    @api.post("/evaluate/rag", response_model=RagResult, tags=["evaluate"])
    def evaluate_rag(body: RagRequest) -> RagResult:
        with state.evaluation_slot():
            return state.judge().evaluate_rag(
                body.question, body.contexts, body.answer, reference=body.reference
            )

    @api.post("/evaluate/agent", response_model=AgentResult, tags=["evaluate"])
    def evaluate_agent(body: AgentRequest) -> AgentResult:
        with state.evaluation_slot():
            return state.judge().evaluate_agent(
                body.trace, _criteria(body.criteria), candidate_model=body.candidate_model
            )

    @api.post("/analysis/consistency", response_model=ConsistencyReport, tags=["analysis"])
    def consistency(body: ConsistencyRequest) -> ConsistencyReport:
        with state.evaluation_slot():
            return measure_consistency(
                state.judge(),
                body.task_description,
                output_a=body.output_a,
                output_b=body.output_b,
                output=body.output,
                criteria=_criteria(body.criteria) or ["Helpfulness"],
                samples=body.samples,
                max_workers=min(settings.batch_max_workers, 2),
            )

    def _load_rows(rows: list[dict[str, Any]], kind: Any) -> list[Record]:
        records, report = load_dataset(rows, kind=kind, settings=settings)
        if not report.ok:
            raise ApiError(
                422,
                "invalid_dataset",
                f"{report.error_count} row error(s); see details.",
                details=[i.model_dump() for i in report.issues],
            )
        return records

    @api.post("/evaluate/batch", response_model=BatchResponse, tags=["evaluate"])
    def evaluate_batch(body: BatchRequest) -> BatchResponse:
        if len(body.rows) > settings.api_max_batch_rows:
            raise ApiError(
                413,
                "too_many_rows",
                f"At most {settings.api_max_batch_rows} rows per synchronous batch; "
                "use /api/v1/jobs for larger datasets.",
            )
        records = _load_rows(body.rows, body.kind)
        with state.evaluation_slot():
            runner = DatasetRunner(
                state.judge(body.mitigate_position_bias),
                _criteria(body.criteria),
                max_workers=min(settings.batch_max_workers, settings.api_max_concurrent),
                instructions=body.instructions,
            )
            report = runner.run(records)
        return BatchResponse(
            kind=report.kind,
            summary=summarize(report),
            rows=[_row_result(o) for o in report.outcomes],
        )

    # -- jobs ----------------------------------------------------------------------------

    def _submit_job(
        records: list[Record],
        criteria_items: CriteriaField | list[str] | None,
        mitigate: bool | None,
        experiment: str | None,
        run_name: str | None,
        idempotency_key: str | None,
        payload: Any,
        response: Response,
        source: str,
        instructions: str | None = None,
    ) -> JobInfo:
        if len(records) > settings.api_max_job_rows:
            raise ApiError(
                413, "too_many_rows", f"At most {settings.api_max_job_rows} rows per job."
            )
        judge = state.judge(mitigate)
        crit = _criteria(criteria_items)  # type: ignore[arg-type]

        def run(recs: list[Record], stop: threading.Event, progress: Any) -> RunReport:
            return DatasetRunner(
                judge,
                crit,
                max_workers=min(settings.batch_max_workers, settings.api_max_concurrent),
                stop_event=stop,
                progress=progress,
                instructions=instructions,
            ).run(recs)

        def record(report: RunReport, job: Job) -> str | None:
            if not job.experiment:
                return None
            info = record_dataset_run(
                state.store(),
                report,
                judge,
                experiment=job.experiment,
                name=job.run_name,
                dataset_source=source,
            )
            return info.id

        try:
            job, created = state.jobs.submit(
                records,
                kind_of(records[0]),
                run,
                payload=payload,
                idempotency_key=idempotency_key,
                experiment=experiment,
                run_name=run_name,
                record=record,
            )
        except QueueFullError:
            raise ApiError(
                429,
                "queue_full",
                "The job queue is full; retry later.",
                headers={"Retry-After": "30"},
            ) from None
        except IdempotencyConflictError:
            raise ApiError(
                409,
                "idempotency_conflict",
                "This Idempotency-Key was already used with a different payload.",
            ) from None
        response.status_code = 202 if created else 200
        return job.info()

    @api.post("/jobs", response_model=JobInfo, status_code=202, tags=["jobs"])
    def create_job(
        body: JobRequest,
        response: Response,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> JobInfo:
        """Queue a dataset evaluation. Poll ``GET /jobs/{id}`` for progress and results."""
        records = _load_rows(body.rows, body.kind)
        return _submit_job(
            records,
            body.criteria,
            body.mitigate_position_bias,
            body.experiment,
            body.run_name,
            idempotency_key,
            body.model_dump(mode="json"),
            response,
            "api:jobs",
            body.instructions,
        )

    @api.get("/jobs", response_model=list[JobInfo], tags=["jobs"], dependencies=admin)
    def list_jobs() -> list[JobInfo]:
        return [job.info() for job in state.jobs.list()]

    def _job(job_id: str) -> Job:
        job = state.jobs.get(job_id)
        if job is None:
            raise ApiError(404, "not_found", f"No job with id '{job_id}'.")
        return job

    @api.get("/jobs/{job_id}", response_model=JobInfo, tags=["jobs"])
    def get_job(job_id: str) -> JobInfo:
        return _job(job_id).info()

    @api.get("/jobs/{job_id}/rows", response_model=RowsPage, tags=["jobs"])
    def get_job_rows(job_id: str, offset: int = 0, limit: int = 50) -> RowsPage:
        job = _job(job_id)
        offset, limit = _page(offset, limit)
        outcomes = job.report.outcomes if job.report else []
        rows = [_row_result(o).model_dump(mode="json") for o in outcomes[offset : offset + limit]]
        return RowsPage(offset=offset, limit=limit, total=len(outcomes), rows=rows)

    @api.delete("/jobs/{job_id}", response_model=JobInfo, tags=["jobs"])
    def cancel_job(job_id: str) -> JobInfo:
        _job(job_id)
        job = state.jobs.cancel(job_id)
        assert job is not None
        return job.info()

    # -- datasets and benchmarks -----------------------------------------------------------

    @api.post("/datasets/validate", response_model=DatasetReport, tags=["datasets"])
    def validate(body: ValidateRequest) -> DatasetReport:
        if (body.rows is None) == (body.content is None):
            raise ApiError(422, "invalid_request", "Provide exactly one of 'rows' or 'content'.")
        if body.rows is not None:
            return load_dataset(body.rows, kind=body.kind, settings=settings)[1]
        if not body.format:
            raise ApiError(422, "invalid_request", "'format' is required with 'content'.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload.{body.format}"
            path.write_text(body.content or "", encoding="utf-8")
            report = load_dataset(path, kind=body.kind, settings=settings)[1]
        report.source = f"<upload.{body.format}>"
        return report

    @api.get("/benchmarks", response_model=list[SampleOut], tags=["datasets"])
    def benchmarks() -> list[SampleOut]:
        return [
            SampleOut(
                name=s.name, kind=s.kind, description=s.description, criteria=list(s.criteria)
            )
            for s in SAMPLE_DATASETS.values()
        ]

    @api.post("/benchmarks/{name}/jobs", response_model=JobInfo, status_code=202, tags=["datasets"])
    def run_benchmark(
        name: str,
        body: BenchmarkJobRequest,
        response: Response,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> JobInfo:
        if name not in SAMPLE_DATASETS:
            raise ApiError(404, "not_found", f"No benchmark named '{name}'.")
        sample = SAMPLE_DATASETS[name]
        return _submit_job(
            load_sample(name, settings=settings),
            body.criteria or list(sample.criteria),
            body.mitigate_position_bias,
            body.experiment,
            body.run_name,
            idempotency_key,
            {"benchmark": name, **body.model_dump(mode="json")},
            response,
            f"sample:{name}",
            body.instructions,
        )

    # -- experiments -------------------------------------------------------------------

    @api.get("/experiments", response_model=list[Experiment], tags=["experiments"])
    def experiments() -> list[Experiment]:
        return state.store().list_experiments()

    @api.get(
        "/experiments/{experiment_name}", response_model=ExperimentDetail, tags=["experiments"]
    )
    def get_experiment(experiment_name: str, offset: int = 0, limit: int = 50) -> ExperimentDetail:
        """One experiment and a page of its runs (newest first)."""
        store_ = state.store()
        found = store_.get_experiment(experiment_name)
        if found is None:
            raise ApiError(404, "not_found", f"No experiment named '{experiment_name}'.")
        offset, limit = _page(offset, limit)
        return ExperimentDetail(
            experiment=found,
            total_runs=store_.count_runs(experiment=experiment_name),
            offset=offset,
            limit=limit,
            runs=store_.list_runs(experiment=experiment_name, limit=limit, offset=offset),
        )

    @api.get("/runs", response_model=RowsPage, tags=["experiments"])
    def runs(experiment: str | None = None, offset: int = 0, limit: int = 50) -> RowsPage:
        offset, limit = _page(offset, limit)
        store_ = state.store()
        items = store_.list_runs(experiment=experiment, limit=limit, offset=offset)
        return RowsPage(
            offset=offset,
            limit=limit,
            total=store_.count_runs(experiment=experiment),
            rows=[r.model_dump(mode="json") for r in items],
        )

    @api.get("/runs/compare", response_model=RunComparison, tags=["experiments"])
    def compare(
        baseline: str,
        candidate: str,
        min_effect: Annotated[float, Query(ge=0, le=1)] = 0.02,
    ) -> RunComparison:
        return compare_runs(state.store(), baseline, candidate, min_effect=min_effect)

    @api.get("/runs/agreement", response_model=JudgeAgreement, tags=["analysis"])
    def agreement(run_a: str, run_b: str) -> JudgeAgreement:
        """Agreement between two runs over the same rows (e.g. two different judges)."""
        return judge_agreement(state.store(), run_a, run_b)

    def _run(run_id: str) -> RunInfo:
        run = state.store().get_run(run_id)
        if run is None:
            raise ApiError(404, "not_found", f"No run with id '{run_id}'.")
        return run

    @api.get("/runs/{run_id}", response_model=RunInfo, tags=["experiments"])
    def get_run(run_id: str) -> RunInfo:
        return _run(run_id)

    @api.get(
        "/runs/{run_id}/report",
        response_class=PlainTextResponse,
        tags=["experiments"],
        responses={200: {"content": {"text/markdown": {}}}},
    )
    def get_run_report(
        run_id: str,
        baseline: str | None = None,
        min_effect: Annotated[float, Query(ge=0, le=1)] = 0.02,
    ) -> PlainTextResponse:
        """A shareable Markdown report of a run, optionally compared with a baseline run."""
        run = _run(run_id)
        text = run_report(state.store(), run.id, baseline_id=baseline, min_effect=min_effect)
        return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")

    @api.get("/runs/{run_id}/rows", response_model=RowsPage, tags=["experiments"])
    def get_run_rows(run_id: str, offset: int = 0, limit: int = 50) -> RowsPage:
        run = _run(run_id)
        offset, limit = _page(offset, limit)
        rows = state.store().get_rows(run.id)
        return RowsPage(
            offset=offset,
            limit=limit,
            total=len(rows),
            rows=[r.model_dump(mode="json") for r in rows[offset : offset + limit]],
        )

    @api.delete("/runs/{run_id}", status_code=204, tags=["admin"], dependencies=admin)
    def delete_run(run_id: str) -> Response:
        run = _run(run_id)
        state.store().delete_run(run.id)
        return Response(status_code=204)

    @api.get("/statistics", response_model=StatisticsOut, tags=["experiments"])
    def statistics() -> StatisticsOut:
        return StatisticsOut(**state.store().statistics())

    app.include_router(api)
    return app


__all__ = ["create_app"]
