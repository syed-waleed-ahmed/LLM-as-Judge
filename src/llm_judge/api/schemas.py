"""Request and response schemas for the REST API (all strictly validated)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..agents import AgentTrace
from ..datasets import DatasetKind, DatasetSummary
from ..models import Criterion, JudgeResult, PointwiseResult


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionInput(_Strict):
    """A criterion given inline. Preset names fill in the description automatically."""

    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    scale_min: int = Field(default=1, ge=0)
    scale_max: int = Field(default=10, le=100)
    weight: float = Field(default=1.0, gt=0, le=100)


CriteriaField = list[str | CriterionInput]


def _default_criteria() -> CriteriaField:
    return ["Helpfulness"]


class PairwiseRequest(_Strict):
    task_description: str
    output_a: str
    output_b: str
    criteria: CriteriaField = Field(default_factory=_default_criteria, min_length=1)
    candidate_a_model: str | None = None
    candidate_b_model: str | None = None
    mitigate_position_bias: bool | None = Field(
        default=None, description="Override the server default for this request."
    )


class PointwiseRequest(_Strict):
    task_description: str
    output: str
    criteria: CriteriaField = Field(default_factory=_default_criteria, min_length=1)
    candidate_model: str | None = None


class ReferenceRequest(_Strict):
    task_description: str
    output: str
    reference: str
    criteria: CriteriaField | None = Field(
        default=None, description="Defaults to Correctness and Completeness."
    )
    candidate_model: str | None = None


class RankRequest(_Strict):
    task_description: str
    candidates: dict[str, str] = Field(..., min_length=2)
    criteria: CriteriaField = Field(default_factory=_default_criteria, min_length=1)
    strategy: Literal["round_robin", "pointwise"] = "round_robin"
    candidate_models: dict[str, str] | None = None
    mitigate_position_bias: bool | None = None


class RagRequest(_Strict):
    question: str
    contexts: list[str] = Field(..., min_length=1, max_length=20, description="In rank order.")
    answer: str
    reference: str | None = Field(default=None, description="Enables context recall.")


class AgentRequest(_Strict):
    trace: AgentTrace
    criteria: CriteriaField | None = Field(default=None, description="Defaults to the agent set.")
    candidate_model: str | None = None


class ConsistencyRequest(_Strict):
    """Give output_a and output_b (pairwise) or output (pointwise)."""

    task_description: str
    output_a: str | None = None
    output_b: str | None = None
    output: str | None = None
    criteria: CriteriaField = Field(default_factory=_default_criteria, min_length=1)
    samples: int = Field(default=5, ge=2, le=10, description="Capped to protect provider quotas.")


class BatchRequest(_Strict):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    kind: DatasetKind | None = Field(default=None, description="Inferred from the columns.")
    criteria: CriteriaField | None = None
    mitigate_position_bias: bool | None = None


class JobRequest(BatchRequest):
    experiment: str | None = Field(default=None, description="Record the run in this experiment.")
    run_name: str | None = None


class BenchmarkJobRequest(_Strict):
    criteria: CriteriaField | None = Field(default=None, description="Defaults to the sample's.")
    mitigate_position_bias: bool | None = None
    experiment: str | None = None
    run_name: str | None = None


class ValidateRequest(_Strict):
    """Provide either ``rows`` (JSON objects) or ``content`` plus its ``format``."""

    rows: list[dict[str, Any]] | None = None
    content: str | None = None
    format: Literal["csv", "jsonl", "json", "yaml"] | None = None
    kind: DatasetKind | None = None


class RowResult(BaseModel):
    id: str
    ok: bool
    error: str | None = None
    result: JudgeResult | PointwiseResult | None = None


class BatchResponse(BaseModel):
    kind: DatasetKind
    summary: DatasetSummary
    rows: list[RowResult]


JobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    kind: DatasetKind
    total: int
    done: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    experiment: str | None = None
    run_id: str | None = None
    summary: DatasetSummary | None = None


class RowsPage(BaseModel):
    offset: int
    limit: int
    total: int
    rows: list[Any]


class ProviderOut(BaseModel):
    name: str
    description: str
    requires_api_key: bool
    local: bool
    cost: str
    configured: bool
    detail: str
    active: bool


class CriteriaOut(BaseModel):
    presets: list[Criterion]
    reference_presets: list[Criterion]


class SampleOut(BaseModel):
    name: str
    kind: DatasetKind
    description: str
    criteria: list[str]


class ProviderHealthOut(BaseModel):
    ok: bool
    detail: str


class StatusOut(BaseModel):
    version: str
    provider: str
    model: str | None
    provider_health: ProviderHealthOut
    auth_required: bool
    limits: dict[str, int]


class StatisticsOut(BaseModel):
    experiments: int
    runs: int
    rows_evaluated: int
    runs_by_kind: dict[str, int]
    runs_by_status: dict[str, int]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody
