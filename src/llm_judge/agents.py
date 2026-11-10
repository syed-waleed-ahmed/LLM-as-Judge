"""Agent trajectory evaluation: deterministic trace checks plus a judged grade.

An agent run is described framework-agnostically as an :class:`AgentTrace`
(the task, the available tools, the ordered steps, and the final answer).
Evaluation has two independent parts:

* **Deterministic checks** (no LLM): unknown tools, missing required
  arguments, malformed calls, failed tool results, repeated identical calls,
  step-budget overruns, expected tools never used, and a missing final answer.
  They come with counts in :class:`AgentMetrics`.
* **A judged grade**: the trajectory is rendered as text and graded pointwise
  on agent criteria (task completion, tool selection, step efficiency, final
  answer quality), optionally against a reference answer.

The two are reported side by side rather than blended into one opaque number.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .models import RESULT_SCHEMA_VERSION, Criterion, PointwiseResult

if TYPE_CHECKING:
    from .judge import Judge

DEFAULT_AGENT_CRITERIA = ("Task Completion", "Tool Selection", "Step Efficiency")
_MAX_STEPS = 200
_MAX_STEP_CHARS = 2_000


class ToolSpec(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    required_args: list[str] = Field(default_factory=list)


class AgentStep(BaseModel):
    type: Literal["thought", "tool_call", "tool_result", "message"]
    content: str = ""
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: bool = False


class AgentTrace(BaseModel):
    task: str = Field(..., min_length=1)
    steps: list[AgentStep] = Field(default_factory=list, max_length=_MAX_STEPS)
    final_answer: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    max_steps: int | None = Field(default=None, ge=1)


class TraceCheck(BaseModel):
    code: str
    severity: Literal["error", "warning"]
    message: str
    step: int | None = None


class AgentMetrics(BaseModel):
    steps: int
    tool_calls: int
    invalid_tool_calls: int
    failed_tool_calls: int
    repeated_calls: int
    expected_tool_recall: float | None = None
    within_step_budget: bool | None = None


class AgentResult(BaseModel):
    schema_version: str = RESULT_SCHEMA_VERSION
    metrics: AgentMetrics
    checks: list[TraceCheck]
    grade: PointwiseResult
    #: The judged overall score (0-1). Deterministic findings are in ``checks``.
    overall_score: float | None


def check_trace(trace: AgentTrace) -> tuple[AgentMetrics, list[TraceCheck]]:
    """Run the deterministic trajectory checks."""
    checks: list[TraceCheck] = []
    tools = {t.name: t for t in trace.tools}
    seen_calls: set[str] = set()
    used: set[str] = set()
    calls = invalid = failed = repeated = 0

    for index, step in enumerate(trace.steps, start=1):
        if step.type == "tool_call":
            calls += 1
            if not step.tool:
                invalid += 1
                checks.append(
                    TraceCheck(
                        code="malformed_tool_call",
                        severity="error",
                        step=index,
                        message="Tool call without a tool name.",
                    )
                )
                continue
            used.add(step.tool)
            if tools and step.tool not in tools:
                invalid += 1
                checks.append(
                    TraceCheck(
                        code="unknown_tool",
                        severity="error",
                        step=index,
                        message=f"Called '{step.tool}', which is not an available tool.",
                    )
                )
            elif step.tool in tools:
                missing = [a for a in tools[step.tool].required_args if a not in step.arguments]
                if missing:
                    invalid += 1
                    checks.append(
                        TraceCheck(
                            code="missing_arguments",
                            severity="error",
                            step=index,
                            message=f"'{step.tool}' is missing required argument(s) {missing}.",
                        )
                    )
            signature = json.dumps([step.tool, step.arguments], sort_keys=True, default=str)
            if signature in seen_calls:
                repeated += 1
                checks.append(
                    TraceCheck(
                        code="repeated_call",
                        severity="warning",
                        step=index,
                        message=f"Repeated an identical call to '{step.tool}'.",
                    )
                )
            seen_calls.add(signature)
        elif step.type == "tool_result" and step.error:
            failed += 1
            checks.append(
                TraceCheck(
                    code="tool_error",
                    severity="warning",
                    step=index,
                    message=f"Tool {step.tool or ''} returned an error.".replace("  ", " "),
                )
            )

    within_budget = None
    if trace.max_steps is not None:
        within_budget = len(trace.steps) <= trace.max_steps
        if not within_budget:
            checks.append(
                TraceCheck(
                    code="over_step_budget",
                    severity="warning",
                    message=f"{len(trace.steps)} steps exceed the budget of {trace.max_steps}.",
                )
            )
    recall = None
    if trace.expected_tools:
        expected = set(trace.expected_tools)
        recall = len(expected & used) / len(expected)
        for name in sorted(expected - used):
            checks.append(
                TraceCheck(
                    code="expected_tool_unused",
                    severity="warning",
                    message=f"Expected tool '{name}' was never called.",
                )
            )
    if not trace.final_answer.strip():
        checks.append(
            TraceCheck(
                code="no_final_answer",
                severity="error",
                message="The agent produced no final answer.",
            )
        )
    metrics = AgentMetrics(
        steps=len(trace.steps),
        tool_calls=calls,
        invalid_tool_calls=invalid,
        failed_tool_calls=failed,
        repeated_calls=repeated,
        expected_tool_recall=None if recall is None else round(recall, 4),
        within_step_budget=within_budget,
    )
    return metrics, checks


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_STEP_CHARS else text[:_MAX_STEP_CHARS] + " [...]"


def render_trace(trace: AgentTrace) -> str:
    """A compact, readable text form of the trajectory for the judge."""
    lines = []
    if trace.tools:
        lines.append("Available tools:")
        for tool in trace.tools:
            req = f" (required: {', '.join(tool.required_args)})" if tool.required_args else ""
            lines.append(f"- {tool.name}: {tool.description}{req}")
        lines.append("")
    lines.append("Trajectory:")
    if not trace.steps:
        lines.append("(no steps)")
    for i, step in enumerate(trace.steps, start=1):
        if step.type == "tool_call":
            args = json.dumps(step.arguments, ensure_ascii=False, default=str)
            lines.append(f"[{i}] tool_call {step.tool or '?'} {_clip(args)}")
        elif step.type == "tool_result":
            status = " (error)" if step.error else ""
            lines.append(f"[{i}] tool_result{status} {step.tool or ''}: {_clip(step.content)}")
        else:
            lines.append(f"[{i}] {step.type}: {_clip(step.content)}")
    lines.append("")
    lines.append(f"Final answer: {trace.final_answer.strip() or '(none)'}")
    return "\n".join(lines)


def run_agent_evaluation(
    judge: Judge,
    trace: AgentTrace | dict[str, Any],
    criteria: Sequence[Criterion | str] | None = None,
    *,
    candidate_model: str | None = None,
) -> AgentResult:
    """Evaluate an agent trajectory; see :meth:`Judge.evaluate_agent`."""
    if not isinstance(trace, AgentTrace):
        trace = AgentTrace.model_validate(trace)
    metrics, checks = check_trace(trace)
    task = f"An AI agent was given this task:\n{trace.task}"
    if trace.max_steps:
        task += f"\n(Step budget: {trace.max_steps}.)"
    grade = judge.evaluate_pointwise(
        task,
        render_trace(trace),
        list(criteria) if criteria else list(DEFAULT_AGENT_CRITERIA),
        reference=trace.reference_answer,
        candidate_model=candidate_model,
    )
    return AgentResult(
        metrics=metrics, checks=checks, grade=grade, overall_score=grade.overall_score
    )
