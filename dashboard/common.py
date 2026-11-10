"""Shared state, safeguards and components for the Streamlit dashboard.

Two operating modes:

* **Local** (default): full features, runs persist to ``JUDGE_DB_PATH``.
* **Public demo** (``JUDGE_DEMO_MODE=true``, e.g. on Streamlit Community
  Cloud): live judging is capped per visitor session and per day, inputs are
  size-limited, and each visitor's runs live only in their own session, so
  nothing is shared between visitors or persisted.

The judge is either the configured live provider (when it has credentials) or
the offline mock. Every result says which one produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import streamlit as st

from llm_judge import (
    CRITERIA_PRESETS,
    REFERENCE_CRITERIA_PRESETS,
    Criterion,
    EvaluationWarning,
    Judge,
    JudgeResult,
    PointwiseResult,
    RankingResult,
    Winner,
    resolve_criterion,
)
from llm_judge.config import Settings
from llm_judge.datasets import DatasetSummary
from llm_judge.exceptions import ConfigurationError, LLMJudgeError
from llm_judge.experiments import ExperimentStore
from llm_judge.providers import (
    CallBudget,
    LLMProvider,
    MockProvider,
    QuotaExceededError,
    QuotaProvider,
    build_provider,
    configuration_status,
)

from . import charts

T = TypeVar("T")

MOCK_NOTICE = "Mock judge: a deterministic text heuristic, not an LLM. Use it to explore the UI."


# --- configuration -------------------------------------------------------------------


def _secrets_to_env() -> None:
    """Expose Streamlit secrets (Streamlit Community Cloud) as environment settings."""
    try:
        secrets = {k: st.secrets[k] for k in st.secrets}
    except Exception:  # no secrets.toml: local runs use .env / the environment instead
        return
    for key, value in secrets.items():
        if key.startswith(("JUDGE_", "GROQ_")) and isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


def app_settings() -> Settings:
    _secrets_to_env()
    return Settings()


def chart_theme() -> charts.Theme:
    try:
        mode = st.context.theme.type
    except Exception:  # pragma: no cover - older Streamlit or bare mode
        mode = None
    return charts.theme_for(mode)


@st.cache_resource(show_spinner=False)
def _shared_provider(name: str, model_key: str, temperature: float) -> LLMProvider:
    """One live client per (provider, model, temperature) per process."""
    return build_provider(app_settings().model_copy(update={"temperature": temperature}), name)


@st.cache_resource(show_spinner=False)
def _daily_budget(limit: int) -> CallBudget:
    return CallBudget(limit, name="shared daily demo budget", window_seconds=24 * 3600)


def live_status(settings: Settings) -> tuple[bool, str]:
    """Whether a live (non-mock) judge is available, with an explanation."""
    if settings.provider == "mock":
        return False, "JUDGE_PROVIDER is set to mock."
    return configuration_status(settings.provider, settings)


def _session_budget(settings: Settings) -> CallBudget:
    if "session_budget" not in st.session_state:
        st.session_state.session_budget = CallBudget(
            settings.demo_session_calls, name="per-session demo budget"
        )
    budget: CallBudget = st.session_state.session_budget
    return budget


def judge_mode() -> str:
    return str(st.session_state.get("judge_mode", "mock"))


def current_judge(*, mitigate: bool | None = None, temperature: float | None = None) -> Judge:
    """The judge for this session, with demo safeguards applied when enabled."""
    settings = app_settings()
    if temperature is not None:
        settings = settings.model_copy(update={"temperature": temperature})
    if mitigate is None:
        mitigate = bool(st.session_state.get("mitigate_bias", False))
    provider: LLMProvider
    if judge_mode() == "live":
        key = hashlib.sha256(
            f"{settings.provider}|{settings.groq_model}|{settings.groq_api_key}".encode()
        ).hexdigest()
        provider = _shared_provider(settings.provider, key, settings.temperature)
        if settings.demo_mode:
            provider = QuotaProvider(
                provider, [_session_budget(settings), _daily_budget(settings.demo_daily_calls)]
            )
    else:
        provider = MockProvider()
    updates: dict[str, Any] = {"mitigate_position_bias": mitigate}
    if settings.demo_mode:
        updates.update(
            max_input_chars=settings.demo_max_input_chars,
            max_candidates=settings.demo_max_candidates,
            max_criteria=min(settings.max_criteria, 8),
        )
    return Judge(provider=provider, settings=settings.model_copy(update=updates))


def experiment_store() -> ExperimentStore:
    """Session-only store in demo mode; the configured SQLite file otherwise."""
    settings = app_settings()
    if settings.demo_mode:
        if "demo_store" not in st.session_state:
            st.session_state.demo_store = ExperimentStore(":memory:")
        store: ExperimentStore = st.session_state.demo_store
        return store
    return _local_store(settings.db_path)


@st.cache_resource(show_spinner=False)
def _local_store(path: str) -> ExperimentStore:
    return ExperimentStore(path)


def row_limit(default: int = 200) -> int:
    settings = app_settings()
    return settings.demo_max_rows if settings.demo_mode else default


# --- layout --------------------------------------------------------------------------


def render_sidebar() -> None:
    settings = app_settings()
    available, detail = live_status(settings)
    with st.sidebar:
        st.markdown("### Judge")
        options = ["live", "mock"] if available else ["mock"]
        if st.session_state.get("judge_mode") not in options:
            st.session_state.judge_mode = options[0]
        labels = {
            "live": f"Live LLM ({settings.provider})",
            "mock": "Mock (offline, not an LLM)",
        }
        st.radio(
            "Judge",
            options,
            key="judge_mode",
            format_func=labels.__getitem__,
            label_visibility="collapsed",
        )
        if not available:
            st.caption(f"Live judging unavailable: {detail}")
        st.toggle(
            "Mitigate position bias",
            key="mitigate_bias",
            help="Judge pairs in both orders and average. More reliable; twice the calls.",
        )


def render_sidebar_status() -> None:
    """Quota and demo notices, rendered after the page so the counts are current."""
    settings = app_settings()
    with st.sidebar:
        if settings.demo_mode and judge_mode() == "live":
            session = _session_budget(settings)
            daily = _daily_budget(settings.demo_daily_calls)
            st.caption(
                f"Demo quota: {session.remaining}/{session.limit} live judge calls left "
                f"this session; {daily.remaining} left today across all visitors."
            )
        if settings.demo_mode:
            st.caption("Public demo: inputs are capped and nothing you enter is stored.")


def show_stop_reason(stop_reason: str | None) -> None:
    """Explain a dataset run that stopped early (e.g. an exhausted demo budget)."""
    if stop_reason:
        st.warning(
            f"{stop_reason} Pick 'Mock' in the sidebar to keep exploring.",
            icon=":material/hourglass_empty:",
        )


def render_judge_badge(metadata: dict[str, Any]) -> None:
    judge = metadata.get("judge")
    if not isinstance(judge, dict):
        return
    provider, model = judge.get("provider"), judge.get("model")
    if provider == "mock":
        st.badge("Mock judge (not an LLM)", icon=":material/science:", color="orange")
        st.caption(MOCK_NOTICE)
    else:
        st.badge(f"{provider} · {model}", icon=":material/gavel:", color="blue")


def run_safely(action: Callable[[], T]) -> T | None:
    """Run an evaluation, turning library errors into friendly messages."""
    try:
        result = action()
    except QuotaExceededError as exc:
        st.warning(f"{exc} Pick 'Mock' in the sidebar to keep exploring.")
        return None
    except ConfigurationError as exc:
        st.error(f"Configuration problem: {exc}")
        return None
    except LLMJudgeError as exc:
        st.error(f"Evaluation failed: {exc}")
        return None
    st.session_state.evaluations = int(st.session_state.get("evaluations", 0)) + 1
    return result


# --- inputs --------------------------------------------------------------------------


def criteria_picker(
    key: str, *, default: Sequence[str], reference: bool = False
) -> list[Criterion]:
    """Preset multiselect + custom criteria + per-criterion weights."""
    presets = list(REFERENCE_CRITERIA_PRESETS if reference else {}) + list(CRITERIA_PRESETS)
    chosen = st.multiselect("Criteria", presets, default=list(default), key=f"{key}:criteria")
    custom_raw = st.text_input(
        "Custom criteria (comma-separated)", key=f"{key}:custom", placeholder="e.g. Politeness"
    )
    names = list(dict.fromkeys(chosen + [c.strip() for c in custom_raw.split(",") if c.strip()]))
    criteria = []
    with st.expander("Weights"):
        st.caption("Relative importance in the overall score.")
        for name in names:
            weight = st.slider(name, 0.5, 5.0, 1.0, 0.5, key=f"{key}:w:{name}")
            base = resolve_criterion(name)
            criteria.append(Criterion(**{**base.model_dump(), "weight": weight}))
    return criteria


# --- results -------------------------------------------------------------------------


def render_warnings(warnings: Sequence[EvaluationWarning]) -> None:
    for warning in warnings:
        st.warning(
            f"**{warning.code.replace('_', ' ')}**: {warning.message}", icon=":material/info:"
        )


def _download(label: str, payload: Any, file_name: str, key: str) -> None:
    st.download_button(
        label,
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name=file_name,
        mime="application/json",
        key=key,
        on_click="ignore",  # downloading must not rerun the page and clear the results
        icon=":material/download:",
    )


def render_pairwise(result: JudgeResult, criteria: Sequence[Criterion], key: str) -> None:
    render_judge_badge(result.metadata)
    verdict = {
        Winner.A: "Output A is preferred overall.",
        Winner.B: "Output B is preferred overall.",
        Winner.TIE: "The judge calls it a tie overall.",
    }[result.overall_winner]
    st.subheader(verdict)
    if result.overall_comment:
        st.write(result.overall_comment)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score A", f"{result.overall_score_a:.2f}")
    c2.metric("Overall score B", f"{result.overall_score_b:.2f}")
    c3.metric("Latency", f"{result.metadata.get('latency_seconds', 0)} s")
    render_warnings(result.warnings)

    scales = {(c.scale_min, c.scale_max) for c in criteria}
    by_name = {c.name: c for c in criteria}
    names = [s.criterion for s in result.criteria]
    if len(scales) == 1:
        a = [s.output_a_score for s in result.criteria]
        b = [s.output_b_score for s in result.criteria]
        scale: tuple[float, float] | None = next(iter(scales))
    else:
        a = [by_name[s.criterion].normalise(s.output_a_score) for s in result.criteria]
        b = [by_name[s.criterion].normalise(s.output_b_score) for s in result.criteria]
        scale = None
    st.plotly_chart(
        charts.criterion_comparison(names, a, b, scale=scale, theme=chart_theme()),
        key=f"{key}:chart",
        config=charts.CHART_CONFIG,
    )
    for score in result.criteria:
        with st.container(border=True):
            st.markdown(f"**{score.criterion}**: winner `{score.winner.value}`")
            left, right = st.columns(2)
            left.markdown(f"**A: {score.output_a_score:g}** · {score.output_a_explanation or '-'}")
            right.markdown(f"**B: {score.output_b_score:g}** · {score.output_b_explanation or '-'}")
    _download("Download JSON", result.model_dump(mode="json"), "judgment.json", f"{key}:dl")


def render_pointwise(result: PointwiseResult, key: str) -> None:
    render_judge_badge(result.metadata)
    title = "Reference grading" if result.mode == "reference" else "Pointwise grading"
    st.subheader(f"{title}: {result.overall_score:.2f} overall")
    columns = st.columns(max(1, len(result.criteria)))
    for column, score in zip(columns, result.criteria, strict=True):
        column.metric(score.criterion, f"{score.score:g}")
    if result.feedback:
        st.info(result.feedback, icon=":material/lightbulb:")
    if result.mode == "reference":
        left, right = st.columns(2)
        with left:
            st.markdown("**Missing information**")
            for item in result.missing_information or ["None found."]:
                st.markdown(f"- {item}")
        with right:
            st.markdown("**Unsupported claims (possible hallucinations)**")
            for item in result.unsupported_claims or ["None found."]:
                st.markdown(f"- {item}")
        if result.reference_metrics:
            m = result.reference_metrics
            st.caption(
                f"Deterministic overlap with the reference (no LLM): token F1 {m.token_f1:.3f}, "
                f"exact match {'yes' if m.exact_match else 'no'}."
            )
    render_warnings(result.warnings)
    for score in result.criteria:
        st.markdown(f"**{score.criterion} ({score.score:g})**: {score.explanation or '-'}")
    _download("Download JSON", result.model_dump(mode="json"), "grade.json", f"{key}:dl")


def render_ranking(result: RankingResult, key: str) -> None:
    render_judge_badge(result.metadata)
    winner = result.leaderboard[0].candidate if result.leaderboard else "-"
    st.subheader(f"Leaderboard: '{winner}' ranks first")
    measure = "Win rate" if result.strategy == "round_robin" else "Overall score"
    st.plotly_chart(
        charts.ranked_bars(
            [r.candidate for r in result.leaderboard],
            [r.score for r in result.leaderboard],
            theme=chart_theme(),
            x_title=f"{measure} (0-1)",
        ),
        key=f"{key}:chart",
        config=charts.CHART_CONFIG,
    )
    st.dataframe(
        [
            {
                "rank": r.rank,
                "candidate": r.candidate,
                measure.lower(): r.score,
                "W-L-T": f"{r.wins}-{r.losses}-{r.ties}"
                if result.strategy == "round_robin"
                else "",
                "mean overall score": r.mean_overall_score,
            }
            for r in result.leaderboard
        ],
        hide_index=True,
    )
    render_warnings(result.warnings)
    _download("Download JSON", result.model_dump(mode="json"), "ranking.json", f"{key}:dl")


def render_summary(summary: DatasetSummary, key: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows evaluated", f"{summary.succeeded}/{summary.total}")
    c2.metric("Failed", summary.failed)
    if summary.kind == "pairwise":
        c3.metric("Wins A / B / ties", f"{summary.wins_a} / {summary.wins_b} / {summary.ties}")
    elif summary.mean_overall_score is not None:
        c3.metric("Mean overall score", f"{summary.mean_overall_score:.3f}")
    if summary.mean_token_f1 is not None:
        st.caption(
            f"Mean token F1 vs reference: {summary.mean_token_f1:.3f} · rows with unsupported "
            f"claims: {summary.rows_with_unsupported_claims}"
        )
    if summary.models:
        pairwise = summary.kind == "pairwise"
        st.markdown("**By model**")
        st.plotly_chart(
            charts.ranked_bars(
                [m.model for m in summary.models],
                [(m.win_rate if pairwise else m.mean_overall_score) or 0.0 for m in summary.models],
                theme=chart_theme(),
                x_title="Win rate (ties count half)" if pairwise else "Mean overall score (0-1)",
                hover_suffix=[f" over {m.rows} row(s)" for m in summary.models],
            ),
            key=f"{key}:models",
            config=charts.CHART_CONFIG,
        )
        st.dataframe([m.model_dump(exclude_none=True) for m in summary.models], hide_index=True)
    if summary.warning_counts:
        counts = ", ".join(f"{k.replace('_', ' ')} x{v}" for k, v in summary.warning_counts.items())
        st.caption(f"Reliability warnings: {counts}")
