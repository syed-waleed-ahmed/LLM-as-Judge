"""Analytics: trends, score distributions, criteria, bias signals and reliability of recorded runs.

Every number on this page is computed from stored results. Nothing is
estimated by an LLM.
"""

from __future__ import annotations

from statistics import fmean, median, pstdev
from typing import Any

import streamlit as st

from dashboard import charts
from dashboard.common import app_settings, chart_theme, experiment_store
from llm_judge.experiments import RunInfo, StoredRow

settings = app_settings()
store = experiment_store()
theme = chart_theme()
st.title("Analytics")
st.caption(
    "Computed from the runs recorded in "
    + ("this browser session" if settings.demo_mode else "the experiment database")
    + ". Every number comes from stored results; none is estimated by an LLM."
)

runs = store.list_runs(limit=500)
if not runs:
    st.info(
        "No recorded runs yet. Run a benchmark, or evaluate a dataset with 'Record as "
        "experiment', and the analysis appears here.",
        icon=":material/insights:",
    )
    st.stop()

# --- scope -----------------------------------------------------------------------------
experiments = sorted({r.experiment for r in runs if r.experiment})
left, right = st.columns(2)
scope = left.selectbox("Experiment", ["All runs", *experiments], key="an:scope")
in_scope = [r for r in runs if scope == "All runs" or r.experiment == scope]
kinds = sorted({r.kind for r in in_scope})
kind = (
    right.radio("Run kind", kinds, horizontal=True, key="an:kind") if len(kinds) > 1 else kinds[0]
)
# list_runs is newest first; analytics read oldest to newest.
selected: list[RunInfo] = [r for r in reversed(in_scope) if r.kind == kind]
labels = {r.id: f"{i + 1}. {r.name}" for i, r in enumerate(selected)}
pairwise = kind == "pairwise"


def _scales(run: RunInfo) -> dict[str, tuple[float, float]]:
    scales = {}
    for c in run.config.get("criteria") or []:
        if "scale_min" in c and "scale_max" in c:
            scales[c["name"]] = (float(c["scale_min"]), float(c["scale_max"]))
    return scales


def _normalise(value: float, scale: tuple[float, float]) -> float:
    low, high = scale
    return (value - low) / (high - low) if high > low else 0.0


@st.cache_data(show_spinner=False, max_entries=64)
def _rows(run_id: str, _store_key: int) -> list[StoredRow]:
    return store.get_rows(run_id)


def rows_of(run: RunInfo) -> list[StoredRow]:
    return _rows(run.id, id(store))


trends_tab, dist_tab, criteria_tab, bias_tab, ops_tab = st.tabs(
    ["Trends", "Score distribution", "Criteria", "Bias signals", "Reliability"]
)

# --- trends ----------------------------------------------------------------------------
with trends_tab:
    names = [labels[r.id] for r in selected]
    if pairwise:
        series = [
            ("Output A", [r.summary.get("mean_overall_score_a") for r in selected]),
            ("Output B", [r.summary.get("mean_overall_score_b") for r in selected]),
        ]
        y_title = "Mean overall score (0-1)"
    else:
        series = [("Mean score", [r.summary.get("mean_overall_score") for r in selected])]
        y_title = "Mean overall score (0-1)"
    if len(selected) < 2:
        st.caption("Record at least two runs of this kind to see a trend.")
    st.plotly_chart(
        charts.trend_lines(names, series, theme=theme, y_title=y_title),
        key="an:trend",
        config=charts.CHART_CONFIG,
    )
    table: list[dict[str, Any]] = []
    for r in selected:
        row: dict[str, Any] = {
            "run": labels[r.id],
            "experiment": r.experiment or "-",
            "judge": f"{r.judge_provider}/{r.judge_model}",
            "rows ok": f"{r.succeeded}/{r.total}",
        }
        for name, values in series:
            row[name] = values[selected.index(r)]
        if pairwise:
            decided = r.summary.get("succeeded") or 0
            row["A win rate"] = (r.summary.get("wins_a") or 0) / decided if decided else None
        table.append(row)
    st.dataframe(table, hide_index=True)
    if len({(r.judge_provider, r.judge_model, r.prompt_version) for r in selected}) > 1:
        st.warning(
            "These runs used different judges or prompt versions, so part of any trend may "
            "come from the judging setup rather than the outputs.",
            icon=":material/info:",
        )

# --- distribution ----------------------------------------------------------------------
with dist_tab:
    run_id = st.selectbox(
        "Run", [r.id for r in reversed(selected)], format_func=labels.__getitem__, key="an:dist"
    )
    run = next(r for r in selected if r.id == run_id)
    ok_rows = [row for row in rows_of(run) if row.ok]
    if pairwise:
        scores = [
            row.overall_score_a - row.overall_score_b
            for row in ok_rows
            if row.overall_score_a is not None and row.overall_score_b is not None
        ]
        figure = charts.margin_histogram(
            scores,
            theme=theme,
            x_title="Score margin, A minus B (negative: B scored higher)",
            tie_margin=float(run.config.get("settings", {}).get("tie_margin", 0.05)),
        )
    else:
        scores = [row.overall_score for row in ok_rows if row.overall_score is not None]
        figure = charts.score_histogram(scores, theme=theme, x_title="Overall score (0-1)")
    if scores:
        st.plotly_chart(figure, key="an:hist", config=charts.CHART_CONFIG)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(scores))
        c2.metric("Mean", f"{fmean(scores):+.3f}" if pairwise else f"{fmean(scores):.3f}")
        c3.metric("Median", f"{median(scores):+.3f}" if pairwise else f"{median(scores):.3f}")
        c4.metric("Spread (SD)", f"{pstdev(scores):.3f}")
    else:
        st.caption("This run has no successfully scored rows.")

# --- criteria --------------------------------------------------------------------------
with criteria_tab:
    run_id = st.selectbox(
        "Run", [r.id for r in reversed(selected)], format_func=labels.__getitem__, key="an:crit"
    )
    run = next(r for r in selected if r.id == run_id)
    means: dict[str, dict[str, float]] = run.summary.get("criteria_means") or {}
    scales = _scales(run)
    names = [n for n in means if n in scales]
    if not names:
        st.caption("No per-criterion scores were recorded for this run.")
    elif pairwise:
        st.plotly_chart(
            charts.criterion_comparison(
                names,
                [_normalise(means[n].get("a", 0.0), scales[n]) for n in names],
                [_normalise(means[n].get("b", 0.0), scales[n]) for n in names],
                scale=(0.0, 1.0),
                theme=theme,
            ),
            key="an:crit-chart",
            config=charts.CHART_CONFIG,
        )
    else:
        st.plotly_chart(
            charts.ranked_bars(
                names,
                [_normalise(means[n].get("score", 0.0), scales[n]) for n in names],
                theme=theme,
                x_title="Mean score, normalised to 0-1",
            ),
            key="an:crit-chart",
            config=charts.CHART_CONFIG,
        )
    if names:
        st.dataframe(
            [
                {
                    "criterion": n,
                    "scale": f"{scales[n][0]:g}-{scales[n][1]:g}",
                    **{f"mean ({side})": v for side, v in means[n].items()},
                }
                for n in names
            ],
            hide_index=True,
        )

# --- bias signals ----------------------------------------------------------------------
SIGNALS = {
    "position_inconsistent": "Verdict changed when the outputs were swapped (direct position-"
    "bias measurement; needs position-bias mitigation).",
    "possible_verbosity_bias": "The winner was much longer than the other output.",
    "self_preference_risk": "A candidate came from the judge's own model family.",
    "possible_prompt_injection": "An output contained text aimed at the judge.",
    "provider_fallback": "A fallback provider answered instead of the configured judge.",
}
with bias_tab:
    st.caption(
        "Share of successfully judged rows that carried each reliability warning. These are "
        "heuristic signals: a flag isn't proof of bias, and no flag isn't proof of its absence."
    )
    shares: list[dict[str, Any]] = []
    for r in selected:
        counts = r.summary.get("warning_counts") or {}
        ok = r.succeeded or 0
        entry: dict[str, Any] = {"run": labels[r.id]}
        for code in SIGNALS:
            entry[code] = (counts.get(code, 0) / ok) if ok else None
        shares.append(entry)
    latest = selected[-1]
    latest_ok = latest.succeeded or 0
    latest_counts = latest.summary.get("warning_counts") or {}
    if latest_ok:
        st.plotly_chart(
            charts.ranked_bars(
                list(SIGNALS),
                [latest_counts.get(code, 0) / latest_ok for code in SIGNALS],
                theme=theme,
                x_title=f"Share of rows flagged in the latest run ({labels[latest.id]})",
            ),
            key="an:bias",
            config=charts.CHART_CONFIG,
        )
    st.dataframe(shares, hide_index=True)
    with st.expander("What each signal means"):
        for code, meaning in SIGNALS.items():
            st.markdown(f"- `{code}`: {meaning}")
    if pairwise and not any(
        r.config.get("settings", {}).get("mitigate_position_bias") for r in selected
    ):
        st.caption(
            "None of these runs used position-bias mitigation, so position inconsistency "
            "couldn't be measured. Turn it on in the sidebar to measure it directly."
        )

# --- reliability (operational) ---------------------------------------------------------
with ops_tab:
    st.caption(
        "How dependable each run was. For repeated-judgment consistency and agreement "
        "between judges, use the Reliability page."
    )
    ops: list[dict[str, Any]] = []
    for r in selected:
        stored = rows_of(r)
        latencies = [
            float(row.result["metadata"]["latency_seconds"])
            for row in stored
            if row.result and "latency_seconds" in (row.result.get("metadata") or {})
        ]
        fallback_rows = sum(
            1 for row in stored if row.result and (row.result.get("metadata") or {}).get("fallback")
        )
        cached_rows = sum(
            1
            for row in stored
            if row.result and (row.result.get("metadata") or {}).get("cache_hit")
        )
        ops.append(
            {
                "run": labels[r.id],
                "status": r.status,
                "failure rate": (r.failed / r.total) if r.total else None,
                "mean judge latency (s)": fmean(latencies) if latencies else None,
                "rows answered by a fallback": fallback_rows,
                "rows from the cache": cached_rows,
                "duration (s)": round(r.duration_seconds, 1),
            }
        )
    st.dataframe(ops, hide_index=True)
