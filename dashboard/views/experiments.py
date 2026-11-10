"""Experiments: browse recorded runs, compare two, and check for regressions."""

from __future__ import annotations

import streamlit as st

from dashboard import charts
from dashboard.common import app_settings, chart_theme, experiment_store
from llm_judge.experiments import compare_runs

settings = app_settings()
store = experiment_store()
st.title("Experiments")
if settings.demo_mode:
    st.caption("Public demo: runs live only in your browser session and disappear when it ends.")

runs = store.list_runs(limit=200)
if not runs:
    st.info(
        "No runs yet. Run a benchmark, or evaluate a dataset with 'Record as experiment', "
        "then come back to compare runs.",
        icon=":material/history:",
    )
    st.stop()

st.dataframe(
    [
        {
            "id": r.id,
            "name": r.name,
            "experiment": r.experiment or "-",
            "kind": r.kind,
            "status": r.status,
            "judge": f"{r.judge_provider}/{r.judge_model}",
            "rows ok": f"{r.succeeded}/{r.total}",
            "created": r.created_at,
        }
        for r in runs
    ],
    hide_index=True,
)

st.markdown("#### Compare two runs")
labels = {r.id: f"{r.name} [{r.kind}] {r.created_at}" for r in runs}
left, right = st.columns(2)
baseline_id = left.selectbox(
    "Baseline", [r.id for r in runs][::-1], format_func=labels.__getitem__, key="ex:base"
)
if baseline_id is None:
    st.stop()
baseline_kind = next(r.kind for r in runs if r.id == baseline_id)
candidates = [r.id for r in runs if r.kind == baseline_kind and r.id != baseline_id]
if not candidates:
    st.caption(f"Record another {baseline_kind} run to compare against this one.")
    st.stop()
candidate_id = right.selectbox(
    "Candidate", candidates, format_func=labels.__getitem__, key="ex:cand"
)
if candidate_id is None:
    st.stop()
min_effect = st.slider(
    "Minimum effect size",
    0.0,
    0.2,
    0.02,
    0.01,
    key="ex:effect",
    help="The smallest mean change (on the 0-1 score) that counts as a regression or improvement.",
)

comparison = compare_runs(store, baseline_id, candidate_id, min_effect=min_effect)
verdict = comparison.verdict
message = {
    "regression": ("Regression detected", st.error, ":material/trending_down:"),
    "improvement": ("Improvement detected", st.success, ":material/trending_up:"),
    "no_significant_change": ("No significant change", st.info, ":material/trending_flat:"),
    "insufficient_data": ("Not enough matched rows to compare", st.info, ":material/help:"),
}[verdict]
message[1](message[0], icon=message[2])

if comparison.mean_delta is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline mean", f"{comparison.baseline_mean:.3f}")
    c2.metric(
        "Candidate mean",
        f"{comparison.candidate_mean:.3f}",
        delta=f"{comparison.mean_delta:+.3f}",
    )
    c3.metric("Matched rows", comparison.matched_rows)
    st.caption(
        f"95% confidence interval for the mean change: [{comparison.ci_low:+.3f}, "
        f"{comparison.ci_high:+.3f}] ({comparison.ci_method}). Metric: {comparison.metric}. "
        "These are judge-scored differences, so the judge's own noise is part of them."
    )
for warning in comparison.warnings:
    st.warning(warning, icon=":material/info:")

base_rows = {r.row_id: r for r in store.get_rows(baseline_id)}
cand_rows = {r.row_id: r for r in store.get_rows(candidate_id)}
deltas: list[tuple[str, float, float]] = []
for row_id, b in base_rows.items():
    c = cand_rows.get(row_id)
    if c is None or not (b.ok and c.ok):
        continue
    if baseline_kind == "pairwise":
        b_a, b_b, c_a, c_b = (
            b.overall_score_a,
            b.overall_score_b,
            c.overall_score_a,
            c.overall_score_b,
        )
        if b_a is None or b_b is None or c_a is None or c_b is None:
            continue
        before, after = b_a - b_b, c_a - c_b
    else:
        if b.overall_score is None or c.overall_score is None:
            continue
        before, after = b.overall_score, c.overall_score
    deltas.append((row_id, before, after))

if deltas:
    st.plotly_chart(
        charts.delta_bars(
            [row for row, _, _ in deltas],
            [after - before for _, before, after in deltas],
            theme=chart_theme(),
        ),
        key="ex:delta",
        config=charts.CHART_CONFIG,
    )
    st.dataframe(
        [
            {"row": row, "baseline": before, "candidate": after, "change": after - before}
            for row, before, after in sorted(deltas, key=lambda d: d[2] - d[1])
        ],
        hide_index=True,
    )

with st.expander("Run details"):
    for run_id in (baseline_id, candidate_id):
        run = store.require_run(run_id)
        st.markdown(f"**{run.name}** · `{run.id}`")
        st.json(
            {
                "judge": f"{run.judge_provider}/{run.judge_model}",
                "prompt_version": run.prompt_version,
                "criteria": run.criteria,
                "dataset": run.dataset_source,
                "dataset_fingerprint": run.dataset_fingerprint,
                "llm_judge_version": run.config.get("llm_judge_version"),
            },
            expanded=False,
        )

if not settings.demo_mode:
    with st.expander("Delete a run"):
        doomed = st.selectbox(
            "Run", [r.id for r in runs], format_func=labels.__getitem__, key="ex:del"
        )
        if doomed is not None and st.button("Delete run", key="ex:delete"):
            store.delete_run(doomed)
            st.rerun()
