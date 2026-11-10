"""Overview: what this is, which judge is active, and recent activity."""

from __future__ import annotations

import streamlit as st

from dashboard.common import app_settings, experiment_store, judge_mode, live_status

settings = app_settings()
st.title("LLM-as-Judge")
st.markdown(
    "An open-source platform for evaluating LLM outputs with an LLM judge: compare two "
    "answers, grade one against criteria or a reference answer, rank several candidates, "
    "run whole datasets, and track experiments for regressions."
)

if judge_mode() == "live":
    st.success(
        f"Live judging with **{settings.provider}**. Results are real LLM judgments.",
        icon=":material/gavel:",
    )
else:
    available, detail = live_status(settings)
    hint = "" if available else f" (live judging unavailable: {detail.rstrip('.')})"
    st.info(
        "You're using the **offline mock judge**, a deterministic heuristic rather than an "
        f"LLM. It shows how everything works without any API key{hint}.",
        icon=":material/science:",
    )

stats = experiment_store().statistics()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Evaluations this session", int(st.session_state.get("evaluations", 0)))
c2.metric("Recorded runs", stats["runs"])
c3.metric("Experiments", stats["experiments"])
c4.metric("Rows evaluated", stats["rows_evaluated"])

st.markdown("#### Start here")
left, middle, right = st.columns(3)
with left, st.container(border=True):
    st.page_link("dashboard/views/playground.py", label="Playground", icon=":material/compare:")
    st.caption("Judge a single example: A vs B, a grade, a reference check, or a ranking.")
with middle, st.container(border=True):
    st.page_link("dashboard/views/benchmarks.py", label="Benchmarks", icon=":material/speed:")
    st.caption("Run a bundled sample dataset and compare systems.")
with right, st.container(border=True):
    st.page_link("dashboard/views/experiments.py", label="Experiments", icon=":material/history:")
    st.caption("Compare two runs and check for regressions with confidence intervals.")

st.markdown("#### How the judge stays honest")
st.markdown(
    "- **Structured, validated output:** the reply must match the schema and the exact "
    "criteria requested, or it's repaired or rejected.\n"
    "- **Position-bias mitigation:** optionally judge both orders and average.\n"
    "- **Injection resistance:** candidate text is fenced as untrusted data.\n"
    "- **Reliability warnings:** verbosity bias, self-preference, order sensitivity and "
    "suspected prompt injection are flagged on every result."
)

recent = experiment_store().list_runs(limit=5)
if recent:
    st.markdown("#### Recent runs")
    st.dataframe(
        [
            {
                "run": r.name,
                "kind": r.kind,
                "judge": f"{r.judge_provider}/{r.judge_model}",
                "rows ok": f"{r.succeeded}/{r.total}",
                "created": r.created_at,
            }
            for r in recent
        ],
        hide_index=True,
    )
