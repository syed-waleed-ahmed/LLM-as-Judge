"""Benchmarks: run a bundled sample dataset and compare the systems in it."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    current_judge,
    experiment_store,
    render_summary,
    row_limit,
    run_safely,
    show_stop_reason,
)
from llm_judge.datasets import SAMPLE_DATASETS, DatasetRunner, load_sample, summarize
from llm_judge.experiments import record_dataset_run

st.title("Benchmarks")
st.caption(
    "Six small, original, synthetic datasets. Their candidate systems (system-alpha, "
    "system-beta) are neutral labels, not real models. They're too small to rank real "
    "models; use them to see the pipeline work end to end."
)

name = st.selectbox(
    "Benchmark",
    list(SAMPLE_DATASETS),
    key="bm:name",
    format_func=lambda n: f"{n} ({SAMPLE_DATASETS[n].kind})",
) or next(iter(SAMPLE_DATASETS))
sample = SAMPLE_DATASETS[name]
st.markdown(f"{sample.description}  \nCriteria: {', '.join(sample.criteria)}")
records = load_sample(name)[: row_limit()]
st.caption(f"{len(records)} row(s). Each run is recorded in the 'benchmarks' experiment.")

if st.button("Run benchmark", type="primary", key="bm:run"):
    judge = current_judge()
    bar = st.progress(0.0, text="Starting...")

    def progress(done: int, total: int) -> None:
        bar.progress(done / max(total, 1), text=f"{done}/{total} rows")

    run = run_safely(
        lambda: DatasetRunner(judge, list(sample.criteria), progress=progress).run(records)
    )
    if run:
        show_stop_reason(run.stop_reason)
    if run and run.succeeded:
        provider = getattr(judge.provider, "name", "judge")
        info = record_dataset_run(
            experiment_store(),
            run,
            judge,
            experiment="benchmarks",
            name=f"{name} · {provider}",
            dataset_source=f"sample:{name}",
        )
        if provider == "mock":
            st.warning(
                "These scores come from the mock heuristic, not an LLM. Switch to the live "
                "judge in the sidebar for real judgments.",
                icon=":material/science:",
            )
        render_summary(summarize(run), key="bm")
        st.caption(f"Recorded as run `{info.id}`. Compare runs on the Experiments page.")
