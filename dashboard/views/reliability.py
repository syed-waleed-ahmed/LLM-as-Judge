"""Reliability: how stable is the judge, and how much do two judges agree?"""

from __future__ import annotations

import streamlit as st

from dashboard.common import app_settings, current_judge, experiment_store, judge_mode, run_safely
from llm_judge.reliability import judge_agreement, measure_consistency

settings = app_settings()
st.title("Judge reliability")
st.caption(
    "LLM judges are noisy. Measure it before trusting a result. Intervals come from repeated "
    "samples, never from the model's own confidence."
)

consistency_tab, agreement_tab = st.tabs(["Repeated-judgment consistency", "Judge agreement"])

with consistency_tab:
    task = st.text_input(
        "Task", value="Explain what an API is to a non-technical reader", key="rel:task"
    )
    left, right = st.columns(2)
    output_a = left.text_area(
        "Output A",
        value="An API is like a waiter: it carries your order to the kitchen and brings back "
        "what you asked for.",
        key="rel:a",
    )
    output_b = right.text_area(
        "Output B", value="An API is an application programming interface.", key="rel:b"
    )
    max_samples = 5 if settings.demo_mode else 10
    samples = st.slider("Samples", 2, max_samples, min(5, max_samples), key="rel:samples")
    temperature = st.slider(
        "Judge temperature",
        0.0,
        1.0,
        0.7,
        0.1,
        key="rel:temperature",
        help="At 0, only provider nondeterminism shows. Higher values probe sensitivity.",
    )
    if judge_mode() == "mock":
        st.caption("The mock judge is deterministic, so expect perfect consistency.")
    if st.button(f"Judge {samples} times", type="primary", key="rel:run"):
        with st.spinner("Sampling..."):
            report = run_safely(
                lambda: measure_consistency(
                    current_judge(temperature=temperature),
                    task,
                    output_a=output_a,
                    output_b=output_b,
                    samples=samples,
                    max_workers=2,
                )
            )
        if report:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Agreement with modal verdict",
                "n/a" if report.agreement_rate is None else f"{report.agreement_rate:.0%}",
            )
            c2.metric("Modal verdict", report.modal_winner or "-")
            c3.metric("Successful samples", f"{report.succeeded}/{report.samples}")
            if report.mean is not None:
                st.caption(
                    f"Score margin (A - B): mean {report.mean:+.3f}, sd {report.std:.3f}, "
                    f"95% bootstrap CI [{report.ci_low:+.3f}, {report.ci_high:+.3f}]."
                )
            st.dataframe(
                [{"verdict": k, "samples": v} for k, v in report.winner_counts.items()],
                hide_index=True,
            )
            for note in report.notes:
                st.info(note, icon=":material/info:")
            for error in report.errors:
                st.warning(f"Failed sample: {error}")

with agreement_tab:
    st.caption(
        "Compare two recorded runs over the same rows, typically the same benchmark judged by "
        "two different judges."
    )
    runs = experiment_store().list_runs(limit=200)
    if len(runs) < 2:
        st.info("Record at least two runs (e.g. on the Benchmarks page) to compare judges.")
    else:
        labels = {r.id: f"{r.name} [{r.kind}] {r.judge_provider}/{r.judge_model}" for r in runs}
        left, right = st.columns(2)
        run_a = left.selectbox("Run A", list(labels), format_func=labels.__getitem__, key="rel:ra")
        kind = next(r.kind for r in runs if r.id == run_a)
        others = [r.id for r in runs if r.kind == kind and r.id != run_a]
        run_b = right.selectbox("Run B", others, format_func=labels.__getitem__, key="rel:rb")
        if run_a is not None and run_b is not None:
            agreement = judge_agreement(experiment_store(), run_a, run_b)
            c1, c2, c3 = st.columns(3)
            c1.metric("Matched rows", agreement.matched_rows)
            if agreement.cohen_kappa is not None:
                c2.metric("Winner agreement", f"{agreement.raw_agreement:.0%}")
                c3.metric("Cohen's kappa", f"{agreement.cohen_kappa:+.2f}")
            if agreement.mean_absolute_difference is not None:
                c2.metric(
                    "Spearman rho",
                    "n/a" if agreement.spearman is None else f"{agreement.spearman:+.2f}",
                )
                c3.metric("Mean |score difference|", f"{agreement.mean_absolute_difference:.3f}")
            for note in agreement.notes:
                st.caption(note)
        elif run_a is not None:
            st.caption(f"Record another {kind} run to compare with this one.")
