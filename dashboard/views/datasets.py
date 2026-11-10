"""Datasets: upload or pick a dataset, validate it, run it, download the results."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from dashboard import charts
from dashboard.common import (
    app_settings,
    chart_theme,
    criteria_picker,
    current_judge,
    experiment_store,
    render_summary,
    row_limit,
    run_safely,
    show_stop_reason,
)
from llm_judge.datasets import (
    SAMPLE_DATASETS,
    DatasetRunner,
    load_dataset,
    load_sample,
    sample_text,
    summarize,
    write_outcomes,
)
from llm_judge.exceptions import LLMJudgeError
from llm_judge.experiments import record_dataset_run

settings = app_settings()
st.title("Datasets")
st.caption(
    "CSV, JSON, JSONL or YAML. The kind is inferred from the columns: output_a/output_b "
    "(pairwise), output (pointwise), or output + reference (reference)."
)

source = st.radio("Source", ["Bundled sample", "Upload a file"], horizontal=True, key="ds:source")
records, report = [], None
upload_id = ""
if source == "Bundled sample":
    name = st.selectbox("Sample", list(SAMPLE_DATASETS), key="ds:sample") or next(
        iter(SAMPLE_DATASETS)
    )
    st.caption(SAMPLE_DATASETS[name].description)
    st.download_button(
        "Download this sample",
        sample_text(name),
        f"{name}.jsonl",
        key="ds:sample-dl",
        on_click="ignore",
    )
    records = load_sample(name, settings=settings)
    records, report = load_dataset(records, settings=settings)
else:
    upload = st.file_uploader(
        "Dataset file", type=["csv", "jsonl", "json", "yaml", "yml"], key="ds:upload"
    )
    if upload is not None:
        upload_id = upload.file_id
        suffix = Path(upload.name).suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(upload.getvalue())
            try:
                records, report = load_dataset(path, settings=settings)
            except LLMJudgeError as exc:  # not an evaluation: don't count it as one
                st.error(f"Can't read this file: {exc}")

if report is not None:
    with st.expander(
        f"Validation: {report.valid_rows}/{report.rows} valid rows · kind {report.kind}",
        expanded=not report.ok,
    ):
        if report.fingerprint:
            st.caption(f"Dataset fingerprint (version): `{report.fingerprint}`")
        for issue in report.issues:
            where = f"row {issue.row}" if issue.row else "dataset"
            (st.error if issue.severity == "error" else st.warning)(f"{where}: {issue.message}")
        st.dataframe(report.preview, hide_index=True)

if records:
    limit = row_limit()
    if len(records) > limit:
        st.info(f"Evaluating the first {limit} of {len(records)} rows (row limit).")
        records = records[:limit]
    default = (
        SAMPLE_DATASETS[name].criteria
        if source == "Bundled sample"
        else (
            ["Correctness", "Completeness"]
            if report and report.kind == "reference"
            else ["Helpfulness"]
        )
    )
    # One picker per dataset, so switching samples restores that sample's criteria.
    picker_key = f"ds:{name}" if source == "Bundled sample" else f"ds:upload:{upload_id}"
    criteria = criteria_picker(
        picker_key, default=default, reference=report is not None and report.kind != "pairwise"
    )
    record_as = st.text_input("Record as experiment (optional)", key="ds:experiment")
    if st.button(f"Evaluate {len(records)} row(s)", type="primary", key="ds:run") and (
        criteria or st.error("Choose at least one criterion.")
    ):
        bar = st.progress(0.0, text="Starting...")
        judge = current_judge()

        def progress(done: int, total: int) -> None:
            bar.progress(done / max(total, 1), text=f"{done}/{total} rows")

        run = run_safely(lambda: DatasetRunner(judge, criteria, progress=progress).run(records))
        if run:
            show_stop_reason(run.stop_reason)
            summary = summarize(run)
            render_summary(summary, key="ds")
            scores = [
                (o.result.overall_score if hasattr(o.result, "overall_score") else None)
                for o in run.outcomes
                if o.ok and o.result is not None
            ]
            values = [s for s in scores if isinstance(s, float)]
            if values:
                st.plotly_chart(
                    charts.score_histogram(
                        values, theme=chart_theme(), x_title="Overall score (0-1)"
                    ),
                    key="ds:hist",
                    config=charts.CHART_CONFIG,
                )
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / "results.csv"
                jsonl_path = Path(tmp) / "results.jsonl"
                write_outcomes(run.outcomes, csv_path)
                write_outcomes(run.outcomes, jsonl_path)
                csv_text = csv_path.read_text(encoding="utf-8")
                jsonl_text = jsonl_path.read_text(encoding="utf-8")
            left, right = st.columns(2)
            left.download_button(
                "Download CSV", csv_text, "results.csv", key="ds:csv", on_click="ignore"
            )
            right.download_button(
                "Download JSONL", jsonl_text, "results.jsonl", key="ds:jsonl", on_click="ignore"
            )
            if record_as.strip() and run.succeeded:
                info = record_dataset_run(
                    experiment_store(),
                    run,
                    judge,
                    experiment=record_as.strip(),
                    dataset_source=report.source if report else "upload",
                )
                st.success(f"Recorded run `{info.id}` in experiment '{record_as.strip()}'.")
