"""Playground: try every evaluation mode on a single example."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    app_settings,
    criteria_picker,
    current_judge,
    render_pairwise,
    render_pointwise,
    render_ranking,
    run_safely,
)
from llm_judge.datasets import load_sample
from llm_judge.datasets.records import PairwiseRecord, ReferenceRecord

settings = app_settings()
max_chars = settings.demo_max_input_chars if settings.demo_mode else None

st.title("Playground")
st.caption("Each tab runs one evaluation with the judge selected in the sidebar.")


def _examples() -> dict[str, PairwiseRecord]:
    picks = {
        "Summary": ("summarization", 1),
        "Code": ("code_generation", 0),
        "Safety": ("safety", 1),
    }
    out = {}
    for label, (name, index) in picks.items():
        record = load_sample(name)[index]
        assert isinstance(record, PairwiseRecord)
        out[label] = record
    return out


def _reference_examples() -> dict[str, ReferenceRecord]:
    out = {}
    for label, (name, index) in {"Factual QA": ("general_qa", 5), "RAG": ("rag", 1)}.items():
        record = load_sample(name)[index]
        assert isinstance(record, ReferenceRecord)
        out[label] = record
    return out


def _fill_pairwise() -> None:
    record = _examples().get(str(st.session_state.get("pw:example") or ""))
    if record:
        st.session_state["pw:task"] = record.task_description
        st.session_state["pw:a"] = record.output_a
        st.session_state["pw:b"] = record.output_b


def _fill_reference() -> None:
    record = _reference_examples().get(str(st.session_state.get("ref:example") or ""))
    if record:
        st.session_state["ref:task"] = record.task_description
        st.session_state["ref:output"] = record.output
        st.session_state["ref:reference"] = record.reference


pairwise_tab, pointwise_tab, reference_tab, rank_tab = st.tabs(
    ["Compare two outputs", "Grade one output", "Check against a reference", "Rank candidates"]
)

with pairwise_tab:
    st.pills(
        "Load an example",
        list(_examples()),
        key="pw:example",
        on_change=_fill_pairwise,
    )
    task = st.text_area("Task", key="pw:task", height=100, max_chars=max_chars)
    left, right = st.columns(2)
    output_a = left.text_area("Output A", key="pw:a", height=180, max_chars=max_chars)
    output_b = right.text_area("Output B", key="pw:b", height=180, max_chars=max_chars)
    criteria = criteria_picker("pw", default=["Helpfulness"])
    if st.button("Compare", type="primary", key="pw:run"):
        if not (task.strip() and output_a.strip() and output_b.strip()):
            st.error("Fill in the task and both outputs.")
        elif not criteria:
            st.error("Choose at least one criterion.")
        else:
            with st.spinner("Judging..."):
                result = run_safely(
                    lambda: current_judge().evaluate(task, output_a, output_b, criteria)
                )
            if result:
                render_pairwise(result, criteria, key="pw")

with pointwise_tab:
    task = st.text_area("Task", key="pt:task", height=100, max_chars=max_chars)
    output = st.text_area("Output", key="pt:output", height=180, max_chars=max_chars)
    criteria = criteria_picker("pt", default=["Helpfulness", "Conciseness"])
    if st.button("Grade", type="primary", key="pt:run"):
        if not (task.strip() and output.strip()):
            st.error("Fill in the task and the output.")
        elif not criteria:
            st.error("Choose at least one criterion.")
        else:
            with st.spinner("Judging..."):
                graded = run_safely(
                    lambda: current_judge().evaluate_pointwise(task, output, criteria)
                )
            if graded:
                render_pointwise(graded, key="pt")

with reference_tab:
    st.pills(
        "Load an example",
        list(_reference_examples()),
        key="ref:example",
        on_change=_fill_reference,
    )
    task = st.text_area("Task", key="ref:task", height=100, max_chars=max_chars)
    left, right = st.columns(2)
    output = left.text_area("Output", key="ref:output", height=160, max_chars=max_chars)
    reference = right.text_area(
        "Reference (expected) answer", key="ref:reference", height=160, max_chars=max_chars
    )
    criteria = criteria_picker("ref", default=["Correctness", "Completeness"], reference=True)
    if st.button("Check", type="primary", key="ref:run"):
        if not (task.strip() and output.strip() and reference.strip()):
            st.error("Fill in the task, the output and the reference answer.")
        elif not criteria:
            st.error("Choose at least one criterion.")
        else:
            with st.spinner("Judging..."):
                graded = run_safely(
                    lambda: current_judge().evaluate_reference(task, output, reference, criteria)
                )
            if graded:
                render_pointwise(graded, key="ref")

with rank_tab:
    limit = settings.demo_max_candidates if settings.demo_mode else settings.max_candidates
    task = st.text_area(
        "Task",
        key="rk:task",
        height=80,
        max_chars=max_chars,
        value="Write a one-line tagline for a privacy-focused note-taking app.",
    )
    st.caption(f"Candidates (2 to {limit}). Edit the table or add rows.")
    table = st.data_editor(
        [
            {"id": "calm", "output": "Your thoughts, encrypted and only ever yours."},
            {"id": "generic", "output": "A good app for notes."},
            {"id": "busy", "output": "Notes! Fast! Private! Secure! Sync! Everything!"},
        ],
        num_rows="dynamic",
        key="rk:table",
        width="stretch",
    )
    strategy = st.radio(
        "Strategy",
        ["round_robin", "pointwise"],
        horizontal=True,
        key="rk:strategy",
        format_func={
            "round_robin": "Round robin (every pair, most reliable)",
            "pointwise": "Pointwise (one call per candidate)",
        }.__getitem__,
    )
    criteria = criteria_picker("rk", default=["Creativity", "Brand Tone Adherence"])
    if st.button("Rank", type="primary", key="rk:run"):
        candidates = {
            str(row.get("id") or i + 1).strip(): str(row.get("output") or "")
            for i, row in enumerate(table)
            if str(row.get("output") or "").strip()
        }
        if not task.strip() or not 2 <= len(candidates) <= limit:
            st.error(f"Give a task and between 2 and {limit} non-empty candidates.")
        elif not criteria:
            st.error("Choose at least one criterion.")
        else:
            with st.spinner("Judging..."):
                ranking = run_safely(
                    lambda: current_judge().rank(
                        task,
                        candidates,
                        criteria,
                        strategy="pointwise" if strategy == "pointwise" else "round_robin",
                    )
                )
            if ranking:
                render_ranking(ranking, key="rk")
