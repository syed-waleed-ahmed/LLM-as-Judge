"""Streamlit UI for the LLM-as-Judge framework.

Run with::

    streamlit run app.py

This is a thin presentation layer over the :mod:`llm_judge` package: all the
evaluation logic lives in the library, so the UI, CLI, and any programmatic
callers share exactly the same behaviour.
"""

from __future__ import annotations

import json

import streamlit as st

from llm_judge import CRITERIA_PRESETS, Judge, JudgeResult, Winner, get_settings
from llm_judge.exceptions import ConfigurationError, LLMJudgeError

st.set_page_config(page_title="LLM-as-Judge", layout="wide")


@st.cache_resource(show_spinner=False)
def _get_judge(mitigate_bias: bool) -> Judge:
    """Build (and cache) a Judge. Cached per bias setting to avoid re-creating clients."""
    settings = get_settings().model_copy(update={"mitigate_position_bias": mitigate_bias})
    return Judge(settings=settings)


def _sidebar() -> tuple[list[str], bool]:
    with st.sidebar:
        st.header("Evaluation Settings")

        selected = st.multiselect(
            "Criteria",
            options=list(CRITERIA_PRESETS.keys()),
            default=["Helpfulness"],
            help="Pick one or more built-in criteria.",
        )

        custom_raw = st.text_input(
            "Custom criteria (comma-separated)",
            placeholder="e.g. Factual Accuracy, Politeness",
            help="Add your own criteria on top of the presets.",
        )
        custom = [c.strip() for c in custom_raw.split(",") if c.strip()]

        st.markdown("---")
        mitigate_bias = st.toggle(
            "Mitigate position bias",
            value=False,
            help="Runs each comparison in both orders and averages the scores. "
            "More reliable, but doubles the number of API calls.",
        )

        st.markdown("---")
        st.caption(
            "Tip: write a clear task description so the judge knows what 'good' means."
        )

        criteria = selected + custom
        return criteria, mitigate_bias


def _render_result(result: JudgeResult) -> None:
    st.subheader("Judgment Results")

    winner = result.overall_winner
    if winner == Winner.A:
        st.success("Output A is preferred overall.")
    elif winner == Winner.B:
        st.success("Output B is preferred overall.")
    else:
        st.info("The judge declared an overall tie.")

    if result.overall_comment:
        st.write(result.overall_comment)

    st.markdown("### Per-criterion breakdown")
    for score in result.criteria:
        with st.container(border=True):
            st.markdown(f"**{score.criterion}** — winner: `{score.winner.value}`")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Output A", score.output_a_score)
                st.caption(score.output_a_explanation or "—")
            with c2:
                st.metric("Output B", score.output_b_score)
                st.caption(score.output_b_explanation or "—")

    with st.expander("Metadata & raw JSON"):
        st.json(result.model_dump(mode="json"))

    st.download_button(
        "Download result (JSON)",
        data=json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        file_name="judgment.json",
        mime="application/json",
    )


def main() -> None:
    st.title("LLM-as-Judge Evaluation Framework")
    st.markdown(
        "Let a powerful LLM act as an impartial judge to compare **two model "
        "outputs** across one or more criteria — with optional position-bias "
        "mitigation for more reliable verdicts."
    )

    criteria, mitigate_bias = _sidebar()

    col_task, col_a, col_b = st.columns(3)
    with col_task:
        task = st.text_area(
            "Task Description",
            height=240,
            placeholder="E.g. 'Write a short, friendly promotional email...'",
        )
    with col_a:
        output_a = st.text_area("Output A", height=240, placeholder="First model's output")
    with col_b:
        output_b = st.text_area("Output B", height=240, placeholder="Second model's output")

    st.markdown("---")

    if st.button("Evaluate Outputs", type="primary"):
        if not task.strip() or not output_a.strip() or not output_b.strip():
            st.error("Please fill in the task description and both outputs.")
            return
        if not criteria:
            st.error("Please select or enter at least one criterion.")
            return

        try:
            judge = _get_judge(mitigate_bias)
            with st.spinner("Asking the LLM judge..."):
                result = judge.evaluate(task, output_a, output_b, criteria)
        except ConfigurationError as exc:
            st.error(f"Configuration problem: {exc}")
            return
        except LLMJudgeError as exc:
            st.error(f"Evaluation failed: {exc}")
            return

        _render_result(result)


if __name__ == "__main__":
    main()
