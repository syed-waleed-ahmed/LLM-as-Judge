import streamlit as st

from judge import judge_pair, JudgeError
from prompts import CRITERIA_PRESETS


st.set_page_config(
    page_title="LLM-as-Judge (Groq)",
    layout="wide",
)


def main():
    st.title("🤖 LLM-as-Judge Evaluation Framework (Groq)")

    st.markdown(
        """
This app lets a powerful LLM judge compare **two model outputs** on a chosen criterion.

1. Describe the task.
2. Paste two outputs (A and B).
3. Pick an evaluation criterion or define your own.
4. Let the Groq-backed judge score and explain.
"""
    )

    with st.sidebar:
        st.header("⚙️ Evaluation Settings")
        preset_names = list(CRITERIA_PRESETS.keys())
        preset_choice = st.selectbox("Preset criterion", preset_names)
        use_custom = st.checkbox("Use custom criterion instead")

        if use_custom:
            criterion_name = st.text_input(
                "Custom criterion name",
                value="Overall Quality",
            )
        else:
            criterion_name = preset_choice

        st.markdown("---")
        st.caption(
            "Tip: make the task description very clear so the judge knows what 'good' means."
        )

    col_task, col_a, col_b = st.columns(3)

    with col_task:
        task_description = st.text_area(
            "Task Description",
            height=220,
            placeholder="E.g. 'Write a short, friendly promotional email about our new product launch...'",
        )

    with col_a:
        output_a = st.text_area(
            "Output A",
            height=220,
            placeholder="Paste the first model's output here",
        )

    with col_b:
        output_b = st.text_area(
            "Output B",
            height=220,
            placeholder="Paste the second model's output here",
        )

    st.markdown("---")

    evaluate_button = st.button("⚖️ Evaluate Outputs", type="primary")

    if evaluate_button:
        if not task_description or not output_a or not output_b:
            st.error("Please fill in the task description and both outputs.")
            return

        with st.spinner("Asking the Groq LLM judge..."):
            try:
                result = judge_pair(
                    task_description=task_description,
                    output_a=output_a,
                    output_b=output_b,
                    criterion_name=criterion_name,
                )
            except JudgeError as e:
                st.error(str(e))
                return

        show_results(result)


def show_results(result: dict):
    st.subheader("📊 Judgment Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Criterion", result.get("criterion", "N/A"))

    with col2:
        st.metric("Output A Score", str(result.get("output_a_score", "N/A")))

    with col3:
        st.metric("Output B Score", str(result.get("output_b_score", "N/A")))

    st.markdown("### Explanations")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Output A**")
        st.write(result.get("output_a_explanation", ""))

    with col_b:
        st.markdown("**Output B**")
        st.write(result.get("output_b_explanation", ""))

    st.markdown("### Overall Decision")

    winner = result.get("winner", "tie")
    if winner == "A":
        st.success("🏆 Output A is preferred.")
    elif winner == "B":
        st.success("🏆 Output B is preferred.")
    else:
        st.info("⚖️ The judge declared a tie.")

    st.markdown("**Overall Comment**")
    st.write(result.get("overall_comment", ""))

    st.markdown("---")
    st.caption("Adjust the task, outputs, or criterion and run again to explore more.")


if __name__ == "__main__":
    main()
