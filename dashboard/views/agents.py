"""Agent evaluation: deterministic trajectory checks plus a judged grade."""

from __future__ import annotations

import json

import streamlit as st
from pydantic import ValidationError as PydanticValidationError

from dashboard.common import current_judge, render_pointwise, run_safely
from llm_judge.agents import AgentTrace

EXAMPLE = {
    "task": "What is the weather in Paris right now? Answer in one sentence.",
    "tools": [
        {
            "name": "geocode",
            "description": "Find coordinates for a place",
            "required_args": ["place"],
        },
        {
            "name": "weather",
            "description": "Current weather at coordinates",
            "required_args": ["lat", "lon"],
        },
    ],
    "expected_tools": ["geocode", "weather"],
    "max_steps": 6,
    "steps": [
        {"type": "thought", "content": "I need Paris's coordinates first."},
        {"type": "tool_call", "tool": "geocode", "arguments": {"place": "Paris"}},
        {"type": "tool_result", "tool": "geocode", "content": '{"lat": 48.86, "lon": 2.35}'},
        {"type": "tool_call", "tool": "weather", "arguments": {"lat": 48.86}},
        {"type": "tool_result", "tool": "weather", "error": True, "content": "missing lon"},
        {"type": "tool_call", "tool": "weather", "arguments": {"lat": 48.86, "lon": 2.35}},
        {"type": "tool_result", "tool": "weather", "content": "Sunny, 21 C"},
    ],
    "final_answer": "It is currently sunny and 21 C in Paris.",
}

st.title("Agent evaluation")
st.caption(
    "Paste or upload an agent trace (any framework, converted to this simple JSON form). "
    "Deterministic checks catch invalid or failed tool use; the judge grades the trajectory."
)

upload = st.file_uploader("Trace file (JSON)", type=["json"], key="ag:upload")
default = upload.getvalue().decode("utf-8") if upload else json.dumps(EXAMPLE, indent=2)
text = st.text_area("Trace", value=default, height=360, key="ag:trace")

if st.button("Evaluate", type="primary", key="ag:run"):
    try:
        trace = AgentTrace.model_validate_json(text)
    except PydanticValidationError as exc:
        st.error(
            f"The trace is not valid: {exc.errors()[0].get('msg')} at {exc.errors()[0].get('loc')}"
        )
    else:
        with st.spinner("Checking and judging..."):
            result = run_safely(lambda: current_judge().evaluate_agent(trace))
        if result:
            m = result.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Steps", m.steps)
            c2.metric("Tool calls", m.tool_calls)
            c3.metric("Invalid / failed calls", f"{m.invalid_tool_calls} / {m.failed_tool_calls}")
            c4.metric(
                "Expected tools used",
                "n/a" if m.expected_tool_recall is None else f"{m.expected_tool_recall:.0%}",
            )
            if result.checks:
                st.markdown("**Trajectory checks** (deterministic, no LLM)")
                for check in result.checks:
                    where = f"step {check.step}: " if check.step else ""
                    message = f"{where}{check.message}"
                    if check.severity == "error":
                        st.error(message, icon=":material/error:")
                    else:
                        st.warning(message, icon=":material/warning:")
            else:
                st.success("No trajectory problems found.", icon=":material/check_circle:")
            st.markdown("**Judged grade**")
            render_pointwise(result.grade, key="ag")
