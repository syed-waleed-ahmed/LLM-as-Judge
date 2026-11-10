"""About: how this works, limits in effect, privacy, and how to run it yourself."""

from __future__ import annotations

import streamlit as st

from dashboard.common import app_settings
from llm_judge import __version__
from llm_judge.providers import configuration_status, provider_info, provider_names

settings = app_settings()
st.title("About")
st.markdown(
    f"**LLM-as-Judge {__version__}**: an open-source evaluation platform that runs on free "
    "tiers and local models. The same engine powers this dashboard, a Python library, a "
    "command-line tool and a REST API. "
    "[Source on GitHub](https://github.com/syed-waleed-ahmed/LLM-as-Judge)."
)

st.markdown("#### Judges")
st.dataframe(
    [
        {
            "provider": name,
            "description": provider_info(name).description,
            "cost": provider_info(name).cost,
            "configured here": "yes" if configuration_status(name, settings)[0] else "no",
        }
        for name in provider_names()
    ],
    hide_index=True,
)
st.caption(
    "The mock judge is a deterministic text heuristic so the app works with no API key. "
    "Its scores are for demonstration only, and every mock result is labelled."
)

st.markdown("#### Limits in effect")
limits = {
    "max criteria per evaluation": settings.max_criteria,
    "max characters per input": (
        settings.demo_max_input_chars if settings.demo_mode else settings.max_input_chars
    ),
    "max rows per dataset run": settings.demo_max_rows if settings.demo_mode else 200,
    "max ranking candidates": (
        settings.demo_max_candidates if settings.demo_mode else settings.max_candidates
    ),
}
if settings.demo_mode:
    limits["live judge calls per session"] = settings.demo_session_calls
    limits["live judge calls per day (all visitors)"] = settings.demo_daily_calls
st.dataframe([{"limit": k, "value": v} for k, v in limits.items()], hide_index=True)

st.markdown("#### Privacy")
if settings.demo_mode:
    st.markdown(
        "This is a public demo. What you type is sent to the selected judge to be evaluated, "
        "and isn't stored by this app: runs stay in your browser session only. Don't paste "
        "confidential data into a public demo."
    )
else:
    st.markdown(
        f"Recorded runs are saved locally to `{settings.db_path}`. Raw inputs are stored only "
        "as hashes, although judge explanations can quote them."
    )

st.markdown("#### Run it yourself")
st.code(
    'pip install "llm-as-judge[all]"\n'
    "streamlit run app.py            # this dashboard\n"
    "llm-judge --provider mock benchmark general_qa\n"
    "llm-judge serve                 # REST API at http://127.0.0.1:8000/docs",
    language="bash",
)
