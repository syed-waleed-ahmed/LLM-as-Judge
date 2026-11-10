"""About: what this is, how results stay honest, privacy, and how to run it yourself."""

from __future__ import annotations

import streamlit as st

from dashboard.common import app_settings
from llm_judge import __version__

settings = app_settings()
st.title("About")
st.markdown(
    f"**LLM-as-Judge {__version__}**: an open-source evaluation platform that runs on free "
    "tiers and local models. The same engine powers this dashboard, a Python library, a "
    "command-line tool and a REST API. "
    "[Documentation](https://syed-waleed-ahmed.github.io/LLM-as-Judge/) · "
    "[Source on GitHub](https://github.com/syed-waleed-ahmed/LLM-as-Judge) · "
    "[Report an issue](https://github.com/syed-waleed-ahmed/LLM-as-Judge/issues/new/choose)"
)

st.markdown("#### How the results stay honest")
st.markdown(
    "- Every result names the judge that produced it. Results from the mock judge, a "
    "deterministic text heuristic, are labelled as not coming from an LLM.\n"
    "- Replies must match a strict schema and the exact criteria requested, or they're "
    "repaired or rejected. Winners are recomputed from the scores.\n"
    "- Reliability warnings flag possible prompt injection, verbosity bias, self-preference "
    "and order sensitivity on each result.\n"
    "- Analytics and reports only restate recorded results; no number is estimated by an LLM."
)
st.caption("The judges, limits and session options in effect are on the Settings page.")

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
    "git clone https://github.com/syed-waleed-ahmed/LLM-as-Judge.git\n"
    "cd LLM-as-Judge\n"
    'pip install -e ".[all]"\n'
    "streamlit run app.py            # this dashboard\n"
    "llm-judge --provider mock benchmark general_qa\n"
    "llm-judge serve                 # REST API at http://127.0.0.1:8000/docs",
    language="bash",
)
