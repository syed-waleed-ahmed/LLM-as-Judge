"""Streamlit dashboard for the LLM-as-Judge platform.

Run with::

    streamlit run app.py

All evaluation logic lives in the :mod:`llm_judge` package. The dashboard is a
thin layer over it, so the UI, CLI, REST API and Python callers behave the same.
Set ``JUDGE_DEMO_MODE=true`` for a public deployment (quotas, input caps,
session-only storage); see DEPLOYMENT.md.
"""

from __future__ import annotations

import streamlit as st

from dashboard.common import render_sidebar, render_sidebar_status

st.set_page_config(page_title="LLM-as-Judge", page_icon=":material/balance:", layout="wide")

VIEWS = "dashboard/views"
navigation = st.navigation(
    {
        "Home": [
            st.Page(
                f"{VIEWS}/overview.py", title="Overview", icon=":material/dashboard:", default=True
            ),
            st.Page(f"{VIEWS}/about.py", title="About", icon=":material/info:"),
        ],
        "Evaluate": [
            st.Page(f"{VIEWS}/playground.py", title="Playground", icon=":material/compare:"),
            st.Page(f"{VIEWS}/datasets.py", title="Datasets", icon=":material/table:"),
            st.Page(f"{VIEWS}/benchmarks.py", title="Benchmarks", icon=":material/speed:"),
            st.Page(f"{VIEWS}/rag.py", title="RAG evaluation", icon=":material/library_books:"),
            st.Page(f"{VIEWS}/agents.py", title="Agent evaluation", icon=":material/smart_toy:"),
        ],
        "Analyze": [
            st.Page(f"{VIEWS}/experiments.py", title="Experiments", icon=":material/history:"),
            st.Page(f"{VIEWS}/reliability.py", title="Reliability", icon=":material/verified:"),
        ],
    }
)
render_sidebar()
navigation.run()
render_sidebar_status()  # after the page, so quota counts are current
