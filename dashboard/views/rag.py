"""RAG evaluation: faithfulness, answer relevance, and context precision/recall."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    app_settings,
    current_judge,
    render_judge_badge,
    render_warnings,
    run_safely,
)
from llm_judge.rag import MAX_CONTEXTS

settings = app_settings()
max_chars = settings.demo_max_input_chars if settings.demo_mode else None

st.title("RAG evaluation")
st.caption(
    "Checks a retrieval-augmented answer against the passages that were retrieved for it. "
    "The judge returns claim-level verdicts; the metrics are computed from them."
)

question = st.text_input(
    "Question",
    value="What is the range of the Aurora 3 e-bike, and how long does it take to charge?",
    key="rag:question",
)
st.caption(f"Retrieved contexts, in rank order (up to {MAX_CONTEXTS}).")
table = st.data_editor(
    [
        {"passage": "The Aurora 3 e-bike has a 500 Wh battery and a range of up to 80 km."},
        {"passage": "Our showroom is open Monday to Saturday from 10:00 to 18:00."},
        {"passage": "A full charge of the Aurora 3 battery takes about 5 hours."},
    ],
    num_rows="dynamic",
    key="rag:contexts",
    width="stretch",
)
answer = st.text_area(
    "Generated answer",
    value=(
        "The Aurora 3 has a range of up to 80 km and charges fully in about 5 hours. "
        "It also comes with a free helmet."
    ),
    key="rag:answer",
    max_chars=max_chars,
)
reference = st.text_area(
    "Reference answer (optional, enables context recall)",
    value="Up to 80 km; a full charge takes about 5 hours.",
    key="rag:reference",
    max_chars=max_chars,
)

if st.button("Evaluate", type="primary", key="rag:run"):
    contexts = [str(row.get("passage") or "").strip() for row in table]
    contexts = [c for c in contexts if c]
    if not question.strip() or not answer.strip() or not contexts:
        st.error("Give a question, at least one context and an answer.")
    else:
        with st.spinner("Judging..."):
            result = run_safely(
                lambda: current_judge().evaluate_rag(
                    question, contexts, answer, reference=reference.strip() or None
                )
            )
        if result:
            render_judge_badge(result.metadata)

            def pct(value: float | None) -> str:
                return "n/a" if value is None else f"{value:.0%}"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness", pct(result.faithfulness), help="Claims supported by contexts")
            c2.metric("Answer relevance", pct(result.answer_relevance))
            c3.metric(
                "Context precision",
                pct(result.context_precision),
                help="Average precision of relevant contexts in rank order",
            )
            c4.metric(
                "Context recall",
                pct(result.context_recall),
                help="Reference statements covered by the contexts (needs a reference)",
            )
            render_warnings(result.warnings)
            st.markdown("**Claims in the answer**")
            st.dataframe(
                [
                    {
                        "claim": c.claim,
                        "supported": "yes" if c.supported else "no",
                        "supporting contexts": ", ".join(map(str, c.context_ids)) or "-",
                    }
                    for c in result.claims
                ],
                hide_index=True,
            )
            st.markdown("**Retrieved contexts**")
            st.dataframe(
                [
                    {
                        "rank": v.context_id,
                        "relevant": "yes" if v.relevant else "no",
                        "reason": v.reason,
                    }
                    for v in result.contexts
                ],
                hide_index=True,
            )
            if result.answer_relevance_explanation:
                st.caption(f"Answer relevance: {result.answer_relevance_explanation}")
