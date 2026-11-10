"""Settings: what's configured, judging options for this session, limits, storage, privacy."""

from __future__ import annotations

import streamlit as st

from dashboard.common import (
    SESSION_SETTINGS,
    app_settings,
    demo_budgets,
    experiment_store,
    judge_mode,
)
from llm_judge.cache import shared_cache
from llm_judge.experiments import ExperimentStore
from llm_judge.providers import configuration_status, provider_info, provider_names
from llm_judge.providers.registry import build_provider, fallback_names

settings = app_settings()
st.title("Settings")
st.caption(
    "Server-wide settings come from environment variables or Streamlit secrets and are shown "
    "read-only. The options under This session apply to your session only."
)


@st.cache_data(ttl=60, show_spinner=False)
def _health(name: str) -> tuple[bool, str]:
    """A provider's health (no inference). Cached for a minute to spare upstream APIs."""
    try:
        check = getattr(build_provider(app_settings(), name), "health", None)
        if check is None:
            return True, "this provider has no health check"
        health = check()
    except Exception as exc:  # a misconfigured provider is a result here, not a crash
        return False, str(exc)
    return bool(health.ok), str(health.detail or "reachable")


# --- this session ----------------------------------------------------------------------
st.markdown("#### This session")
judge_label = (
    f"live {settings.provider} judge" if judge_mode() == "live" else "mock judge (not an LLM)"
)
st.markdown(f"Judging with the **{judge_label}**. Switch judges in the sidebar.")
left, right = st.columns(2)
left.slider(
    "Judge temperature",
    0.0,
    1.0,
    value=float(settings.temperature),
    step=0.1,
    key="set:temperature",
    help="0 gives the most repeatable judgments. Higher values add sampling variation, "
    "which the Reliability page can measure. The mock judge ignores it.",
)
right.slider(
    "Tie margin",
    0.0,
    0.2,
    value=float(settings.tie_margin),
    step=0.01,
    key="set:tie_margin",
    help="Overall scores closer than this, on the 0-1 scale, count as a tie.",
)
if st.button("Reset to server defaults", key="set:reset", icon=":material/restart_alt:"):
    for name in SESSION_SETTINGS:
        st.session_state.pop(f"set:{name}", None)
    st.rerun()

# --- judges ----------------------------------------------------------------------------
st.markdown("#### Judges")
st.dataframe(
    [
        {
            "provider": name,
            "description": provider_info(name).description,
            "cost": provider_info(name).cost,
            "settings present": "yes" if configuration_status(name, settings)[0] else "no",
            "details": configuration_status(name, settings)[1],
        }
        for name in provider_names()
    ],
    hide_index=True,
)
try:
    fallbacks = fallback_names(settings)
except Exception as exc:  # show a broken fallback setting instead of failing the page
    st.error(f"JUDGE_FALLBACK_PROVIDERS is invalid: {exc}")
    fallbacks = []
st.caption(
    "Settings present means the configuration exists; it doesn't mean the provider is "
    "reachable. Use Check provider health for that."
)
st.caption(
    "Fallback: "
    + (
        " then ".join([settings.provider, *fallbacks]) + " (each fallback is reported on results)"
        if fallbacks
        else "off (set JUDGE_FALLBACK_PROVIDERS to enable)"
    )
)
if st.button("Check provider health", key="set:health", icon=":material/monitor_heart:"):
    st.session_state["set:health_results"] = [
        {"provider": name, "healthy": "yes" if ok else "no", "detail": detail}
        for name in provider_names()
        if configuration_status(name, settings)[0]
        for ok, detail in [_health(name)]
    ]
if st.session_state.get("set:health_results"):
    st.dataframe(st.session_state["set:health_results"], hide_index=True)
    st.caption("Health checks don't run any inference and don't use the demo quota.")

# --- limits ----------------------------------------------------------------------------
st.markdown("#### Limits in effect")
limits: dict[str, object] = {
    "max criteria per evaluation": min(settings.max_criteria, 8)
    if settings.demo_mode
    else settings.max_criteria,
    "max characters per input": (
        settings.demo_max_input_chars if settings.demo_mode else settings.max_input_chars
    ),
    "max characters of judge instructions": (
        min(settings.max_instructions_chars, 1_000)
        if settings.demo_mode
        else settings.max_instructions_chars
    ),
    "max rows per dataset run": settings.demo_max_rows if settings.demo_mode else 200,
    "max ranking candidates": (
        settings.demo_max_candidates if settings.demo_mode else settings.max_candidates
    ),
}
if settings.demo_mode:
    session, daily = demo_budgets(settings)
    st.markdown(
        f"Live judge calls: **{session.remaining} of {session.limit}** left this session, "
        f"and {daily.remaining} of {daily.limit} left today across all visitors."
    )
st.dataframe([{"limit": k, "value": str(v)} for k, v in limits.items()], hide_index=True)
with st.expander("REST API limits (llm-judge serve)"):
    api_limits = {
        "requests per minute per client": settings.api_rate_limit_per_minute,
        "max request body (bytes)": settings.api_max_body_bytes,
        "concurrent evaluations": settings.api_max_concurrent,
        "rows per synchronous batch": settings.api_max_batch_rows,
        "rows per job": settings.api_max_job_rows,
        "queued jobs": settings.api_max_queued_jobs,
        "API keys required": "yes" if settings.api_keys else "no",
    }
    # One text column: mixing numbers and words in a column breaks Arrow serialisation.
    st.dataframe([{"limit": k, "value": str(v)} for k, v in api_limits.items()], hide_index=True)

# --- storage and cache -------------------------------------------------------------------
st.markdown("#### Storage and cache")
store = experiment_store()
stats = store.statistics()
location = (
    "this browser session only (cleared when it ends)"
    if settings.demo_mode
    else f"`{settings.db_path}`"
)
st.markdown(
    f"Recorded runs are kept in {location}: {stats['experiments']} experiment(s), "
    f"{stats['runs']} run(s), {stats['rows_evaluated']} row(s) evaluated."
)
if settings.cache_size:
    cache = shared_cache(settings.cache_size)
    cache_stats = cache.stats()
    st.markdown(
        f"Result cache: on, {cache_stats['size']} of {cache_stats['maxsize']} results, "
        f"{cache_stats['hits']} hit(s) and {cache_stats['misses']} miss(es). It's used only "
        "at temperature 0 and lives in memory."
    )
    if not settings.demo_mode and st.button("Clear the result cache", key="set:clear-cache"):
        cache.clear()
        st.rerun()
else:
    st.markdown("Result cache: off (set `JUDGE_CACHE_SIZE` to reuse identical judgments).")

# --- privacy ---------------------------------------------------------------------------
st.markdown("#### Privacy")
if settings.demo_mode:
    st.markdown(
        "This is a public demo. What you type is sent to the selected judge to be evaluated "
        "and isn't stored by this app beyond your session. Don't paste confidential data "
        "into a public demo."
    )
    if st.button("Delete my session's runs", key="set:forget", icon=":material/delete:"):
        st.session_state.demo_store = ExperimentStore(":memory:")
        st.session_state.evaluations = 0
        st.success("Your session's runs were deleted.")
else:
    st.markdown(
        "Raw task text, outputs and references are stored only as hashes, although judge "
        "explanations can quote them. Delete individual runs on the Experiments page."
    )
