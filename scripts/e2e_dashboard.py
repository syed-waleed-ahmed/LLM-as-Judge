"""Real-browser end-to-end test of the dashboard in public-demo mode.

Drives every page with Playwright through an installed Microsoft Edge (no browser
download), using both the live judge and the offline mock, and checks for Streamlit
exceptions and browser console errors at every step. It also exercises the
public-demo quota until it is exhausted.

Prerequisites:
  pip install playwright
  # .streamlit/secrets.toml (or environment) with a working GROQ_API_KEY and:
  #   JUDGE_DEMO_MODE = "true"
  #   JUDGE_DEMO_SESSION_CALLS = "6"
  streamlit run app.py --server.port 8600 --server.headless true

Run:
  python scripts/e2e_dashboard.py [screenshot-dir]

It makes about eight live judge calls. Set E2E_BASE_URL to target another server.
"""

import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, expect, sync_playwright

BASE = os.environ.get("E2E_BASE_URL", "http://localhost:8600")
SHOTS = Path(sys.argv[1] if len(sys.argv) > 1 else "e2e-screenshots")
SHOTS.mkdir(parents=True, exist_ok=True)
results: list[tuple[str, bool, str]] = []
console_errors: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""), flush=True)


def no_exceptions(page: Page, where: str) -> None:
    count = page.locator('[data-testid="stException"]').count()
    check(f"{where}: no Streamlit exception", count == 0, f"{count} exception block(s)")


def idle(page: Page, timeout: float = 90) -> None:
    """Wait until Streamlit has finished running the script."""
    page.wait_for_timeout(400)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if page.locator('[data-testid="stStatusWidget"]').filter(has_text="Running").count() == 0:
            page.wait_for_timeout(300)
            if (
                page.locator('[data-testid="stStatusWidget"]').filter(has_text="Running").count()
                == 0
            ):
                return
        page.wait_for_timeout(300)


def nav(page: Page, label: str) -> None:
    # Match the end of the link text: older Streamlit versions include the icon's
    # ligature text in it (for example "comparePlayground"). No page label is a
    # suffix of another.
    ends_with_label = re.compile(rf"{re.escape(label)}\s*$")
    page.locator('[data-testid="stSidebarNav"] a').filter(has_text=ends_with_label).first.click()
    idle(page)


def shot(page: Page, name: str) -> None:
    page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)


def text_visible(page: Page, pattern: str, timeout: float = 90_000) -> bool:
    try:
        expect(page.get_by_text(re.compile(pattern)).first).to_be_visible(timeout=timeout)
        return True
    except AssertionError:
        return False


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.goto(BASE)
    page.wait_for_selector("h1", timeout=60_000)
    idle(page)

    # --- Overview ----------------------------------------------------------------------
    check("overview: title", text_visible(page, r"^LLM-as-Judge$"))
    check("overview: live judge offered", text_visible(page, r"Live LLM \(groq\)"))
    check("overview: live judging banner", text_visible(page, r"Live judging with"))
    check("sidebar: demo quota shown", text_visible(page, r"6/6 live judge calls left"))
    no_exceptions(page, "overview")
    shot(page, "01_overview_live")

    # --- Playground: live pairwise -----------------------------------------------------
    nav(page, "Playground")
    page.get_by_text("Summary", exact=True).first.click()
    idle(page)
    task_value = page.get_by_label("Task").first.input_value()
    check("playground: example loads", "Summarise" in task_value, task_value[:40])
    page.get_by_role("button", name="Compare", exact=True).click()
    idle(page, 120)
    check("playground: live verdict", text_visible(page, r"preferred overall|tie overall", 120_000))
    check("playground: live badge", text_visible(page, r"groq · openai/gpt-oss-120b"))
    check("sidebar: quota decremented", text_visible(page, r"5/6 live judge calls left"))
    no_exceptions(page, "playground live")
    shot(page, "02_playground_live")

    # --- Switch to mock ----------------------------------------------------------------
    page.get_by_text("Mock (offline, not an LLM)").click()
    idle(page)
    page.get_by_role("button", name="Compare", exact=True).click()
    idle(page)
    check("playground: mock badge", text_visible(page, r"Mock judge \(not an LLM\)"))
    no_exceptions(page, "playground mock")

    page.get_by_role("tab", name="Check against a reference").click()
    page.get_by_text("Factual QA", exact=True).first.click()
    idle(page)
    page.get_by_role("button", name="Check", exact=True).click()
    idle(page)
    check("playground: reference grading", text_visible(page, r"Reference grading"))
    no_exceptions(page, "playground reference")

    page.get_by_role("tab", name="Rank candidates").click()
    page.get_by_role("button", name="Rank", exact=True).click()
    idle(page)
    check("playground: ranking", text_visible(page, r"Leaderboard: '"))
    no_exceptions(page, "playground rank")

    page.get_by_role("tab", name="Grade one output").click()
    page.get_by_label("Task").nth(1).fill("Explain DNS in one sentence.")
    page.get_by_label("Output", exact=True).first.fill(
        "DNS translates domain names into IP addresses."
    )
    idle(page)
    page.get_by_role("button", name="Grade", exact=True).click()
    idle(page)
    check("playground: pointwise", text_visible(page, r"Pointwise grading"))
    no_exceptions(page, "playground pointwise")
    shot(page, "03_playground_mock")

    # --- Datasets ----------------------------------------------------------------------
    nav(page, "Datasets")
    check("datasets: validation summary", text_visible(page, r"Validation: \d+/\d+ valid rows"))
    page.get_by_role("button", name=re.compile(r"^Evaluate \d+ row")).click()
    idle(page)
    check("datasets: run summary", text_visible(page, r"Rows evaluated"))
    download = page.get_by_text("Download CSV", exact=True).first
    check("datasets: CSV download offered", download.is_visible())
    with page.expect_download() as info:
        download.click()
    check("datasets: CSV downloads", info.value.suggested_filename == "results.csv")
    idle(page)
    check("datasets: results survive the download", text_visible(page, r"Rows evaluated", 5_000))
    combo = page.locator('[data-testid="stSelectbox"]').get_by_role("combobox").first
    combo.click()
    combo.press("Control+a")
    combo.type("summarization")
    combo.press("Enter")
    idle(page)
    check("datasets: criteria follow the sample", text_visible(page, r"^Factual Accuracy$", 10_000))
    no_exceptions(page, "datasets")
    shot(page, "04_datasets")

    # --- Benchmarks (twice, for comparisons) -------------------------------------------
    nav(page, "Benchmarks")
    for _ in range(2):
        page.get_by_role("button", name="Run benchmark").click()
        idle(page)
    check("benchmarks: mock warning", text_visible(page, r"mock heuristic, not an LLM"))
    check("benchmarks: per-model stats", text_visible(page, r"By model"))
    no_exceptions(page, "benchmarks")
    shot(page, "05_benchmarks")

    # --- Experiments -------------------------------------------------------------------
    nav(page, "Experiments")
    check(
        "experiments: verdict",
        text_visible(page, r"No significant change|Regression detected|Improvement detected"),
    )
    check(
        "experiments: session-only notice",
        text_visible(page, r"runs live only in your browser session"),
    )
    no_exceptions(page, "experiments")
    shot(page, "06_experiments")

    # --- RAG ---------------------------------------------------------------------------
    nav(page, "RAG evaluation")
    page.get_by_role("button", name="Evaluate", exact=True).click()
    idle(page)
    check("rag: metrics", text_visible(page, r"Context precision"))
    check("rag: unsupported helmet claim", text_visible(page, r"free helmet"))
    no_exceptions(page, "rag")
    shot(page, "07_rag")

    # --- Agents ------------------------------------------------------------------------
    nav(page, "Agent evaluation")
    page.get_by_role("button", name="Evaluate", exact=True).click()
    idle(page)
    check("agents: deterministic check", text_visible(page, r"missing required argument"))
    check("agents: judged grade", text_visible(page, r"Pointwise grading"))
    trace = SHOTS / "clean_trace.json"
    trace.write_text(
        '{"task": "Weather in Paris?", "tools": [{"name": "weather", "required_args": ["city"]}],'
        ' "steps": [{"type": "tool_call", "tool": "weather", "arguments": {"city": "Paris"}},'
        ' {"type": "tool_result", "tool": "weather", "content": "Sunny"}],'
        ' "final_answer": "Sunny in Paris."}',
        encoding="utf-8",
    )
    page.locator('[data-testid="stFileUploader"] input[type="file"]').set_input_files(str(trace))
    idle(page)
    page.get_by_role("button", name="Evaluate", exact=True).click()
    idle(page)
    check(
        "agents: uploaded trace is evaluated",
        text_visible(page, r"No trajectory problems found", 30_000),
    )
    no_exceptions(page, "agents")
    shot(page, "08_agents")

    # --- Reliability -------------------------------------------------------------------
    nav(page, "Reliability")
    page.get_by_role("button", name=re.compile(r"^Judge \d+ times$")).click()
    idle(page)
    check("reliability: consistency", text_visible(page, r"Agreement with modal verdict"))
    page.get_by_role("tab", name="Judge agreement").click()
    idle(page)
    check("reliability: agreement", text_visible(page, r"Matched rows"))
    no_exceptions(page, "reliability")
    shot(page, "09_reliability")

    # --- About -------------------------------------------------------------------------
    nav(page, "About")
    check("about: limits section", text_visible(page, r"^Limits in effect$"))
    check("about: public-demo privacy note", text_visible(page, r"This is a public demo"))
    no_exceptions(page, "about")

    # --- Quota exhaustion on the live judge --------------------------------------------
    nav(page, "Playground")
    page.get_by_text("Live LLM (groq)").click()
    idle(page)
    page.get_by_role("tab", name="Compare two outputs").click()
    page.get_by_text("Summary", exact=True).first.click()
    idle(page)
    exhausted = False
    attempts = 0
    for _ in range(7):
        attempts += 1
        page.get_by_role("button", name="Compare", exact=True).click()
        idle(page, 120)
        if text_visible(page, r"per-session demo budget of 6", timeout=1_500):
            exhausted = True
            break
    check("quota: session budget enforced", exhausted, f"after {attempts} more compare(s)")
    check("quota: offers mock instead", text_visible(page, r"Pick 'Mock' in the sidebar"))
    no_exceptions(page, "quota")
    shot(page, "10_quota")
    browser.close()

serious = [e for e in console_errors if "favicon" not in e.lower()]
check("browser console: no errors", not serious, "; ".join(serious[:3]))
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
