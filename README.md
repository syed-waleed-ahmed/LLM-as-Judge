# 🤖 LLM-as-Judge

A **production-grade LLM-as-Judge framework** for evaluating and comparing model
outputs. It uses a [Groq](https://groq.com)-hosted LLM (Llama 3 by default) as an
impartial judge, and ships as a reusable **Python library**, a **command-line
tool**, and a **Streamlit web app** — all sharing the same evaluation core.

[![CI](https://github.com/syed-waleed-ahmed/LLM-as-Judge/actions/workflows/ci.yml/badge.svg)](https://github.com/syed-waleed-ahmed/LLM-as-Judge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

"LLM-as-Judge" is a common pattern for scoring subjective qualities — creativity,
tone, code readability, factual accuracy — where there is no exact-match ground
truth. Doing it *well* in production requires more than a single API call, and
this project handles the parts that usually get skipped:

- **Structured, validated output.** The judge is pinned to a strict JSON schema
  (via the provider's JSON mode) and every response is validated with Pydantic.
- **Position-bias mitigation.** LLM judges tend to favour whichever answer is
  shown first. Optionally, each pair is judged in *both* orders and the scores
  are averaged, cancelling that bias.
- **Self-consistency.** Out-of-range scores are clamped and winners are
  recomputed from the numbers, so the verdict never contradicts the scores.
- **Resilience.** Transient provider failures (rate limits, timeouts, 5xx) are
  retried with exponential backoff; permanent errors fail fast.
- **Scale.** A concurrent batch runner grades thousands of rows from CSV/JSONL,
  isolating per-row failures so one bad row never aborts the run.

---

## Features

| | |
|---|---|
| 🧩 **Library, CLI & UI** | One evaluation core, three ways to use it. |
| 🎯 **Multi-criteria** | Score on several criteria at once (built-in or custom). |
| ⚖️ **Bias mitigation** | Optional A/B order swapping to reduce position bias. |
| 🔁 **Retries & backoff** | Automatic retry of transient errors via `tenacity`. |
| 📦 **Batch mode** | Concurrent CSV/JSONL evaluation for large datasets. |
| 🧪 **Fully tested** | 59 unit tests, ~97% coverage, no network calls in CI. |
| 🔌 **Pluggable providers** | Add a backend by implementing one method. |
| 🛡️ **Typed & linted** | Pydantic models, `mypy`-clean, `ruff`-clean. |

---

## Installation

```bash
git clone https://github.com/syed-waleed-ahmed/LLM-as-Judge.git
cd LLM-as-Judge

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

# Library + UI + dev tooling
pip install -e ".[ui,dev]"
```

Then configure your key:

```bash
cp .env.example .env      # then edit .env and set GROQ_API_KEY
```

Get a free key at <https://console.groq.com/keys>.

---

## Usage

### 1. Web app (Streamlit)

```bash
streamlit run app.py
```

Paste a task description and two outputs, pick criteria in the sidebar,
optionally enable position-bias mitigation, and click **Evaluate**. Download the
full JSON judgment with one click.

### 2. Command line

```bash
# List built-in criteria (no API key required)
llm-judge list-criteria

# Judge a single pair
llm-judge evaluate \
  --task "Write a friendly tweet announcing our new AI tool" \
  --output-a "🚀 Meet our new AI tool! Saves you hours every week. Try it free." \
  --output-b "We have released a new artificial intelligence product." \
  --criteria "Creativity,Brand Tone Adherence" \
  --mitigate-bias

# Read long inputs from files and emit raw JSON
llm-judge evaluate --task-file task.txt \
  --output-a-file a.txt --output-b-file b.txt --json

# Grade a whole dataset concurrently
llm-judge batch --input examples/sample_batch.csv \
  --output results.csv --criteria Helpfulness --workers 8
```

### 3. Python library

```python
from llm_judge import Judge

judge = Judge()  # reads config from environment / .env

result = judge.evaluate(
    task_description="Summarise the text in one sentence.",
    output_a="A concise, accurate one-line summary.",
    output_b="A rambling three-paragraph response.",
    criteria=["Helpfulness", "Conciseness"],
)

print(result.overall_winner)            # Winner.A
print(result.score_for("A"))            # mean score across criteria
print(result.model_dump(mode="json"))   # full structured result
```

---

## Batch input format

CSV (or JSONL with the same keys). An `id` column is optional; the row index is
used if it's missing.

```csv
id,task_description,output_a,output_b
1,Write a tagline,"Fast. Simple. Yours.","Our product is good."
```

The output file mirrors the input `id` and adds the winner, an overall comment,
and per-criterion scores, plus an `error` column for any rows that failed.

---

## Configuration

All settings are read from the environment (or `.env`). See
[`.env.example`](.env.example).

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** for live calls. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq-hosted model to use. |
| `JUDGE_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic). |
| `JUDGE_REQUEST_TIMEOUT` | `60` | Per-request timeout in seconds. |
| `JUDGE_MAX_RETRIES` | `3` | Retries for transient failures. |
| `JUDGE_MITIGATE_POSITION_BIAS` | `false` | Judge each pair in both orders. |
| `JUDGE_BATCH_MAX_WORKERS` | `4` | Concurrency for batch mode. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## Architecture

```text
src/llm_judge/           # Installable package (src layout)
├─ config.py             # Typed settings (pydantic-settings); no import-time crash
├─ models.py             # Criterion / CriterionScore / JudgeResult (validated)
├─ criteria.py           # Built-in criteria presets + resolver
├─ prompts.py            # System/user prompt construction
├─ parsing.py            # Defensive JSON → validated JudgeResult
├─ judge.py              # Core engine: retries, bias mitigation, normalisation
├─ batch.py              # Concurrent CSV/JSONL batch evaluation
├─ cli.py                # `llm-judge` command-line interface
├─ logging_config.py     # Opt-in logging setup
├─ py.typed              # PEP 561 marker: ships type information
└─ providers/            # Pluggable LLM backends
   ├─ base.py            #   LLMProvider protocol
   └─ groq_provider.py   #   Groq implementation (JSON mode + error mapping)
app.py                   # Streamlit UI (thin layer over the library)
tests/                   # 59 offline unit tests (mocked provider)
```

The judge depends only on the `LLMProvider` protocol, so supporting a new backend
(OpenAI, Anthropic, a local model, …) means implementing a single `complete_json`
method — no changes to the judging logic.

---

## Development

```bash
pip install -e ".[ui,dev]"

pytest --cov=llm_judge     # run tests with coverage
ruff check .               # lint
mypy                       # type-check (config in pyproject.toml)
```

Every test uses an in-memory fake provider, so the suite runs fully offline and
deterministically — no API key or network required.

---

## Security

- `.env` is git-ignored; **never commit real keys**.
- If a key is exposed, rotate it immediately in the Groq console.
- The library never logs secret values.

---

## Roadmap

- Additional providers (OpenAI, Anthropic, local models)
- Pointwise (single-output) grading in addition to pairwise
- Confidence intervals via multi-sample judging
- Experiment-tracking integrations (W&B, LangSmith)

Contributions are welcome — please open an issue or PR.

---

## License

[MIT](LICENSE) © Syed Waleed Ahmed
