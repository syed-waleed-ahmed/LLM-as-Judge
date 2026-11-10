# LLM-as-Judge

A **production-grade LLM-as-Judge framework** for evaluating and comparing model
outputs. It uses a [Groq](https://groq.com)-hosted LLM (`openai/gpt-oss-120b` by default) as an
impartial judge, and ships as a reusable **Python library**, a **command-line
tool**, and a **Streamlit web app** — all sharing the same evaluation core.

[![CI](https://github.com/syed-waleed-ahmed/LLM-as-Judge/actions/workflows/ci.yml/badge.svg)](https://github.com/syed-waleed-ahmed/LLM-as-Judge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Batch input format](#batch-input-format)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development](#development)
- [Security](#security)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why this exists

"LLM-as-Judge" is a common pattern for scoring subjective qualities — creativity,
tone, code readability, factual accuracy — where there is no exact-match ground
truth. Doing it *well* in production requires more than a single API call, and
this project handles the parts that usually get skipped:

- **Structured, validated output.** The judge is pinned to a strict JSON schema
  (via the provider's JSON mode), and every response is validated with Pydantic.
  The returned criteria must match the requested ones exactly. A malformed
  reply gets a bounded "repair" retry that tells the judge what was wrong.
- **Position-bias mitigation.** LLM judges tend to favour whichever answer is
  shown first. Optionally, each pair is judged in *both* orders (concurrently)
  and the scores are averaged, cancelling that bias. The explanations shown
  always agree with the averaged verdict.
- **Self-consistency.** Out-of-range scores are clamped (and flagged), and
  winners are recomputed from the numbers, so the verdict never contradicts the
  scores. Criteria on different scales are normalised before they're combined.
- **Injection resistance.** Candidate outputs are treated as untrusted data,
  wrapped in boundary markers that carry a random per-request token.
- **Bias diagnostics.** Results carry warnings for suspected prompt injection,
  verbosity bias, self-preference risk and order-dependent verdicts.
- **Resilience.** Transient provider failures (rate limits, timeouts, 5xx) are
  retried with exponential backoff; permanent errors fail fast.
- **Scale.** A concurrent batch runner grades thousands of rows from CSV/JSONL,
  isolating per-row failures so one bad row never aborts the run.

---

## Features

| Capability | Description |
|---|---|
| **Library, CLI & UI** | One evaluation core, three ways to use it. |
| **Four evaluation modes** | Pairwise (A vs B), pointwise (one output), reference-based (vs. an expected answer), and multi-candidate ranking. |
| **Multi-criteria** | Score on several criteria at once (built-in or custom), with optional weights. |
| **Bias mitigation** | Optional A/B order swapping to reduce position bias. |
| **Reliability warnings** | Flags prompt injection, verbosity bias, self-preference and order sensitivity. |
| **Retries & repair** | Backoff for transient errors; repair retries for malformed replies. |
| **Batch mode** | Concurrent CSV/JSONL evaluation for large datasets. |
| **Fully tested** | Offline unit and property-based tests (Hypothesis), no network calls in CI. |
| **Pluggable providers** | Groq, Ollama (local), any OpenAI-compatible API, and an offline mock; add your own by implementing one method. |
| **Typed & linted** | Pydantic models, `mypy`-clean, `ruff`-clean and `ruff format`-clean. |

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

Get a free key at <https://console.groq.com/keys>. No key yet? Every
command works offline with `--provider mock`, a deterministic heuristic judge
(clearly labelled, not an LLM), or locally with `--provider ollama`.

---

## Usage

### 1. Web app (Streamlit)

```bash
streamlit run app.py
```

Paste a task description and two outputs, then pick criteria (and optionally
their weights) in the sidebar. You can also enable position-bias mitigation.
Click **Evaluate**, and download the full JSON judgment with one click.

### 2. Command line

```bash
# List built-in criteria (no API key required)
llm-judge list-criteria

# Judge a single pair
llm-judge evaluate \
  --task "Write a friendly tweet announcing our new AI tool" \
  --output-a "Meet our new AI assistant - it saves you hours every week. Try it free!" \
  --output-b "We have released a new artificial intelligence product." \
  --criteria "Creativity,Brand Tone Adherence" \
  --mitigate-bias

# Weight criteria and record which models produced each output
llm-judge evaluate --task-file task.txt \
  --output-a-file a.txt --output-b-file b.txt \
  --criteria "Helpfulness,Conciseness" --weights "Helpfulness=2" \
  --model-a gpt-4o --model-b llama-3.1-8b-instant

# Read long inputs from files and emit raw JSON
llm-judge evaluate --task-file task.txt \
  --output-a-file a.txt --output-b-file b.txt --json

# Grade a whole dataset concurrently
llm-judge batch --input examples/sample_batch.csv \
  --output results.csv --criteria Helpfulness --workers 8

# Grade one output on its own, or against an expected answer
llm-judge grade --task "Explain DNS" --output-file answer.txt
llm-judge grade --task "When did Apollo 11 land?" \
  --output "In 1969." --reference "Apollo 11 landed on the Moon on 20 July 1969."

# Rank several candidates (round robin, or --strategy pointwise for fewer calls)
llm-judge rank --task "Write a tagline" --candidates-file candidates.json
```

`candidates.json` is either `{"id": "text", ...}` or a list of
`{"id": ..., "output": ..., "model": ...}` objects.

### 3. Python library

```python
from llm_judge import Criterion, Judge, resolve_criterion

judge = Judge()  # reads config from environment / .env

result = judge.evaluate(
    task_description="Summarise the text in one sentence.",
    output_a="A concise, accurate one-line summary.",
    output_b="A rambling three-paragraph response.",
    criteria=["Helpfulness", "Conciseness"],
    candidate_a_model="gpt-4o",  # optional; recorded in metadata
)

print(result.overall_winner)  # Winner.A
print(result.overall_score_a)  # weighted, scale-normalised score (0-1)
for warning in result.warnings:  # reliability signals worth a second look
    print(warning.code, warning.message)

# Custom criteria can set their own scale and weight.
coverage = Criterion(
    name="Coverage", description="Covers every key point.", scale_min=0, scale_max=100, weight=2
)
judge.evaluate("...", "...", "...", [resolve_criterion("Helpfulness"), coverage])
```

The same `Judge` also grades single outputs and ranks candidates:

```python
# Pointwise: grade one output on its own merits.
grade = judge.evaluate_pointwise("Explain DNS.", answer, ["Helpfulness", "Conciseness"])
print(grade.overall_score, grade.feedback)

# Reference-based: compare with an expected answer (defaults to Correctness + Completeness).
check = judge.evaluate_reference("When did Apollo 11 land?", "In 1968.", reference="20 July 1969.")
print(check.unsupported_claims)  # hallucination indicators
print(check.missing_information)  # key points the output omits
print(check.reference_metrics)  # exact_match and token_f1 (no LLM involved)

# Ranking: round robin (every pair head to head) or pointwise (cheaper).
ranking = judge.rank("Write a tagline", {"v1": "...", "v2": "...", "v3": "..."})
for row in ranking.leaderboard:
    print(row.rank, row.candidate, row.score, (row.wins, row.losses, row.ties))
```

Every pairwise evaluation returns a validated, versioned result:

```json
{
  "schema_version": "1.2",
  "criteria": [
    {
      "criterion": "Helpfulness",
      "output_a_score": 8.0,
      "output_b_score": 3.0,
      "output_a_explanation": "Directly and accurately answers the task.",
      "output_b_explanation": "Vague and largely off-topic.",
      "winner": "A"
    }
  ],
  "overall_winner": "A",
  "overall_comment": "Output A is clearly more helpful.",
  "overall_score_a": 0.7778,
  "overall_score_b": 0.2222,
  "warnings": [],
  "metadata": {
    "model": "openai/gpt-oss-120b",
    "judge": { "provider": "groq", "model": "openai/gpt-oss-120b" },
    "candidates": { "A": { "model": "gpt-4o" }, "B": { "model": null } },
    "prompt_version": "2.0",
    "position_bias_mitigated": false,
    "latency_seconds": 0.93
  }
}
```

---

## Reliability safeguards

| Safeguard | What it does |
|---|---|
| Neutral schema | The prompt shows the output shape with placeholders, never example scores or winners that could anchor the judge. |
| Untrusted-content boundaries | The task and outputs sit between markers that carry a random per-request token. The judge is told to treat them as data and to ignore embedded instructions. |
| Exact criteria matching | The reply must contain every requested criterion exactly once (case-insensitive). Missing, extra or duplicate criteria are rejected, never guessed. |
| Repair retries | A reply that is malformed or doesn't match the schema is sent back with the reason it was rejected, up to `JUDGE_MAX_PARSE_RETRIES` times. |
| Brace-aware recovery | JSON embedded in stray prose is recovered with a real JSON decoder, not string slicing. |
| Scale normalisation and weights | Each score is mapped to 0-1 on its own scale before the weighted overall score is computed. |
| Consistent bias merges | With mitigation on, explanations come from the ordering that agrees with the averaged verdict. When neither ordering agrees, both are kept and labelled. |

Results also carry **warnings** (they never change scores):

| Code | Meaning |
|---|---|
| `possible_prompt_injection` | An output contains text aimed at manipulating the judge. |
| `possible_verbosity_bias` | The winner is much longer than the other output (`JUDGE_VERBOSITY_RATIO`). |
| `self_preference_risk` | A candidate model is from the judge's own model family. |
| `position_inconsistent` | The verdict changed when the outputs were swapped. |
| `score_out_of_range` | The judge scored outside the scale; the score was clamped. |
| `input_truncated` | An input exceeded `JUDGE_MAX_INPUT_CHARS` and was truncated. |

These signals are heuristics. A clean result doesn't prove the absence of bias
or manipulation, and a warning doesn't prove its presence.

---

## Batch input format

CSV (or JSONL with the same keys). An `id` column is optional; the row index is
used if it's missing.

```csv
id,task_description,output_a,output_b
1,Write a tagline,"Fast. Simple. Yours.","Our product is good."
```

Optional `model_a` and `model_b` columns name the models behind each output.

The output file mirrors the input `id`. For each row it adds the winner, an
overall comment, the weighted overall scores, per-criterion scores and warning
codes, plus an `error` column for any row that failed.

---

## Configuration

All settings are read from the environment (or `.env`). See
[`.env.example`](.env.example).

| Variable | Default | Description |
|---|---|---|
| `JUDGE_PROVIDER` | `groq` | `groq`, `ollama`, `openai_compatible` or `mock`. |
| `GROQ_API_KEY` | — | Required for the `groq` provider. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq-hosted model to use. |
| `JUDGE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server. |
| `JUDGE_OLLAMA_MODEL` | `llama3.2` | Local model to judge with. |
| `JUDGE_OPENAI_BASE_URL` | — | Base URL of an OpenAI-compatible API (e.g. `.../v1`). |
| `JUDGE_OPENAI_MODEL` | — | Model name on that endpoint. |
| `JUDGE_OPENAI_API_KEY` | — | Optional bearer token for that endpoint. |
| `JUDGE_OPENAI_JSON_MODE` | `true` | Send `response_format: json_object` (disable for servers that reject it). |
| `JUDGE_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic). |
| `JUDGE_REQUEST_TIMEOUT` | `60` | Per-request timeout in seconds. |
| `JUDGE_MAX_RETRIES` | `3` | Retries for transient failures. |
| `JUDGE_MAX_PARSE_RETRIES` | `1` | Repair retries for malformed or schema-violating replies. |
| `JUDGE_MITIGATE_POSITION_BIAS` | `false` | Judge each pair in both orders. |
| `JUDGE_PARALLEL_BIAS_RUNS` | `true` | Run the two orderings concurrently. |
| `JUDGE_TIE_MARGIN` | `0.06` | Normalised (0-1) score gap treated as a tie (just over half a point on a 1-10 scale). |
| `JUDGE_VERBOSITY_RATIO` | `2.0` | Length ratio at which a long winner is flagged. |
| `JUDGE_MAX_CRITERIA` | `20` | Maximum criteria per evaluation. |
| `JUDGE_MAX_CANDIDATES` | `8` | Maximum candidates per ranking (round robin makes n(n-1)/2 comparisons). |
| `JUDGE_MAX_INPUT_CHARS` | `50000` | Longer inputs are truncated (with a warning). |
| `JUDGE_BATCH_MAX_WORKERS` | `4` | Concurrency for batch mode (doubled when parallel bias runs are on). |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## Providers

The judge runs on any of these backends. Pick one with `JUDGE_PROVIDER` or the
`--provider` flag. Nothing here requires a paid service.

| Provider | API key | Runs | Notes |
|---|---|---|---|
| `groq` (default) | yes ([free](https://console.groq.com/keys)) | cloud | Free plan, rate-limited per model. Default judge `openai/gpt-oss-120b`. |
| `ollama` | no | your machine | Free local models via [Ollama](https://ollama.com): `ollama pull llama3.2`. |
| `openai_compatible` | optional | either | Any `/chat/completions` server: LM Studio, vLLM, llama.cpp, or hosted free models. |
| `mock` | no | offline | Deterministic heuristic, **not an LLM**. For demos, tests and trying the tool without a key. |

```bash
llm-judge providers                      # what's available and configured
llm-judge --provider ollama health       # reachability check, no inference
llm-judge --provider mock evaluate --task "..." --output-a "..." --output-b "..."
```

```python
from llm_judge import Judge
from llm_judge.providers import OllamaProvider, OpenAICompatibleProvider

judge = Judge(provider=OllamaProvider("qwen2.5:7b"))
judge = Judge(provider=OpenAICompatibleProvider("http://localhost:1234/v1", "local-model"))
```

Every result records which provider and model produced it (`metadata.judge`),
and the library never switches provider or model silently. Mock results say so
in every explanation. Third-party packages can add providers through the
`llm_judge.providers` entry-point group (see `llm_judge/providers/registry.py`).

Small local models are weaker judges than large hosted ones. Expect noisier
scores and more repair retries, and prefer a judge from a different model
family than the candidates (the `self_preference_risk` warning flags this).

---

## Architecture

```text
src/llm_judge/           # Installable package (src layout)
├─ config.py             # Typed settings (pydantic-settings); no import-time crash
├─ models.py             # Criterion / CriterionScore / JudgeResult (validated)
├─ criteria.py           # Built-in criteria presets + resolver
├─ prompts.py            # Versioned, injection-resistant prompt construction
├─ parsing.py            # Defensive JSON → validated JudgeResult
├─ scoring.py            # Pure helpers: criteria matching, normalisation, winners
├─ diagnostics.py        # Injection, verbosity and self-preference signals
├─ judge.py              # Core engine: retries, repair, bias mitigation, merging
├─ pointwise.py          # Single-output and reference-based grading
├─ ranking.py            # Multi-candidate round-robin / pointwise leaderboards
├─ metrics.py            # Deterministic reference metrics (exact match, token F1)
├─ batch.py              # Concurrent CSV/JSONL batch evaluation
├─ cli.py                # `llm-judge` command-line interface
├─ logging_config.py     # Opt-in logging setup
├─ py.typed              # PEP 561 marker: ships type information
└─ providers/            # Pluggable LLM backends
   ├─ base.py            #   LLMProvider protocol, health, usage counters
   ├─ registry.py        #   Name -> provider factory, plugin discovery
   ├─ groq_provider.py   #   Groq (default cloud judge)
   ├─ ollama_provider.py #   Local models via Ollama
   ├─ openai_compatible.py # Any /chat/completions endpoint
   └─ mock_provider.py   #   Offline heuristic judge (not an LLM)
app.py                   # Streamlit UI (thin layer over the library)
tests/                   # Offline unit + property tests (fake providers)
```

The judge depends only on the `LLMProvider` protocol, so supporting a new backend
means implementing a single `complete_json` method, with no changes to the
judging logic.

---

## Development

```bash
pip install -e ".[ui,dev]"

pytest --cov=llm_judge     # run tests with coverage
ruff check .               # lint
ruff format --check .      # formatting
mypy                       # type-check (config in pyproject.toml)
```

Every test uses an in-memory fake provider, so the suite runs fully offline and
deterministically — no API key or network required.

---

## Security

- `.env` is git-ignored; **never commit real keys**.
- If a key is exposed, rotate it immediately in the [Groq console](https://console.groq.com/keys).
- The library never logs secret values.

See [`SECURITY.md`](SECURITY.md) for the vulnerability-reporting policy.

---

## Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development setup and the quality gate (`ruff check`, `ruff format`, `mypy`, `pytest`) your change
should pass, then open an issue or pull request.

---

## Roadmap

The project is evolving into a free-tier, open-source evaluation platform:
pointwise and reference-based grading, multi-candidate ranking, local (Ollama)
and OpenAI-compatible providers, experiment tracking and a public demo. See
[`ROADMAP.md`](ROADMAP.md) for the phased plan and current status.

---

## License

[MIT](LICENSE) © Syed Waleed Ahmed
