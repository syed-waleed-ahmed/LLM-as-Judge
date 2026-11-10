# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - Unreleased

Phase 7 of the platform roadmap: the dashboard and public demo mode.

### Added
- Multi-page Streamlit dashboard: Overview, Playground (all four evaluation
  modes, with example loaders), Datasets (upload, validate, evaluate,
  download), Benchmarks, Experiments (run comparison with verdict, confidence
  interval and per-row changes), and About.
- A sidebar to choose the live or offline mock judge, with every result
  badged by provider. Mock results are always labelled as not coming from an
  LLM.
- Themed Plotly charts. The palette passes the colorblind and contrast checks
  in both light and dark modes, and every chart has a table view.
- Public demo mode (`JUDGE_DEMO_MODE`): per-session and daily live-call
  budgets, input, row, candidate and upload caps, and session-only storage.
- `QuotaProvider` and `CallBudget`: enforce call budgets on any provider,
  counting retries and bias runs, and refusing with a non-retryable
  `QuotaExceededError`.
- `DEPLOYMENT.md` (free Streamlit Community Cloud steps, secrets, smoke
  tests), `.streamlit/config.toml` and `.streamlit/secrets.toml.example`.

### Changed
- `app.py` is now a navigation shell; the pages live in `dashboard/`.
- The `ui` extra adds Plotly and requires Streamlit 1.50 or later. mypy now
  also checks the dashboard.

## [1.6.0] - Unreleased

Phase 6 of the platform roadmap: REST API.

### Added
- `llm_judge.api` (FastAPI, `[api]` extra) and `llm-judge serve`, which
  binds to localhost by default and warns when exposed without API keys.
- Versioned `/api/v1` endpoints for pairwise, pointwise, reference and
  ranking evaluation, synchronous batches, background dataset jobs, dataset
  validation, benchmarks, experiments and run comparison, providers,
  criteria, status and statistics. OpenAPI docs are at `/docs`.
- Background jobs: a bounded queue, cancellation, `Idempotency-Key` replay
  with conflict detection, paginated rows, and optional experiment recording.
- Security and resilience:
  - optional API keys (constant-time comparison);
  - admin routes disabled unless `JUDGE_API_ADMIN_KEY` is set;
  - per-client token-bucket rate limiting;
  - an ASGI request-size limit that also covers chunked bodies;
  - a concurrency cap that sheds load with 503.
- Operations: one JSON error envelope with request ids, and structured JSON
  access logs (no bodies).
- `[all]` extra; `make serve`; `JUDGE_API_*` settings.

### Fixed
- Queued jobs are marked cancelled on shutdown instead of staying "queued".

## [1.5.0] - Unreleased

Phase 5 of the platform roadmap: experiment tracking.

### Added
- `llm_judge.experiments.ExperimentStore`: a SQLite store (standard library,
  no server) for experiments, runs and per-row results, with reproducibility
  metadata. Raw inputs are stored only as hashes.
- `compare_runs()`: rows paired by id, a mean paired difference with a seeded
  bootstrap (or normal-approximation) 95% CI, a regression/improvement
  verdict with a minimum effect size, the most-changed rows, winner flips for
  pairwise runs, per-criterion deltas, and comparability warnings.
- CLI: `--experiment`, `--run-name` and `--db` on `batch` and `benchmark`;
  `experiments list|runs|show|compare|export|delete`;
  `compare --fail-on-regression` for CI gates.
- Setting `JUDGE_DB_PATH`.

### Deferred
- A Supabase/PostgreSQL storage adapter. The store has a narrow interface so
  one can be added, but it can't be verified in this project's CI, so it isn't
  shipped untested.

## [1.4.0] - Unreleased

Phase 4 of the platform roadmap: datasets, resumable batches and benchmarks.

### Added
- `llm_judge.datasets`: CSV, JSON, JSONL and YAML (`[yaml]` extra) readers
  with byte and row limits; typed pairwise, pointwise and reference records
  with column aliases; validation reports (missing values, duplicate ids and
  content, unknown columns, oversized fields, preview); content fingerprints
  for dataset versioning; reproducible sampling and splitting.
- `DatasetRunner`: bounded-concurrency evaluation with an append-only JSONL
  checkpoint, graceful Ctrl+C, and safe resume (it refuses changed records and
  retries failed rows).
- `summarize()`: winners and per-model win rates (pairwise), and mean scores
  per model, mean token F1 and the unsupported-claims rate (graded).
- Six original synthetic sample datasets: general QA, code generation,
  summarization, RAG, instruction following, safety.
- Criteria presets: Instruction Following, Safety.
- CLI: `datasets validate|samples|export` and `benchmark`.
- Settings: `JUDGE_MAX_DATASET_BYTES`, `JUDGE_MAX_DATASET_ROWS`.

### Changed
- `llm-judge batch` accepts all three dataset kinds and every format. It
  checkpoints by default, resumes with `--resume`, supports `--sample`,
  `--limit` and `--skip-invalid`, prints a summary, and exits with 130 when
  interrupted. Existing flags and the pairwise CSV columns are unchanged (two
  model columns were added).
- ruff is pinned to 0.17.x so local and CI formatting agree.

## [1.3.0] - Unreleased

Phase 3 of the platform roadmap: multi-provider architecture.

### Added
- `MockProvider`: an offline, deterministic heuristic judge (not an LLM) for
  demos, tests and keyless use. Every explanation is labelled as mock output.
- `OllamaProvider`: free local inference through Ollama's native API, with
  actionable errors for a stopped server or a model that isn't pulled.
- `OpenAICompatibleProvider`: any `/chat/completions` endpoint, configured via
  prefixed `JUDGE_OPENAI_*` settings so an unrelated paid key is never used.
- Provider registry (`build_provider`, `register_provider`, `provider_names`)
  with plugin discovery through the `llm_judge.providers` entry-point group.
- Provider health checks that run no inference, and thread-safe per-provider
  usage counters (requests and tokens, where the backend reports them).
- CLI: global `--provider` flag, plus `providers`, `health` and `version` commands.
- Settings: `JUDGE_PROVIDER`, `JUDGE_OLLAMA_*`, `JUDGE_OPENAI_*`.

### Fixed (found by live testing against Groq)
- The default Groq model `llama-3.3-70b-versatile` has been retired by Groq;
  the default is now `openai/gpt-oss-120b`. A retired or unknown model now
  produces an actionable error pointing at `GROQ_MODEL` and `llm-judge health`.
- The CLI no longer crashes when a judge's reply contains characters the
  console can't encode (e.g. U+202F on Windows cp1252). `--json` output stays
  lossless by escaping non-ASCII on non-UTF-8 streams.

### Changed
- `Judge()` builds its provider from `JUDGE_PROVIDER` (default `groq`, as before).
- API keys are excluded from the `Settings` repr.
- The missing-key error suggests `--provider mock`.
- `httpx` is now an explicit dependency. It was already installed via `groq`.

## [1.2.0] - Unreleased

Phase 2 of the platform roadmap: new evaluation modes. Result schema "1.2"
adds new result types; existing pairwise results are unchanged.

### Added
- `Judge.evaluate_pointwise()`: grade a single output on weighted,
  scale-normalised criteria, with improvement feedback.
- `Judge.evaluate_reference()`: grade an output against an expected answer.
  Results list `missing_information` and `unsupported_claims` (hallucination
  indicators, which also raise an `unsupported_claims` warning) and include
  deterministic `reference_metrics` (exact match, token F1).
- Reference criteria presets: Correctness, Completeness, Factual Consistency.
- `Judge.rank()`: multi-candidate leaderboards. The `round_robin` strategy
  plays every pair with alternating presentation order; the `pointwise`
  strategy grades each candidate independently. Competition ranking handles
  ties, and individual comparison failures are isolated.
- CLI: `llm-judge grade` (with `--reference`) and `llm-judge rank`
  (`--candidates-file`, `--strategy`). `list-criteria` shows the reference presets.
- Setting `JUDGE_MAX_CANDIDATES`.

## [1.1.0] - Unreleased

Phase 1 of the platform roadmap: core reliability hardening. All changes are
backwards compatible. Existing calls, CLI commands and result fields keep
working, and new result fields are additive (`schema_version` "1.1").

### Added
- Criterion `weight`, plus `overall_score_a` and `overall_score_b`: weighted
  means of scale-normalised (0-1) scores. CLI `--weights`; weight inputs in the UI.
- Typed `warnings` on every result: `possible_prompt_injection`,
  `possible_verbosity_bias`, `self_preference_risk`, `position_inconsistent`,
  `score_out_of_range`, `input_truncated`.
- Judge and candidate metadata (`metadata.judge`, `metadata.candidates`), with
  `candidate_a_model` and `candidate_b_model` arguments, CLI `--model-a` and
  `--model-b`, and optional batch `model_a` and `model_b` columns.
- Repair retries for malformed or schema-violating replies
  (`JUDGE_MAX_PARSE_RETRIES`).
- `schema_version` on results; `PROMPT_VERSION` recorded in metadata.
- Settings: `JUDGE_PARALLEL_BIAS_RUNS`, `JUDGE_TIE_MARGIN`,
  `JUDGE_VERBOSITY_RATIO`, `JUDGE_MAX_CRITERIA`, `JUDGE_MAX_PARSE_RETRIES`.
- Property-based tests (Hypothesis), Streamlit smoke tests, `ruff format`
  check, Python 3.13 in CI, and a `pip-audit` dependency scan.

### Changed
- The system prompt shows the output shape with placeholders instead of
  example values that could anchor the judge.
- The task and outputs are wrapped in boundary markers that carry a random
  per-request token, and the judge is told to treat them as untrusted data.
- Returned criteria must match the requested criteria exactly
  (case-insensitive). Missing, extra or duplicate entries are rejected instead
  of being accepted or clamped to 0-100.
- The overall winner is now decided on weighted, scale-normalised scores. On
  the default 1-10 scale the tie behaviour is unchanged: whole-point gaps pick
  a winner and half-point gaps tie.
- Bias-mitigation runs execute concurrently. Merged explanations and comments
  come from the ordering that agrees with the averaged verdict, with "Output
  A/B" references relabelled for the swapped run.
- JSON recovery uses a real JSON decoder instead of first-`{`-to-last-`}`
  slicing. Non-finite scores are rejected.
- The package version is defined once (`llm_judge.__version__`).

## [1.0.0] — 2026-07-04

First production-ready release. The project was rebuilt from a small collection
of scripts into an installable, tested, and documented framework.

### Added
- Installable `llm_judge` package (src layout) exposing a stable public API,
  shipped with a `py.typed` marker (PEP 561).
- `Judge` engine with:
  - strict, Pydantic-validated JSON output (provider JSON mode),
  - **position-bias mitigation** (judge both A/B orders and average),
  - score clamping and winner reconciliation for internally consistent results,
  - automatic retries with exponential backoff for transient provider errors.
- Pluggable provider layer (`LLMProvider` protocol) with a Groq implementation.
- Multi-criteria scoring and six built-in criteria presets.
- Concurrent batch evaluation (`llm-judge batch`) for CSV/JSONL datasets, with
  per-row failure isolation.
- `llm-judge` command-line interface (`evaluate`, `batch`, `list-criteria`).
- Streamlit UI rebuilt as a thin layer over the library, with multi-criteria
  selection, bias toggle, and JSON download.
- Typed configuration via `pydantic-settings` (no import-time crash).
- Test suite: 59 offline unit tests (~97% coverage), plus `ruff` and `mypy`.
- GitHub Actions CI across Python 3.10–3.12; `Makefile`; `MIT` license.

### Changed
- Configuration no longer raises at import time; API-key validation is lazy.
- Groq client is created per `Judge` instance instead of at module import.

### Removed
- Legacy flat modules (`config.py`, `judge.py`, `prompts.py`) superseded by the
  package.
