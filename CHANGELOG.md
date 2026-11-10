# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-11-10

The next phases from the roadmap:
- a PostgreSQL experiment store, including Supabase, tested in CI against a
  real server;
- live CI checks for the Ollama provider and Ollama embeddings, both passing;
- three more benchmark samples.

The public API and the result schema are unchanged.

### Added
- PostgreSQL experiment store, including Supabase.
  - Set `JUDGE_DB_PATH` to a `postgresql://` URL (`pip install -e ".[postgres]"`)
    to keep recorded runs in PostgreSQL instead of SQLite, for example on a free
    Supabase project.
  - Both backends sit behind the same `ExperimentStore` API and store the same
    data, so comparisons, reports, analytics, the CLI, the REST API and the
    dashboard all work unchanged.
  - Server-side prepared statements are disabled, so Supabase's transaction
    pooler works.
  - The password is redacted wherever the location is shown (`store.location`),
    and the setting is excluded from the settings repr.
  - A new CI job runs the store tests against a real PostgreSQL 16 service
    container, and fails if they're skipped.
- Three more bundled benchmark samples, all original and synthetic:
  - `math_reasoning` (reference): word problems, some answered wrongly. In a
    live Groq run, the judge scored both wrong answers near 0 and flagged them
    as unsupported claims.
  - `customer_support` (pairwise): helpful, on-brand replies against
    dismissive or rambling ones.
  - `data_extraction` (reference): JSON fields, some missing or invented.
- A "Live Ollama" workflow checks the Ollama provider and Ollama embeddings
  against a real server: the official `ollama/ollama` image, with
  `qwen2.5:1.5b` and `nomic-embed-text`. It runs weekly, on demand, and on
  pull requests that touch them (`scripts/live_ollama_check.py`).

## [2.1.0] - 2026-11-10

Closes the gaps that a line-by-line review against the original platform plan
found:
- judge instructions, opt-in provider fallback, async methods and a result
  cache;
- Markdown evaluation reports;
- Analytics and Settings dashboard pages;
- optional semantic similarity;
- a free-tier rate-limit fix, screenshots and a coverage badge.

Python 3.10 is deprecated. The public API is backwards compatible. The result
schema only gains optional fields (`reference_metrics.semantic_similarity` and
`embedding_model`, plus new metadata keys), so `schema_version` is now "1.3".

### Added
- Evaluator instructions: optional guidance for the judge ("penalise answers
  without sources") on pairwise, pointwise, reference and ranking
  evaluations, dataset runs and benchmarks. It is available as
  `instructions=` in the Python API, `--instructions` /
  `--instructions-file` in the CLI, an `instructions` field in the REST API,
  and a field on the dashboard's Playground. It goes into the system prompt
  outside the untrusted-content markers, and can't change the response
  format. It is recorded in the result metadata and the experiment run's
  configuration; run comparisons flag different guidance, and resuming a
  checkpoint with different guidance is refused. Its length is capped by
  `JUDGE_MAX_INSTRUCTIONS_CHARS`. The prompt version is now 2.1.
- Opt-in provider fallback (`JUDGE_FALLBACK_PROVIDERS`). When the primary
  provider fails, the configured fallbacks are tried in order.
  - Every result answered by a fallback carries a `provider_fallback` warning
    and a `metadata.fallback` record (which provider failed, which answered,
    and why). `metadata.judge` keeps naming the configured judge.
  - An exhausted demo budget and replies that fail validation never fall
    back, and the mock judge can't be a fallback.
  - `llm-judge health` and `/api/v1/system/status` report every provider in
    the chain.
  - Verified live, with Ollama down and Groq answering.
- Async evaluation methods: `aevaluate`, `aevaluate_pointwise`,
  `aevaluate_reference`, `arank`, `aevaluate_rag` and `aevaluate_agent`.
  They run the synchronous engine in worker threads, so they can be awaited
  and gathered without blocking an event loop, and they're fully typed.
- Opt-in result cache (`JUDGE_CACHE_SIZE`, `llm_judge.cache`).
  - An in-memory LRU cache of identical temperature-0 evaluations; the key
    covers inputs, criteria, instructions, judge, prompt version and scoring
    settings.
  - Hits are marked `metadata["cache_hit"]` and don't spend provider quota.
  - Answers from a fallback provider are never cached, and consistency
    analysis always bypasses the cache.
- Evaluation reports: `run_report()`, `llm-judge experiments report` and
  `GET /api/v1/runs/{id}/report` produce a shareable Markdown report of a
  recorded run. It covers how the run was judged (criteria, instructions,
  settings), the results (overall, per criterion, per model, warnings,
  failures), an optional comparison with a baseline, and the reproducibility
  metadata. Mock runs are labelled. The dashboard's Experiments page offers
  run and comparison reports as downloads.
- `GET /api/v1/experiments/{name}`: one experiment and a page of its runs.
- Optional semantic similarity for reference grading.
  - With `JUDGE_EMBEDDINGS_PROVIDER` set to `ollama` (local, e.g.
    `nomic-embed-text`) or `openai_compatible` (any `/v1/embeddings` server),
    `reference_metrics.semantic_similarity` holds the cosine similarity of
    the output's and the reference's embeddings, alongside token F1.
  - Batch outputs, run summaries (`mean_semantic_similarity`), reports, the
    CLI and the dashboard show it.
  - An embedding failure leaves it empty, with a
    `semantic_similarity_unavailable` warning.
  - Tested against simulated responses only: Groq offers no embeddings API.
- README and docs site:
  - screenshots of the Playground, RAG evaluation, Experiments and Analytics
    pages, with live Groq judgments;
  - an architecture diagram, a technology-stack section and a limitations
    section;
  - a coverage badge. The docs workflow measures coverage on every deploy
    from `main` and publishes it for shields.io (`scripts/coverage_badge.py`),
    so no third-party coverage service is needed.
- Dashboard **Analytics** page, computed only from recorded runs:
  - score trends across runs, and the score or margin distribution of a run;
  - per-criterion comparison;
  - bias signals: the share of rows with each reliability warning, with the
    caveat that these are heuristics;
  - operational reliability: failure rate, judge latency, rows answered by
    a fallback, cache hits.
- Dashboard **Settings** page:
  - judge temperature and tie margin for the current session;
  - providers and their configuration, a provider health check (no
    inference), and the fallback chain;
  - limits in effect, including REST API limits;
  - storage and cache status;
  - privacy controls: in the public demo, delete your session's runs.

  The About page now focuses on how results stay honest, and its install
  instructions no longer point at PyPI.
- The public demo is live at <https://llm-as-judge.streamlit.app/> (Streamlit Community
  Cloud, demo mode). The README, the documentation site and the package
  metadata link to it.

### Deprecated
- Python 3.10, which reached end of life in October 2026.
  - The library, CLI and REST API still run on it, and a new CI job audits
    their dependencies on 3.10.
  - Importing the package on 3.10 emits a `DeprecationWarning`.
  - The dashboard should run on Python 3.11 or later, because Pillow's
    security fixes need it. The Settings page warns when the app runs on
    3.10.

### Verified
- The `openai_compatible` provider was tested live against Groq's
  OpenAI-compatible endpoint: health check, pairwise, reference grading and
  RAG evaluation. Previously it was only tested against simulated responses.

### Fixed
- The public demo kept running the package version installed at its first
  deploy. `requirements.txt` installed the package non-editably, and Streamlit
  Community Cloud only reinstalls when that file changes, so the new Settings
  page failed with an `ImportError`. The demo now installs it editable
  (`-e .[ui]`). The dashboard also shows a clear error, rather than failing
  on some pages, whenever the installed package doesn't match its code.
- Rate limits on free tiers. Batch runs on Groq's free plan (8,000 tokens per
  minute) lost rows: retries waited 0.5 to 2 s when Groq asked for 5 to 12 s,
  and concurrent workers retried in lockstep. Retries now follow the provider's
  own hint: the `Retry-After` header, or Groq's "try again in 6.48s" message
  (`ProviderError.retry_after`), capped at 60 s and with jitter. A rate-limit
  pause also applies to every worker sharing the judge. In a live stress test
  of back-to-back benchmarks, 11 of 11 rows now succeed, against 8 of 11
  before.
- The CLI's log handler kept the `stderr` stream that existed when logging was
  set up. After a host or test runner replaced and closed that stream, log
  calls failed with "I/O operation on closed file". It now follows the
  current `sys.stderr`.

## [2.0.1] - 2026-11-10

A maintenance release: a dashboard fix, documentation-site fixes and
improvements, and project files for contributors. The library's Python API,
CLI and REST API are unchanged.

### Added
- Community files for the public repository: issue forms, a pull request
  template, `CODEOWNERS`, `CITATION.cff` and a Dependabot configuration.
- CodeQL code scanning for the Python code and the GitHub Actions workflows.
- `make security` and `make docs`. `make check` now also runs Bandit, as CI
  does.
- Pre-commit hooks (`.pre-commit-config.yaml`, `make hooks`): ruff, whitespace
  and line endings, YAML and TOML syntax, merge-conflict markers, large files
  and private keys.
- Documentation site:
  - instant navigation with a progress bar, top-level tabs, previous and
    next links, search suggestions and shareable searches;
  - Open Graph and Twitter card tags for link previews;
  - fonts served from the site itself (privacy plugin), so pages make no
    third-party requests.

### Fixed
- Dashboard: with Streamlit 1.67, the first chart could fail with
  "partially initialized module 'pandas' has no attribute 'Series'". Newer
  Streamlit imports pandas lazily in another thread, and Plotly could see the
  half-imported module. The charts module now imports pandas itself first,
  which waits for that import to finish. `pandas` is now declared in the `ui`
  extra.
- `scripts/e2e_dashboard.py` finds sidebar links on older Streamlit versions
  too, where the link text includes the icon name.
- Links on the documentation site to repository files (`DEPLOYMENT.md`,
  `.env.example`, `ROADMAP.md` and others) returned 404. A build hook
  (`scripts/docs_links.py`) now points them to the matching page or to the
  file on GitHub, and fails the strict build on links to missing files.
- The committed knowledge graph (`graphify-out/`) described v1 and contained
  absolute local paths. It is regenerated for v2, with repository-relative
  paths.

### Changed
- GitHub Release notes come from this changelog (`scripts/release_notes.py`)
  instead of being generated from pull request titles. The release build fails
  if the version has no changelog section.
- Install instructions no longer point at PyPI, where the package isn't
  published yet. They cover installing from a clone, from a Git tag, or from
  the wheel attached to a GitHub Release.
- Workflows default to a read-only `GITHUB_TOKEN`.
- Package metadata: description, keywords, classifiers, and links to the
  documentation, changelog and releases.

## [2.0.0] - 2026-11-10

The first platform release. It bundles development versions 1.1.0 to 1.9.0,
which were never published separately; `CHANGELOG.md` has their detailed
entries. LLM-as-Judge grows from a pairwise judging library into an evaluation
platform:
- pointwise, reference, ranking, RAG and agent evaluation;
- reliability safeguards and analytics;
- Groq, Ollama, OpenAI-compatible and offline mock providers;
- datasets, resumable batches and benchmarks;
- experiment tracking with regression detection;
- a REST API and a multi-page dashboard with a public demo mode;
- a security hardening pass, a documentation site and release automation.

### Upgrade notes
- The Python API is backwards compatible. `Judge.evaluate(...)`, the result
  fields and the CLI flags from 1.0 still work, and new result fields are
  additive (`schema_version`).
- The default Groq model is `openai/gpt-oss-120b`, because Groq retired
  `llama-3.3-70b-versatile`. Update `GROQ_MODEL` only if you pinned
  the old model.
- `llm-judge batch` now writes a checkpoint by default and accepts any dataset
  kind. JSONL output rows gain a nested `result` object.
- The overall winner is decided on weighted, scale-normalised scores. On the
  default 1-10 scale the tie behaviour is unchanged.

### Fixed (independent code review and end-to-end testing)

Two independent code reviews and a real-browser end-to-end run of the
dashboard, deployed as on Streamlit Community Cloud, found these defects. All
are fixed, with regression tests.

- Checkpoints: a torn last line (crash mid-write) broke every later resume.
  Resuming with a different judge, criteria or bias setting silently mixed
  results; it is now refused.
- YAML datasets: aliases allowed a ~600-byte "billion laughs" file to exhaust
  memory. Aliases are now rejected, and so are nested field values.
- Dataset columns:
  - padded header names crashed loading;
  - two aliases for the same field overwrote each other;
  - encoding and oversized-field CSV errors escaped as raw exceptions;
  - duplicate-id issues reported the wrong row.
- The Groq SDK's hidden retries multiplied requests (up to 12 per call) and
  made the demo quotas undercount. Groq JSON-mode rejections skipped the
  repair retry.
- Run comparisons ignored failed and missing rows, so a mostly failed
  candidate could pass the regression gate. They now warn, and return
  `insufficient_data` when too few rows succeeded.
- REST API:
  - random `X-API-Key` values bypassed the rate limit on open servers;
  - authentication ran after body parsing;
  - jobs ignored the concurrency cap;
  - over-long criterion names returned 500;
  - `%` and `_` were wildcards in run lookups.
- CLI:
  - usage errors exited 1 instead of 2;
  - an unsupported `--output` failed only after the paid run;
  - `--json` output mixed progress text into stdout;
  - `consistency` exited 0 when nothing could be measured;
  - `LOG_LEVEL` was ignored.
- Dashboard:
  - uploaded agent traces were ignored;
  - switching dataset samples dropped their criteria;
  - downloads cleared the results;
  - an exhausted quota during a batch was not explained;
  - the quota caption lagged one action behind;
  - numeric-looking ids broke chart order.
- Long agent traces lost their final answer to truncation.
- "Output A and B" was relabelled incorrectly in bias-mitigated merges.
- Criterion presets matched case-sensitively, and invalid criterion names
  raised the wrong exception type.

### Phase 11: documentation and release
- A MkDocs Material site (`mkdocs.yml`, `docs/`) that composes the README and
  the project documents (single source of truth), with a GitHub Pages
  workflow.
- `ARCHITECTURE.md`, `API_REFERENCE.md` and `CODE_OF_CONDUCT.md`. A test fails
  if any CLI command or REST route is missing from the API reference.
- CI jobs: a strict docs build, plus package build, `twine check` and a
  clean-environment wheel smoke test.
- A tag-triggered release workflow that creates a GitHub Release and, once
  enabled, publishes to PyPI with Trusted Publishing.
- Packaging uses an SPDX license expression (`license = "MIT"`) and requires
  setuptools 77 or later.

## [1.9.0] - Included in 2.0.0

Phase 9 of the platform roadmap: security and performance hardening.

### Added
- A threat model in `SECURITY.md`, with a tested mitigation for each risk.
- CSV-injection protection (`llm_judge.csv_safety`) in every CSV writer:
  batch results, dataset results and experiment exports.
- A Bandit job in CI, with reviewed and documented suppressions.
- A `Dockerfile` (non-root; dashboard by default, REST API on demand) and a
  CI job that builds the image and smoke-tests both services in containers.
- Hardening tests:
  - hostile datasets (deep nesting, YAML object tags, oversized files);
  - secret hygiene in errors, logs and API responses;
  - exact quotas and rate limits under 32-64 concurrent threads;
  - a 400-row, 16-worker runner stress test with checkpoint integrity.
- `scripts/perf_baseline.py` and `BENCHMARKS.md`: measured library overhead,
  concurrency scaling and the parallel bias-mitigation speed-up.

### Fixed
- Deeply nested JSON in datasets, checkpoints, candidate files or judge
  replies raised `RecursionError` (a crash, or a 500 from the API). It is now
  a clean validation or parsing error.

## [1.8.0] - Included in 2.0.0

Phase 8 of the platform roadmap: RAG, agents, and reliability analytics.

### Added
- `Judge.evaluate_rag()`: claim-level verdicts, from which faithfulness,
  answer relevance, context precision (average precision) and context recall
  are computed deterministically. It lists unsupported claims and flags prompt
  injection in retrieved passages.
- `Judge.evaluate_agent()`: framework-agnostic `AgentTrace` with
  deterministic checks (unknown tools, missing arguments, malformed or failed
  calls, repeats, step budget, expected tools, missing answer), plus a judged
  grade on new agent criteria presets.
- `llm_judge.reliability`: `measure_consistency()` (agreement rate, spread,
  bootstrap CI), `judge_agreement()` across stored runs, `cohen_kappa()`, and
  tie-aware `spearman()`.
- The mock judge understands RAG prompts, so everything works offline.
- CLI: `rag`, `agent`, `consistency`, `experiments agreement`.
- API: `POST /evaluate/rag`, `/evaluate/agent`, `/analysis/consistency`
  (capped at 10 samples), and `GET /runs/agreement`.
- Dashboard pages: RAG evaluation, Agent evaluation, Reliability.

### Changed
- mypy is pinned to 2.4.x. Its inference changes between versions, which made
  local and CI results disagree.
- The dashboard handles `None` from select boxes, which newer Streamlit types
  allow.

### Verified live
- RAG and agent prompts were run against Groq `openai/gpt-oss-120b` with
  schema-valid results.

## [1.7.0] - Included in 2.0.0

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

## [1.6.0] - Included in 2.0.0

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

## [1.5.0] - Included in 2.0.0

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

## [1.4.0] - Included in 2.0.0

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

## [1.3.0] - Included in 2.0.0

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

## [1.2.0] - Included in 2.0.0

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

## [1.1.0] - Included in 2.0.0

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

## [1.0.0] - 2026-07-04

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

[Unreleased]: https://github.com/syed-waleed-ahmed/LLM-as-Judge/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/syed-waleed-ahmed/LLM-as-Judge/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/syed-waleed-ahmed/LLM-as-Judge/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/syed-waleed-ahmed/LLM-as-Judge/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/syed-waleed-ahmed/LLM-as-Judge/releases/tag/v2.0.0
