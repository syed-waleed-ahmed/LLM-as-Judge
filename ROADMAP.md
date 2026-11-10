# Roadmap

The project is evolving from a pairwise judging library (v1.0.0) into an
open-source LLM evaluation platform. The guiding constraints:

- **$0 to run.** Use only open-source software and free tiers, with no
  mandatory paid service and no credit card. Every cloud integration is
  optional, and the project works fully offline with the mock provider.
- **Evolution, not rewrite.** The existing public API stays backwards
  compatible.
- **No fake functionality.** A feature counts as done when it is tested.
  Anything not externally verified is labelled as such.

Status key: **done** = implemented and tested, **next** = in progress,
**planned** = not started.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit ([report](docs/design/phase-0-audit.md)) | done |
| 1 | Core hardening: neutral prompt, injection-resistant prompting, exact criteria validation, scale normalisation and weights, consistent bias merges, parallel swap runs, JSON repair retries, brace-aware parsing, verbosity and self-preference signals, judge/candidate metadata | done |
| 2 | Evaluation modes: pointwise, reference-based, multi-candidate ranking | done |
| 3 | Providers: mock (no key), Ollama (local), generic OpenAI-compatible, provider registry | done |
| 4 | Datasets: validation, streaming reads, checkpointed and resumable batches, bundled synthetic sample sets | done |
| 5 | Experiment tracking: SQLite store, run comparison, regression detection (Supabase adapter deferred: not verifiable in CI) | done |
| 6 | REST API (FastAPI, local or self-hosted) and expanded CLI | next |
| 7 | Multi-page Streamlit dashboard with demo mode and quotas | planned |
| 8 | RAG and agent evaluation, reliability analytics (repeat-sampling confidence intervals) | planned |
| 9 | Security and performance hardening (upload limits, dependency audit, concurrency tests) | planned |
| 10 | Free-tier public demo on Streamlit Community Cloud | planned |
| 11 | Documentation site (MkDocs Material), release preparation | planned |

## Known free-tier limits

These limits shape the design. Each is documented where it applies.

- **Groq free plan**: rate-limited per model. The judge retries 429s with
  backoff, and batch concurrency is configurable.
- **Streamlit Community Cloud**: apps sleep when idle and have limited
  resources. The demo therefore defaults to mock mode, and live calls are
  quota-limited.
- **Supabase free plan**: inactive projects can be paused. SQLite is the
  default store, and Supabase stays optional.
- **Public API hosting**: no verified always-free host is assumed. The FastAPI
  service is documented for local or self-hosted use.
