# Roadmap

Version 2.0.0 (released 2026-11-10) turned a pairwise judging library (v1.0.0)
into an open-source LLM evaluation platform, in the phases below. Version 2.1.0
closed the remaining gaps that a line-by-line review against the original plan
found; the [delivery report](docs/design/delivery-report.md) lists them. The
guiding constraints, which still apply to new work:

- **$0 to run.** Use only open-source software and free tiers, with no
  mandatory paid service and no credit card. Every cloud integration is
  optional, and the project works fully offline with the mock provider.
- **Evolution, not rewrite.** The existing public API stays backwards
  compatible.
- **No fake functionality.** A feature counts as done when it is tested.
  Anything not externally verified is labelled as such.

Status key: **done** = implemented and tested; **ready** = implemented and
tested, waiting for an account-specific step.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository audit ([report](docs/design/phase-0-audit.md)) | done |
| 1 | Core hardening: neutral prompt, injection-resistant prompting, exact criteria validation, scale normalisation and weights, consistent bias merges, parallel swap runs, JSON repair retries, brace-aware parsing, verbosity and self-preference signals, judge/candidate metadata | done |
| 2 | Evaluation modes: pointwise, reference-based, multi-candidate ranking | done |
| 3 | Providers: mock (no key), Ollama (local), generic OpenAI-compatible, provider registry | done |
| 4 | Datasets: validation, streaming reads, checkpointed and resumable batches, bundled synthetic sample sets | done |
| 5 | Experiment tracking: SQLite store, run comparison, regression detection (Supabase adapter deferred: not verifiable in CI) | done |
| 6 | REST API (FastAPI, local or self-hosted) and expanded CLI | done |
| 7 | Multi-page Streamlit dashboard with demo mode and quotas | done |
| 8 | RAG and agent evaluation, reliability analytics (repeat-sampling confidence intervals) | done |
| 9 | Security and performance hardening (upload limits, dependency audit, concurrency tests) | done |
| 10 | Free-tier public demo on Streamlit Community Cloud ([live](https://llm-as-judge.streamlit.app/), [guide](DEPLOYMENT.md)), tested in a browser | done |
| 11 | Documentation site (MkDocs Material, [published](https://syed-waleed-ahmed.github.io/LLM-as-Judge/)) and release automation ([v2.0.0](https://github.com/syed-waleed-ahmed/LLM-as-Judge/releases/tag/v2.0.0)) | done; PyPI publishing is opt-in (see DEPLOYMENT.md) |

## What's next

These are open and welcome contributions. Open an issue to discuss an approach
before starting.

- **Supabase/Postgres experiment store.** An optional adapter behind the
  existing store interface. It needs a way to test it in CI without a paid
  service, such as a Postgres service container.
- **Publishing to PyPI.** The release workflow is ready; it needs the PyPI
  trusted-publisher setup described in DEPLOYMENT.md.
- **More providers.** Further free or local backends, through the provider
  registry or the `llm_judge.providers` entry-point group.
- **More benchmark samples.** Small, synthetic, clearly licensed datasets for
  the bundled benchmarks.
- **Live checks for Ollama and embeddings.** The Ollama provider and the
  embedding backends are tested against simulated responses only; a CI job
  with a small local model would verify them end to end.
- **Drop Python 3.10.** It is deprecated (end of life since October 2026) and
  can be removed in the next major or minor release.

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
