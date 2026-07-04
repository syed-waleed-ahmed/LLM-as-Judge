# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
