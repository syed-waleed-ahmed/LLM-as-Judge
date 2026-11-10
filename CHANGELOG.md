# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
