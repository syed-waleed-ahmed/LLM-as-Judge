# Security Policy

## Reporting a vulnerability

If you discover a security issue, please **don't open a public issue**.
Report it privately instead:

- **Preferred:** use GitHub's
  [private vulnerability reporting](https://github.com/syed-waleed-ahmed/LLM-as-Judge/security/advisories/new)
  (Security tab, then "Report a vulnerability").
- **Or** email the maintainer at **syedwaleedahmed9@gmail.com**.

Please include:

- a description of the issue and its impact, and
- steps to reproduce (a minimal proof of concept if possible).

You can expect an acknowledgement within a few days. Please allow a
reasonable window for a fix before any public disclosure. Fixed
vulnerabilities are published as GitHub security advisories and noted in the
changelog.

## Supported versions

| Version | Supported |
|---|---|
| 2.x | yes |
| earlier than 2.0 | no |

Security fixes go into the latest release.

**Python versions.** Python 3.11 to 3.14 are supported. Support for Python
3.10 ended in 2.3.0: it reached end of life in October 2026, and the fixed
releases of dependencies such as Pillow need Python 3.11 or later. The public
demo runs on Python 3.12.

## Threat model

The judge processes **untrusted text**: candidate outputs, retrieved RAG
passages, uploaded datasets and agent traces. It runs on **finite free-tier
quotas**, and can be exposed as a **public demo** or a **REST API**. These are
the risks considered and how each is mitigated. Every mitigation listed here
has tests.

| Risk | Mitigation | Where |
|---|---|---|
| Prompt injection in candidate outputs or retrieved passages | Inputs are fenced between boundary markers carrying a random per-request token, and the judge is told they're data. Suspicious text raises `possible_prompt_injection` warnings. | `prompts.py`, `diagnostics.py`, `rag.py` |
| Judge reply manipulation or malformed output | Strict schema validation, exact criteria and context matching, bounded repair retries, and winners recomputed from the scores | `parsing.py`, `scoring.py`, `judge.py` |
| Hostile dataset files (huge, deeply nested, YAML object tags or alias bombs) | Byte and row limits checked before parsing; a `SafeLoader`-based YAML loader that also rejects aliases; nested field values rejected; deep nesting is a validation error. Uploaded content is never executed. | `datasets/io.py`, `datasets/validation.py` |
| CSV / formula injection in exported results | Text cells starting with `=`, `+`, `-`, `@`, tab or CR are prefixed with `'` (OWASP guidance) | `csv_safety.py` |
| Secret leakage | Keys are excluded from the `Settings` repr, error messages, logs, API responses and experiment metadata. `.env` and `.streamlit/secrets.toml` are git-ignored, and the OpenAI-compatible provider only reads prefixed `JUDGE_OPENAI_*` variables. | `config.py`, `api/`, `experiments/store.py` |
| Quota exhaustion and abuse (public demo) | Per-session and daily call budgets enforced at the provider layer (retries included), plus input, row, candidate and upload caps | `providers/quota.py`, `dashboard/common.py` |
| Abuse of the REST API | Authentication and rate limiting in middleware, before the body is parsed; the rate-limit identity is the *authenticated* key or the client address, never an unverified header, and failed authentication is rate-limited too. Optional API keys use constant-time comparison. Admin routes are disabled unless `JUDGE_API_ADMIN_KEY` is set. Also a request-size limit that covers chunked bodies, a concurrency cap that sheds load with 503 (jobs included), and a bounded job queue. | `api/security.py`, `api/app.py`, `api/jobs.py` |
| Information disclosure in API errors | One JSON error envelope with request ids; stack traces and internals are never returned | `api/app.py` |
| Privacy of evaluated content | The experiment store keeps hashes of raw inputs, not the inputs. Public-demo sessions keep nothing after the session ends. | `experiments/store.py`, `dashboard/common.py` |
| Vulnerable dependencies | `pip-audit` runs in CI | `.github/workflows/ci.yml` |

## Static analysis

CI runs [Bandit](https://bandit.readthedocs.io) on the library, CLI, API and
dashboard. Three checks are suppressed after review (see `[tool.bandit]` in
`pyproject.toml`):

- **B101 (`assert`)**: used for internal invariants and type narrowing, never
  for security decisions.
- **B311 (`random`)**: seeded pseudo-randomness for reproducible sampling and
  bootstrap resampling, which has no security purpose. Boundary tokens use
  `secrets`.
- **B615 (Hugging Face download)**: a false positive. This project's own
  `load_dataset()` reads local files only.

## Handling secrets

- Keys come from the environment, a local `.env` file or Streamlit secrets.
  They're never committed: `.env.example` and
  `.streamlit/secrets.toml.example` contain no real values.
- When exposing the REST API beyond localhost, set `JUDGE_API_KEYS` and serve
  it behind TLS.
- A PostgreSQL `JUDGE_DB_PATH` (for example a Supabase connection string)
  contains a password. It's excluded from the settings repr and redacted
  wherever the store's location is shown. Keep it in `.env` or secrets.
- If a key is ever exposed, **rotate it immediately** in the provider's
  console (for Groq: <https://console.groq.com/keys>).
