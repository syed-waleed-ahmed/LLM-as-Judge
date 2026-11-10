# Contributing

Thanks for your interest in improving **LLM-as-Judge**! Contributions of all
kinds are welcome: bug reports, features, documentation, tests and new
benchmark samples. By taking part, you agree to follow the
[Code of Conduct](https://github.com/syed-waleed-ahmed/LLM-as-Judge/blob/main/CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug or request a feature** with the
  [issue forms](https://github.com/syed-waleed-ahmed/LLM-as-Judge/issues/new/choose).
  If you can, reproduce bugs with `--provider mock`, so no API key is needed.
- **Pick up an issue.** Issues labelled `good first issue` are a good start.
  Comment on the issue before you begin larger work, so effort isn't duplicated.
- **Improve the docs.** The documentation site is built from `README.md`, the
  top-level project files and `docs/`.
- **Report a security vulnerability** privately, as described in
  [`SECURITY.md`](https://github.com/syed-waleed-ahmed/LLM-as-Judge/blob/main/SECURITY.md),
  never in a public issue.

## Development setup

```bash
git clone https://github.com/syed-waleed-ahmed/LLM-as-Judge.git
cd LLM-as-Judge

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -e ".[all,dev,docs]"   # or: make install
```

Python 3.10 to 3.13 are supported. No API key is needed for development: the
test suite runs fully offline against an in-memory fake provider, and every
command works with `--provider mock`.

## The quality gate

CI runs these checks on every pull request. Run them locally first:

```bash
ruff check .                                  # lint
ruff format --check .                         # formatting
mypy                                          # type-check (src, app.py, dashboard)
bandit -q -c pyproject.toml -r src app.py dashboard   # security lint
pytest --cov=llm_judge                        # tests and coverage
mkdocs build --strict                         # docs, if you changed them
```

With `make`, `make check` runs all of these except the docs build, and
`make docs` builds the site. CI also runs the tests on Python 3.10 to 3.13,
audits dependencies with `pip-audit`, builds the Docker image and the wheel,
and runs CodeQL.

`ruff` and `mypy` are pinned to minor versions in `pyproject.toml`, because
newer releases format and infer differently. Use the pinned versions to avoid
noisy diffs.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make a focused change. One logical change per pull request is easiest to
   review.
3. Add or update tests. New behaviour needs tests, and they must run offline,
   without an API key or network access.
4. Update the docs where behaviour or configuration changes: `README.md`,
   `.env.example`, `API_REFERENCE.md` for CLI commands and REST routes (a test
   checks this), and the pages under `docs/`.
5. Add an entry under **Unreleased** in `CHANGELOG.md`.
6. Open the pull request and fill in the template. CI must pass before
   merging.

Commit messages follow the
[Conventional Commits](https://www.conventionalcommits.org/) style used in the
history, written in the imperative mood. For example:
`feat(providers): add a retry budget to the Ollama provider` or
`fix(cli): exit 2 on usage errors`.

## Project principles

These keep the project usable by anyone, at no cost:

- **$0 to run.** No feature may require a paid service or a credit card. Cloud
  integrations stay optional, and everything works offline with the mock
  provider.
- **No fake functionality.** A feature counts as done when it's tested. Mock
  results are always labelled as such, and benchmark numbers are never
  invented.
- **Backwards compatibility.** The public Python API, CLI flags and result
  fields keep working across minor versions.
- **Secrets stay secret.** Never commit keys, and never include them in
  issues, logs, test fixtures or screenshots.

## Code style

- Code must be `ruff`-clean and `mypy`-clean. Prefer small, focused functions
  with clear docstrings, as in the rest of `src/llm_judge/`.
- Documentation and UI text are plain and professional, with no emojis. The
  dashboard uses Streamlit's Material icons (`:material/...`) where an icon
  helps.
- **Adding a provider?** Implement the `LLMProvider` protocol in
  `src/llm_judge/providers/` and register it with `register_provider` (or,
  for a separate package, the `llm_judge.providers` entry-point group). The
  judging logic shouldn't need any changes. See
  [`ARCHITECTURE.md`](https://github.com/syed-waleed-ahmed/LLM-as-Judge/blob/main/ARCHITECTURE.md).

## Releases (maintainers)

1. Move the **Unreleased** changelog entries under a new version heading, and
   bump `__version__` in `src/llm_judge/__init__.py`.
2. Merge to `main`, then push a tag such as `v2.1.0`.

The release workflow builds the package and creates a GitHub Release, using
that version's changelog section as the notes. If PyPI publishing is enabled,
it also publishes there. See
[`DEPLOYMENT.md`](https://github.com/syed-waleed-ahmed/LLM-as-Judge/blob/main/DEPLOYMENT.md).
