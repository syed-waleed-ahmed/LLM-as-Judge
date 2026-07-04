# Contributing

Thanks for your interest in improving **LLM-as-Judge**! Contributions of all
kinds — bug reports, features, docs, and tests — are welcome.

## Development setup

```bash
git clone https://github.com/syed-waleed-ahmed/LLM-as-Judge.git
cd LLM-as-Judge

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -e ".[ui,dev]"
```

## Before you open a pull request

Please make sure the full quality gate passes locally — the same checks run in CI:

```bash
ruff check .          # lint
mypy                  # type-check
pytest --cov=llm_judge   # tests + coverage
```

Or, with the provided `Makefile`:

```bash
make check            # runs lint, type-check, and tests
```

## Guidelines

- **Keep it tested.** New behaviour should come with tests. The suite runs fully
  offline against an in-memory fake provider — no API key or network calls.
- **Keep it typed and linted.** Code must be `ruff`-clean and `mypy`-clean.
- **Match the surrounding style.** Prefer small, focused functions with clear
  docstrings, as used throughout `src/llm_judge/`.
- **Adding a provider?** Implement the `LLMProvider` protocol in
  `src/llm_judge/providers/` — no changes to the judging logic should be needed.
- **Update docs.** If you change behaviour or configuration, update the
  `README.md`, `.env.example`, and `CHANGELOG.md` accordingly.

## Commit messages

Write clear, imperative commit messages (e.g. "Add OpenAI provider"). Group
related changes into a single, coherent commit where practical.

## Reporting bugs

Open an issue with a minimal reproduction, the expected vs. actual behaviour,
and your Python version. Never include real API keys in issues or logs.
