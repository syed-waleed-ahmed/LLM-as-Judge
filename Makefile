.PHONY: install test lint type security docs check format format-check run serve cli clean

install:  ## Install the package with every extra plus dev and docs tooling (editable)
	pip install -e ".[all,dev,docs]"

test:  ## Run the test suite with coverage
	pytest --cov=llm_judge --cov-report=term-missing

lint:  ## Lint with ruff
	ruff check .

type:  ## Type-check with mypy
	mypy

security:  ## Static security analysis with Bandit (as in CI)
	bandit -q -c pyproject.toml -r src app.py dashboard

docs:  ## Build the documentation site, failing on warnings (as in CI)
	mkdocs build --strict

format:  ## Auto-format and auto-fix with ruff
	ruff format .
	ruff check --fix .

format-check:  ## Verify formatting without changing files
	ruff format --check .

check: lint format-check type security test  ## Run lint, format check, type-check, Bandit and tests

run:  ## Launch the Streamlit app
	streamlit run app.py

serve:  ## Launch the REST API on http://127.0.0.1:8000 (docs at /docs)
	llm-judge serve

cli:  ## Show the CLI help
	llm-judge --help

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist site *.egg-info
