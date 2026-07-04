.PHONY: install test lint type check format run cli clean

install:  ## Install the package with UI + dev extras (editable)
	pip install -e ".[ui,dev]"

test:  ## Run the test suite with coverage
	pytest --cov=llm_judge --cov-report=term-missing

lint:  ## Lint with ruff
	ruff check .

type:  ## Type-check with mypy
	mypy

format:  ## Auto-format / auto-fix with ruff
	ruff check --fix .

check: lint type test  ## Run lint, type-check, and tests

run:  ## Launch the Streamlit app
	streamlit run app.py

cli:  ## Show the CLI help
	llm-judge --help

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
