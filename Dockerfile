# Local / self-hosted image for the dashboard (default) or the REST API.
#   docker build -t llm-judge .
#   docker run -p 8501:8501 -e JUDGE_PROVIDER=mock llm-judge          # dashboard
#   docker run -p 8000:8000 -e GROQ_API_KEY=... -e JUDGE_API_KEYS=... llm-judge \
#       llm-judge serve --host 0.0.0.0 --port 8000                    # REST API
# Pass secrets at run time (-e / --env-file); never bake them into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[all]"

COPY app.py ./
COPY dashboard ./dashboard
COPY .streamlit/config.toml ./.streamlit/config.toml

USER app
ENV JUDGE_DB_PATH=/home/app/.llm_judge/experiments.db
EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
