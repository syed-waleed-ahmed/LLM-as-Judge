# Deployment

Everything here runs at **$0**: Streamlit Community Cloud for the public
dashboard, Groq's free plan (or the offline mock judge) for inference, and your
own machine for the REST API and local models. No step needs a credit card.

| Component | Where | Cost | Notes |
|---|---|---|---|
| Dashboard (`app.py`) | Streamlit Community Cloud | Free | Public demo; sleeps when idle |
| Live judge | Groq free plan | Free | Rate-limited; capped further by demo quotas |
| Offline judge | Built in (`mock`) | Free | Always available; clearly labelled, not an LLM |
| REST API (`llm-judge serve`) | Your machine / your server | Free | No always-free API host is assumed |
| Local models | Ollama on your machine | Free | `JUDGE_PROVIDER=ollama` |

## 1. Public dashboard on Streamlit Community Cloud

**Prerequisites:** the repository on GitHub, and a free Streamlit Community
Cloud account (sign in with GitHub at <https://share.streamlit.io>).

1. Click **Create app**, then **Deploy a public app from GitHub**.
2. Pick the repository, the branch (`main`), and **Main file path: `app.py`**.
   Under **Advanced settings**, choose Python 3.12.
3. In **Advanced settings, Secrets**, paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with
   your real key:

   ```toml
   GROQ_API_KEY = "gsk_..."
   JUDGE_DEMO_MODE = "true"
   JUDGE_DEMO_SESSION_CALLS = "25"
   JUDGE_DEMO_DAILY_CALLS = "500"
   ```

   You can leave `GROQ_API_KEY` out. The demo then runs with the offline mock
   judge only, and that's fully functional.
4. Click **Deploy**. Community Cloud installs `requirements.txt`, which
   installs this package with its `[ui]` extra (Streamlit and Plotly).

The dashboard copies `JUDGE_*` and `GROQ_*` secrets into the environment at
start-up, so every setting in [`.env.example`](.env.example) can also be set
as a secret.

### What demo mode protects

`JUDGE_DEMO_MODE=true` turns on safeguards for an anonymous audience:

| Safeguard | Setting | Default |
|---|---|---|
| Live judge calls per visitor session | `JUDGE_DEMO_SESSION_CALLS` | 25 |
| Live judge calls per day, all visitors | `JUDGE_DEMO_DAILY_CALLS` | 500 |
| Characters per input | `JUDGE_DEMO_MAX_INPUT_CHARS` | 4,000 |
| Rows per dataset run | `JUDGE_DEMO_MAX_ROWS` | 25 |
| Ranking candidates | `JUDGE_DEMO_MAX_CANDIDATES` | 5 |
| Upload size | `.streamlit/config.toml` (`maxUploadSize`) | 5 MB |
| Storage | session-only (in memory) | nothing persisted or shared |

Quotas are enforced at the provider layer, so every request counts, including
retries, repair attempts and both bias-mitigation orderings. When a budget
runs out, visitors are offered the mock judge instead of an error.

Set the daily budget comfortably below your Groq plan's limits. Those limits
vary by model and change over time, so check
<https://console.groq.com/settings/limits>. The daily counter lives in the app
process, so it resets if Community Cloud restarts the app. Treat it as abuse
protection, not billing-grade accounting.

### Free-tier behaviour to expect

- **Sleeping.** Community Cloud puts idle apps to sleep. The first visitor
  after a quiet period wakes it, which takes a little while, and session data
  starts empty.
- **Ephemeral storage.** In demo mode nothing is written to disk. Without demo
  mode, the SQLite database would live on the container's disk and be lost on
  restart.
- **Model retirement.** Groq retires models over time. If live judging
  starts failing with a "model not found" error, set `GROQ_MODEL` to a current
  model. Run `llm-judge health` locally to check one.

### Smoke test after deploying

1. **Overview** loads. The sidebar shows *Live LLM (groq)* when a key is
   configured, and only *Mock* otherwise.
2. **Playground → Compare two outputs.** Load the *Summary* example, then
   click Compare. You should see a blue `groq · <model>` badge, the scores and
   the per-criterion chart. The sidebar quota caption goes down by one (by two
   with bias mitigation on).
3. Switch the sidebar to **Mock** and compare again. You should see an orange
   "Mock judge (not an LLM)" badge.
4. **Benchmarks:** run `safety`. **Experiments:** run it a second time and
   compare the two runs. The verdict is usually "No significant change".
5. **About** lists the demo limits that are in effect.

## 2. Running the dashboard locally

```bash
pip install -e ".[ui]"
streamlit run app.py
```

Without `JUDGE_DEMO_MODE`, there are no quotas or input caps, and recorded
runs persist to `JUDGE_DB_PATH` (default `.llm_judge/experiments.db`).

## 3. REST API (local or self-hosted)

```bash
pip install -e ".[api]"
llm-judge serve                                   # http://127.0.0.1:8000/docs
JUDGE_API_KEYS=change-me llm-judge serve --host 0.0.0.0 --port 8000
```

Before exposing the API beyond localhost:
- set `JUDGE_API_KEYS`;
- keep `JUDGE_API_ADMIN_KEY` unset unless you need the admin routes;
- tune the `JUDGE_API_*` limits (rate, body size, concurrency, rows, queue).

This project doesn't claim a free public host for the API. Run it on
infrastructure you control, and put it behind TLS.

## 4. Local models with Ollama

```bash
ollama serve
ollama pull llama3.2
JUDGE_PROVIDER=ollama llm-judge health
JUDGE_PROVIDER=ollama streamlit run app.py
```

Small local models are weaker judges, so expect noisier scores. Prefer a
judge from a different model family than the outputs you're judging.
