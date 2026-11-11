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

This project's demo runs at <https://llm-as-judge.streamlit.app/>, deployed exactly as described here.
To run your own copy:

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

Updates deploy automatically when you push to `main`. `requirements.txt`
installs the package *editable* (`-e .[ui]`) on purpose. Community Cloud
reinstalls requirements only when that file changes, so a regular install
would keep running the package version from the first deploy while the
dashboard code moves on. If the app ever reports that the installed
`llm_judge` package doesn't match its code, choose **Reboot app** in the
app's menu on <https://share.streamlit.io>.

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
- **Ephemeral storage.** In demo mode nothing is written to disk. That's
  deliberate: visitors' runs stay in their session. Without demo mode, the SQLite
  database would live on the container's disk and be lost on restart. To keep
  runs, set `JUDGE_DB_PATH` to a PostgreSQL or Supabase URL (see below).
- **Model retirement.** Groq retires models over time. If live judging
  starts failing with a "model not found" error, set `GROQ_MODEL` to a current
  model. Run `llm-judge health` locally to check one.

### Automated browser check

`scripts/e2e_dashboard.py` drives every page in a real browser (Playwright with
the installed Microsoft Edge). It runs with both the live and the mock judge,
exercises the demo quota until it is exhausted, and fails on any Streamlit
exception or browser console error. See the script's docstring for setup. It
was run against a fresh clone installed only from `requirements.txt`, with
configuration from `.streamlit/secrets.toml` exactly as on Community Cloud,
and all 48 checks passed.

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

### Persistent runs on Supabase (optional, free)

To share recorded runs between the CLI, the REST API and a self-hosted dashboard,
or across machines, store them in PostgreSQL. Supabase's free plan provides one:

1. Create a project at <https://supabase.com>. The store connects to Postgres
   directly, so you can untick **Enable Data API** when creating it; that keeps
   the run tables out of Supabase's public REST API.
2. Open **Connect** → **Direct** and copy a pooler connection string: the
   **Transaction pooler** (port 6543) for short-lived processes, such as the
   CLI, CI jobs and a Streamlit app, or the **Session pooler** (port 5432) for a
   long-running REST API server. Both work over IPv4; the "Direct connection"
   host is IPv6-only on the free plan.
3. Install the extra: `pip install -e ".[postgres]"`, or `.[all]`.
4. Set `JUDGE_DB_PATH` to the connection string, with your database password
   filled in, in `.env`, Streamlit secrets or the environment. Percent-encode
   special characters in the password (`@` becomes `%40`). The string is a
   secret, so treat it like an API key.

The tables are created on first use. The store disables server-side prepared
statements, which transaction poolers don't support. The location is shown with
the password redacted everywhere. CI runs the store's tests against a real
PostgreSQL 16 server, and they have also passed against a free Supabase project
through both poolers. Free Supabase projects pause after a week of inactivity,
so resume yours in the Supabase dashboard if connections start failing.

To run the store's tests against your own database, set
`JUDGE_TEST_POSTGRES_URL` and run `pytest tests/test_store_backends.py`. The
tests drop and recreate the store's tables, so use a database that holds
nothing else.

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

## 4. Documentation site (GitHub Pages)

`mkdocs.yml` builds a documentation site from the README and project documents.
CI builds it strictly on every push, and the `Docs` workflow deploys it on every
push to `main`. This repository's site is live at
<https://syed-waleed-ahmed.github.io/LLM-as-Judge/>. To publish it from a fork:

1. In the repository, go to **Settings → Pages → Build and deployment → Source**
   and choose **GitHub Actions**. This is a one-time step.
2. Merge to `main`. The `Docs` workflow then deploys the site to
   `https://<user>.github.io/LLM-as-Judge/`.

Preview it locally with `pip install -e ".[docs]"` and then `mkdocs serve`.

## 5. Releases (GitHub and PyPI)

Releases need no stored API token. PyPI uses trusted publishing, and the package
is published as [`llm-as-judge-platform`](https://pypi.org/project/llm-as-judge-platform/).

1. **One-time setup on PyPI.** Under *Publishing*, add a **trusted publisher**
   for this repository, with workflow `release.yml` and environment `pypi`.
   Before the first upload, use a *pending* publisher for the project name
   `llm-as-judge-platform`.
2. **Bump the version.** Update `__version__` in `src/llm_judge/__init__.py`,
   and move the changelog's **Unreleased** entries under the new version's
   heading. Then merge to `main`.
3. **Release.** Open **Actions → Release → Run workflow**, enter the version
   (for example `2.3.0`), and keep "Publish to PyPI" ticked. To release an
   earlier commit, enter its SHA as the target.

The workflow checks that the version and the changelog match. It then builds and
checks the distribution, creates the tag and the GitHub Release (using the
changelog section as the notes), and publishes to PyPI. Pushing a tag such as
`v2.3.0` also releases; that path publishes to PyPI only when the repository
variable `PUBLISH_TO_PYPI` is `true`. Re-running a release updates it rather
than failing.

CI's `package` job already builds the wheel on every push and smoke-tests it
in a clean environment, so a release only runs steps that are already checked
on every push.

## 6. Local models with Ollama

```bash
ollama serve
ollama pull llama3.2
JUDGE_PROVIDER=ollama llm-judge health
JUDGE_PROVIDER=ollama streamlit run app.py
```

Small local models are weaker judges, so expect noisier scores. Prefer a
judge from a different model family than the outputs you're judging.
