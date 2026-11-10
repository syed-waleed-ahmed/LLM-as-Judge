# API reference

The same engine has four interfaces. This page lists every entry point; the
docstrings and the OpenAPI schema (`/docs` on a running server) hold the full
details. A test keeps this page in sync with the code, failing if a CLI
command or REST route is added without being documented here.

## Python library

```python
from llm_judge import Judge

judge = Judge()  # provider and settings from env / .env
judge = Judge(provider=..., settings=...)
```

| Method | Returns | Purpose |
|---|---|---|
| `judge.evaluate(task_description, output_a, output_b, criteria, *, candidate_a_model=None, candidate_b_model=None, instructions=None)` | `JudgeResult` | Pairwise comparison |
| `judge.evaluate_pointwise(task_description, output, criteria="Helpfulness", *, reference=None, candidate_model=None, instructions=None)` | `PointwiseResult` | Grade one output |
| `judge.evaluate_reference(task_description, output, reference, criteria=None, *, candidate_model=None, instructions=None)` | `PointwiseResult` | Grade against an expected answer |
| `judge.rank(task_description, candidates, criteria="Helpfulness", *, strategy="round_robin", candidate_models=None, max_workers=None, instructions=None)` | `RankingResult` | Leaderboard of 2+ candidates |
| `judge.evaluate_rag(question, contexts, answer, *, reference=None)` | `RagResult` | RAG faithfulness, relevance, context precision/recall |
| `judge.evaluate_agent(trace, criteria=None, *, candidate_model=None)` | `AgentResult` | Agent trajectory checks and grade |

Criteria may be preset names, custom names, or `Criterion(name, description,
scale_min=1, scale_max=10, weight=1.0)` objects.

Each method has an async twin with the same arguments and result:
`aevaluate`, `aevaluate_pointwise`, `aevaluate_reference`, `arank`,
`aevaluate_rag` and `aevaluate_agent`. Each runs the method in a worker thread
(`asyncio.to_thread`), so it doesn't block the event loop.

`Judge(provider=None, settings=None, *, cache=None)`: `cache` is an optional
`llm_judge.cache.ResultCache`. With `JUDGE_CACHE_SIZE > 0`, judges share a
process-wide cache of that size. Only temperature-0 results are cached, hits
carry `metadata["cache_hit"] = True`, and `judge.without_cache()` returns an
uncached copy.

`Judge(..., embedder=None)`: an optional `llm_judge.providers.embeddings`
backend (`OllamaEmbeddings`, `OpenAICompatibleEmbeddings`, or anything with
`model` and `embed(texts)`). By default it comes from `JUDGE_EMBEDDINGS_*`. With
one, reference grading fills `reference_metrics.semantic_similarity` (cosine,
-1 to 1) and `embedding_model`. If embedding fails, the grade still succeeds
and carries a `semantic_similarity_unavailable` warning.

`instructions` is optional guidance for the judge (for example "Penalise answers
that state facts without a source"). It goes into the system prompt as evaluator
guidance, outside the untrusted-content markers, and can't change the response
format. It is recorded in `result.metadata["instructions"]`, and
`DatasetRunner(..., instructions=...)` records it in the experiment run's
configuration, so comparisons flag runs judged with different guidance. Text
longer than `JUDGE_MAX_INSTRUCTIONS_CHARS` is truncated with an
`input_truncated` warning. The mock judge, a text heuristic, ignores
instructions.

| Module | Key entry points |
|---|---|
| `llm_judge.datasets` | `load_dataset`, `validate_dataset`, `DatasetRunner`, `write_outcomes`, `summarize`, `load_sample`, `sample_records`, `split_records` |
| `llm_judge.experiments` | `ExperimentStore(location)` (a SQLite path, `:memory:` or a `postgresql://` URL; `.backend`, and `.location` with the password redacted), `record_dataset_run`, `compare_runs`, `run_report`, `confidence_interval` |
| `llm_judge.reliability` | `measure_consistency`, `judge_agreement`, `cohen_kappa`, `spearman` |
| `llm_judge.providers` | `GroqProvider`, `OllamaProvider`, `OpenAICompatibleProvider`, `MockProvider`, `QuotaProvider`, `CallBudget`, `build_provider`, `register_provider`, `provider_names` |
| `llm_judge.agents` | `AgentTrace`, `AgentStep`, `ToolSpec`, `check_trace`, `render_trace` |
| `llm_judge.rag` | `RagResult`, `average_precision` |
| `llm_judge.metrics` | `token_f1`, `exact_match`, `reference_metrics` |
| `llm_judge.api` | `create_app` (FastAPI, `[api]` extra) |

Every library error subclasses `LLMJudgeError`:

- `ConfigurationError`
- `ProviderError`, which has a `.retryable` flag
- `ResponseParsingError`, which carries `.raw_response`
- `ValidationError`, the parent of `DatasetError`

## Command line

Global options: `--provider {groq,ollama,openai_compatible,mock}` and
`--log-level`. Run `llm-judge <command> --help` for all flags.

| Command | Purpose |
|---|---|
| `llm-judge evaluate` | Pairwise comparison (`--weights`, `--model-a/-b`, `--mitigate-bias`, `--json`) |
| `llm-judge grade` | Pointwise grade; with `--reference`, reference grading |
| `llm-judge rank` | Rank candidates from `--candidates-file` (`--strategy round_robin\|pointwise`) |
| `llm-judge rag` | RAG evaluation (`--question`, `--context` repeated or `--contexts-file`, `--answer`, `--reference`) |
| `llm-judge agent` | Agent evaluation from `--trace-file` |
| `llm-judge consistency` | Repeated-judgment consistency (`--samples`, `--temperature`) |
| `llm-judge batch` | Dataset evaluation with checkpoints (`--resume`, `--sample`, `--limit`, `--skip-invalid`, `--experiment`) |
| `llm-judge benchmark` | Run a bundled sample or a dataset file and print per-model statistics |
| `llm-judge datasets` | `validate`, `samples`, `export` |
| `llm-judge experiments` | `list`, `runs`, `show`, `compare` (`--fail-on-regression`), `agreement`, `report` (Markdown, optional `--baseline`), `export`, `delete` |
| `llm-judge serve` | Run the REST API (`[api]` extra; localhost by default) |
| `llm-judge providers` | List providers and whether each is configured |
| `llm-judge health` | Check that the active provider is reachable (no inference) |
| `llm-judge list-criteria` | List criteria presets |
| `llm-judge version` | Print the version |

Exit codes:

- `0`: success.
- `1`: some rows failed, a health check failed, or `compare --fail-on-regression` found a regression.
- `2`: configuration or validation error.
- `130`: the run was interrupted (resume it with `--resume`).

## REST API (`/api/v1`)

Start it with `llm-judge serve`. Interactive OpenAPI docs are at `/docs`.
Errors use one envelope:
`{"error": {"code", "message", "request_id", "details"}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness (no auth, no rate limit) |
| GET | `/api/v1/system/status` | Version, provider health, limits in effect |
| GET | `/api/v1/providers` | Providers and their configuration status |
| GET | `/api/v1/criteria` | Criteria presets |
| POST | `/api/v1/evaluate/pairwise` | Pairwise comparison |
| POST | `/api/v1/evaluate/pointwise` | Pointwise grade |
| POST | `/api/v1/evaluate/reference` | Reference grading |
| POST | `/api/v1/evaluate/rank` | Ranking |
| POST | `/api/v1/evaluate/rag` | RAG evaluation |
| POST | `/api/v1/evaluate/agent` | Agent trajectory evaluation |
| POST | `/api/v1/analysis/consistency` | Repeated-judgment consistency (at most 10 samples) |
| POST | `/api/v1/evaluate/batch` | Small synchronous batch |
| POST | `/api/v1/jobs` | Queue a dataset job (`Idempotency-Key` supported) |
| GET | `/api/v1/jobs` | List all jobs (admin) |
| GET | `/api/v1/jobs/{job_id}` | Job status and summary |
| DELETE | `/api/v1/jobs/{job_id}` | Cancel a job |
| GET | `/api/v1/jobs/{job_id}/rows` | Paginated job results |
| POST | `/api/v1/datasets/validate` | Validate rows or file content |
| GET | `/api/v1/benchmarks` | Bundled sample benchmarks |
| POST | `/api/v1/benchmarks/{name}/jobs` | Run a sample benchmark as a job |
| GET | `/api/v1/experiments` | Experiments |
| GET | `/api/v1/experiments/{experiment_name}` | One experiment and a page of its runs |
| GET | `/api/v1/runs` | Recorded runs (paginated) |
| GET | `/api/v1/runs/compare` | Compare two runs (regression check) |
| GET | `/api/v1/runs/agreement` | Agreement between two runs (e.g. two judges) |
| GET | `/api/v1/runs/{run_id}` | Run details |
| DELETE | `/api/v1/runs/{run_id}` | Delete a run (admin) |
| GET | `/api/v1/runs/{run_id}/report` | Markdown report (`?baseline=` adds a comparison) |
| GET | `/api/v1/runs/{run_id}/rows` | Paginated run rows |
| GET | `/api/v1/statistics` | Store-wide counts |

The pairwise, pointwise, reference, rank, batch, job and benchmark-job bodies
accept an optional `instructions` string: guidance for the judge, as described
for the Python API above.

Authentication: set `JUDGE_API_KEYS` and send `X-API-Key: <key>` or
`Authorization: Bearer <key>`. Admin routes additionally need `X-Admin-Key`,
and they're disabled unless `JUDGE_API_ADMIN_KEY` is set.
