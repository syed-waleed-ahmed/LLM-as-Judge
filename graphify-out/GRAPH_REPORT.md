# Graph Report - llm-as-judge  (2026-11-10)

## Corpus Check
- 90 files · ~226,193 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1355 nodes · 5313 edges · 30 communities detected
- Extraction: 38% EXTRACTED · 62% INFERRED · 0% AMBIGUOUS · INFERRED: 3284 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]

## God Nodes (most connected - your core abstractions)
1. `Criterion` - 189 edges
2. `Settings` - 182 edges
3. `Judge` - 153 edges
4. `JudgeResult` - 152 edges
5. `PointwiseResult` - 143 edges
6. `ProviderError` - 106 edges
7. `ValidationError` - 88 edges
8. `EvaluationWarning` - 81 edges
9. `ResponseParsingError` - 79 edges
10. `FakeProvider` - 74 edges

## Surprising Connections (you probably didn't know these)
- `Datasets: upload or pick a dataset, validate it, run it, download the results.` --uses--> `LLMJudgeError`  [INFERRED]
  dashboard/views/datasets.py → src/llm_judge/exceptions.py
- `main()` --calls--> `test_cli_empty_database_and_unknown_run()`  [INFERRED]
  src/llm_judge/cli.py → tests/test_experiments.py
- `main()` --calls--> `test_consistency_exit_codes_and_temperature_range()`  [INFERRED]
  src/llm_judge/cli.py → tests/test_review_fixes.py
- `main()` --calls--> `test_cli_usage_errors_exit_2_and_respect_log_level()`  [INFERRED]
  src/llm_judge/cli.py → tests/test_review_fixes.py
- `Settings` --uses--> `Phase 8 through every interface: CLI, REST API and dashboard (offline mock judge`  [INFERRED]
  src/llm_judge/config.py → tests/test_phase8_interfaces.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (157): BaseSettings, Runtime settings sourced from the environment and ``.env``.      All values have, Settings, BarrierProvider, FakeProvider, FlakyProvider, FnProvider, _hermetic_environment() (+149 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (161): AgentResult, ChatMessage, LLMProvider, A single chat message in the OpenAI/Groq chat-completions format., Minimal interface the judge needs from an LLM backend.      Implementations are, Return the assistant's response content as a JSON string.          Implementatio, Built-in evaluation criteria and helpers for resolving user selections., describe_candidate() (+153 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (155): LLM provider implementations.  The judge is written against the :class:`~llm_jud, DatasetError, detect_format(), _document_rows(), iter_rows(), _no_alias_loader(), Bounded, incremental readers for CSV, JSON, JSONL and YAML datasets.  Rows are p, Rows from a whole-document format (JSON or YAML): a list, or {"records": [...]}. (+147 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (131): AgentMetrics, AgentStep, AgentTrace, check_trace(), _clip(), Agent trajectory evaluation: deterministic trace checks plus a judged grade.  An, A compact, readable text form of the trajectory for the judge.      With ``max_c, Evaluate an agent trajectory; see :meth:`Judge.evaluate_agent`. (+123 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (108): Build (and cache) a Judge. Cached per bias setting to avoid re-creating clients., ProviderHealth, Provider protocol shared by all LLM backends., Result of a provider health check (no inference is performed)., Thread-safe cumulative request and token counters for one provider.      Token c, Trim provider error bodies before they go into exception messages., shorten(), UsageTracker (+100 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (102): _add_dataset_run_options(), _build_criteria(), _build_parser(), _check_output_path(), CliError, _cmd_agent(), _cmd_batch(), _cmd_benchmark() (+94 more)

### Community 6 - "Community 6"
Cohesion: 0.05
Nodes (55): _categorical_y(), criterion_comparison(), delta_bars(), _layout(), ranked_bars(), Theme-aware Plotly charts for the dashboard.  Colors come from the reference dat, Keep category order even for ids like "1", "2" (Plotly would make them numbers)., Grouped horizontal bars: Output A vs Output B on each criterion.      ``scale`` (+47 more)

### Community 7 - "Community 7"
Cohesion: 0.09
Nodes (44): _cmd_experiments(), _print_comparison(), _comparability(), compare_runs(), confidence_interval(), _criteria_means(), _metric(), Compare two stored runs and detect regressions.  Rows are paired by id. The per- (+36 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (43): create_app(), _examples(), _fill_pairwise(), _fill_reference(), _reference_examples(), _BodyTooLarge, client_identity(), key_matches() (+35 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (29): BatchEvaluator, BatchRecord, BatchRowResult, _optional_str(), Concurrent batch evaluation for large datasets.  Reads many task/output-A/output, Evaluate many records concurrently against a shared :class:`Judge`.      With po, Judge every record, preserving input order in the returned list.          ``prog, Write results to ``.csv`` or ``.jsonl`` based on the file extension. (+21 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (27): criteria_picker(), Preset multiselect + custom criteria + per-criterion weights., Return a :class:`Criterion` for ``name``.      If ``name`` matches a preset it i, resolve_criterion(), Return a copy with A/B roles exchanged.          Used by position-bias mitigatio, Return the unweighted mean raw score across criteria for ``"A"`` or ``"B"``., Return a copy with every A/B role (scores, text, winners) exchanged., Exchange "Output A" and "Output B" references in free text.      Used when a jud (+19 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (27): Defences for CSV files that people open in spreadsheet software.  Result files c, Neutralise a text value that a spreadsheet could interpret as a formula., Apply :func:`safe_cell` to every value of every row., safe_cell(), safe_rows(), parse_judge_result(), Parse and validate raw pairwise-judge output into a :class:`JudgeResult`.      O, test_bandit_reports_no_unreviewed_issues() (+19 more)

### Community 12 - "Community 12"
Cohesion: 0.23
Nodes (18): _judge(), _rag_payload(), test_average_precision(), test_clean_trace_has_no_findings(), test_evaluate_agent_combines_checks_and_a_judged_grade(), test_evaluate_agent_with_reference_and_custom_criteria(), test_rag_edge_cases(), test_rag_flags_injection_in_retrieved_contexts() (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.22
Nodes (14): exact_match(), Deterministic, LLM-free text metrics for reference-based evaluation.  These are, Lower-cased word tokens (punctuation dropped)., True when both texts have identical tokens (ignoring case and punctuation)., SQuAD-style bag-of-words F1 between ``prediction`` and ``reference``.      Retur, Compute all reference metrics for one prediction., reference_metrics(), token_f1() (+6 more)

### Community 14 - "Community 14"
Cohesion: 0.19
Nodes (11): main(), Print the CHANGELOG.md section for one version, for use as GitHub Release notes., Return the body of the ``## [version]`` section, without its heading., section(), The release workflow publishes CHANGELOG sections as GitHub Release notes., Bumping the version without a changelog section would fail the release., test_current_version_has_release_notes(), test_last_section_stops_at_link_references() (+3 more)

### Community 15 - "Community 15"
Cohesion: 0.28
Nodes (7): check(), idle(), nav(), no_exceptions(), Real-browser end-to-end test of the dashboard in public-demo mode.  Drives every, Wait until Streamlit has finished running the script., shot()

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (1): Benchmarks: run a bundled sample dataset and compare the systems in it.

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): About: how this works, limits in effect, privacy, and how to run it yourself.

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Experiments: browse recorded runs, compare two, and check for regressions.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Overview: what this is, which judge is active, and recent activity.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Which output a judge preferred.

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Return the winner as if outputs A and B had been swapped.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): A single dimension to evaluate outputs against.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Return a copy with A/B roles exchanged.          Used by position-bias mitigatio

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): The complete result of judging a pair of outputs.      This is the model the LLM

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Return the mean score across criteria for ``"A"`` or ``"B"``.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): A single chat message in the OpenAI/Groq chat-completions format.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Minimal interface the judge needs from an LLM backend.      Implementations are

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Return the assistant's response content as a JSON string.          Implementatio

## Knowledge Gaps
- **79 isolated node(s):** `Theme`, `Theme-aware Plotly charts for the dashboard.  Colors come from the reference dat`, `Keep category order even for ids like "1", "2" (Plotly would make them numbers).`, `Grouped horizontal bars: Output A vs Output B on each criterion.      ``scale```, `Single-series horizontal bars, best at the top (leaderboards, model stats).` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 17`** (2 nodes): `About: how this works, limits in effect, privacy, and how to run it yourself.`, `about.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (2 nodes): `experiments.py`, `Experiments: browse recorded runs, compare two, and check for regressions.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (2 nodes): `overview.py`, `Overview: what this is, which judge is active, and recent activity.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Which output a judge preferred.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Return the winner as if outputs A and B had been swapped.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `A single dimension to evaluate outputs against.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Return a copy with A/B roles exchanged.          Used by position-bias mitigatio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `The complete result of judging a pair of outputs.      This is the model the LLM`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Return the mean score across criteria for ``"A"`` or ``"B"``.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `A single chat message in the OpenAI/Groq chat-completions format.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Minimal interface the judge needs from an LLM backend.      Implementations are`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Return the assistant's response content as a JSON string.          Implementatio`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 10`, `Community 11`, `Community 12`?**
  _High betweenness centrality (0.138) - this node is a cross-community bridge._
- **Why does `Judge` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 12`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `Criterion` connect `Community 3` to `Community 0`, `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 184 inferred relationships involving `Criterion` (e.g. with `ToolSpec` and `AgentStep`) actually correct?**
  _`Criterion` has 184 INFERRED edges - model-reasoned connections that need verification._
- **Are the 177 inferred relationships involving `Settings` (e.g. with `Shared state, safeguards and components for the Streamlit dashboard.  Two operat` and `Expose Streamlit secrets (Streamlit Community Cloud) as environment settings.`) actually correct?**
  _`Settings` has 177 INFERRED edges - model-reasoned connections that need verification._
- **Are the 134 inferred relationships involving `Judge` (e.g. with `ToolSpec` and `AgentStep`) actually correct?**
  _`Judge` has 134 INFERRED edges - model-reasoned connections that need verification._
- **Are the 147 inferred relationships involving `JudgeResult` (e.g. with `CliError` and `Command-line interface for llm_judge.  Examples::      llm-judge list-criteria`) actually correct?**
  _`JudgeResult` has 147 INFERRED edges - model-reasoned connections that need verification._