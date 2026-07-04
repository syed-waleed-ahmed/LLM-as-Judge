# Graph Report - llm-as-judge  (2026-07-05)

## Corpus Check
- 26 files · ~31,652 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 238 nodes · 605 edges · 15 communities detected
- Extraction: 49% EXTRACTED · 51% INFERRED · 0% AMBIGUOUS · INFERRED: 306 edges (avg confidence: 0.63)
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

## God Nodes (most connected - your core abstractions)
1. `Judge` - 39 edges
2. `Settings` - 31 edges
3. `Criterion` - 30 edges
4. `ProviderError` - 29 edges
5. `JudgeResult` - 26 edges
6. `ConfigurationError` - 19 edges
7. `GroqProvider` - 19 edges
8. `CriterionScore` - 18 edges
9. `LLM provider implementations.  The judge is written against the :class:`~llm_jud` - 16 edges
10. `BatchEvaluator` - 15 edges

## Surprising Connections (you probably didn't know these)
- `get_settings()` --calls--> `test_get_settings_is_cached()`  [INFERRED]
  D:\Projects & Stuff\llm-as-judge\src\llm_judge\config.py → D:\Projects & Stuff\llm-as-judge\tests\test_config.py
- `Runtime settings sourced from the environment and ``.env``.      All values have` --uses--> `ConfigurationError`  [INFERRED]
  D:\Projects & Stuff\llm-as-judge\src\llm_judge\config.py → D:\Projects & Stuff\llm-as-judge\src\llm_judge\exceptions.py
- `Return the Groq API key or raise :class:`ConfigurationError`.          Call this` --uses--> `ConfigurationError`  [INFERRED]
  D:\Projects & Stuff\llm-as-judge\src\llm_judge\config.py → D:\Projects & Stuff\llm-as-judge\src\llm_judge\exceptions.py
- `Return a process-wide cached :class:`Settings` instance.` --uses--> `ConfigurationError`  [INFERRED]
  D:\Projects & Stuff\llm-as-judge\src\llm_judge\config.py → D:\Projects & Stuff\llm-as-judge\src\llm_judge\exceptions.py
- `Built-in evaluation criteria and helpers for resolving user selections.` --uses--> `Criterion`  [INFERRED]
  D:\Projects & Stuff\llm-as-judge\src\llm_judge\criteria.py → D:\Projects & Stuff\llm-as-judge\src\llm_judge\models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (36): _get_judge(), main(), Streamlit UI for the LLM-as-Judge framework.  Run with::      streamlit run app., Build (and cache) a Judge. Cached per bias setting to avoid re-creating clients., _render_result(), _sidebar(), ChatMessage, Provider protocol shared by all LLM backends. (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (26): BatchEvaluator, Evaluate many records concurrently against a shared :class:`Judge`., _build_parser(), _cmd_batch(), _cmd_evaluate(), _cmd_list_criteria(), main(), _parse_criteria() (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.13
Nodes (22): BatchRecord, BatchRowResult, Concurrent batch evaluation for large datasets.  Reads many task/output-A/output, Judge every record, preserving input order in the returned list.          ``prog, Write results to ``.csv`` or ``.jsonl`` based on the file extension., One input row to be judged., The outcome of judging a single :class:`BatchRecord`., Flatten into a single CSV-friendly row. (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.23
Nodes (15): FakeProvider, FlakyProvider, Raises a retryable ProviderError ``fail_times`` times, then succeeds., _judge(), test_basic_evaluation(), test_close_scores_are_a_tie(), test_empty_input_is_rejected(), test_gives_up_after_max_retries() (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.18
Nodes (14): Raised when the judge response cannot be parsed into a valid result., ResponseParsingError, _extract_json_object(), parse_judge_result(), Defensive parsing of raw LLM output into a validated :class:`JudgeResult`., Remove a surrounding ```json ... ``` fence if the model added one., Return the outermost ``{...}`` block from ``text``.      Even with JSON mode ena, Parse and validate raw model output into a :class:`JudgeResult`.      Raises :cl (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (14): BaseSettings, Runtime settings sourced from the environment and ``.env``.      All values have, Return the Groq API key or raise :class:`ConfigurationError`.          Call this, Settings, hermetic_settings(), PositionBiasedProvider, Always scores whatever is in position A higher — a pure position bias., Settings that ignore the developer's real ``.env`` file. (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.21
Nodes (10): BaseModel, Criterion, A single dimension to evaluate outputs against., build_system_prompt(), build_user_prompt(), Prompt construction for the judge.  The prompts are deliberately explicit about, Build the system prompt describing the judging task and output schema., Build the user message containing the task and the two outputs. (+2 more)

### Community 7 - "Community 7"
Cohesion: 0.35
Nodes (5): Raised when user-supplied input fails validation before a request., ValidationError, Judge, Average two runs (original + swapped) into one bias-mitigated result., Judge ``output_a`` against ``output_b`` for ``task_description``.          ``cri

### Community 8 - "Community 8"
Cohesion: 0.24
Nodes (8): CriterionScore, The judge's assessment of both outputs on one criterion., Return a copy with A/B roles exchanged.          Used by position-bias mitigatio, test_criterion_is_frozen(), test_criterion_rejects_invalid_scale(), test_criterion_score_swapped(), test_judge_result_requires_at_least_one_criterion(), test_winner_swapped()

### Community 9 - "Community 9"
Cohesion: 0.25
Nodes (5): _build_default_provider(), _clamp(), The core judging engine.  :class:`Judge` orchestrates a single pairwise evaluati, Clamp scores to each criterion's range and recompute winners.          LLMs occa, _winner_from_scores()

### Community 10 - "Community 10"
Cohesion: 0.28
Nodes (6): Built-in evaluation criteria and helpers for resolving user selections., Return a :class:`Criterion` for ``name``.      If ``name`` matches a preset it i, resolve_criterion(), test_resolve_custom_builds_new_criterion(), test_resolve_custom_uses_supplied_description(), test_resolve_preset_returns_registered_criterion()

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (5): Enum, Typed data models for evaluation inputs and results.  Using Pydantic gives us th, Which output a judge preferred., Return the winner as if outputs A and B had been swapped., Winner

### Community 12 - "Community 12"
Cohesion: 0.32
Nodes (6): JudgeResult, The complete result of judging a pair of outputs.      This is the model the LLM, Return the mean score across criteria for ``"A"`` or ``"B"``., _FakeJudge, Stand-in for Judge that records inputs and returns a fixed result., test_judge_result_score_for_averages_criteria()

### Community 13 - "Community 13"
Cohesion: 0.29
Nodes (6): LLMProvider, Minimal interface the judge needs from an LLM backend.      Implementations are, Return the assistant's response content as a JSON string.          Implementatio, Call the provider, retrying only on retryable :class:`ProviderError`s., Evaluate and compare pairs of model outputs with an LLM judge., Protocol

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **20 isolated node(s):** `Exception hierarchy for the llm_judge package.  A single base exception (:class:`, `Base class for all errors raised by llm_judge.`, `Raised when the library is misconfigured (e.g. missing API key).`, `Raised when the underlying LLM provider fails or is unreachable.      ``retryabl`, `Raised when the judge response cannot be parsed into a valid result.` (+15 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 14`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Judge` connect `Community 7` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 8`, `Community 9`, `Community 11`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.275) - this node is a cross-community bridge._
- **Why does `Settings` connect `Community 5` to `Community 0`, `Community 1`, `Community 3`, `Community 7`, `Community 9`, `Community 12`, `Community 13`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `ProviderError` connect `Community 0` to `Community 3`, `Community 4`, `Community 5`, `Community 7`, `Community 9`, `Community 13`?**
  _High betweenness centrality (0.132) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `Judge` (e.g. with `BatchRecord` and `BatchRowResult`) actually correct?**
  _`Judge` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 26 inferred relationships involving `Settings` (e.g. with `ConfigurationError` and `Judge`) actually correct?**
  _`Settings` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `Criterion` (e.g. with `BatchRecord` and `BatchRowResult`) actually correct?**
  _`Criterion` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 25 inferred relationships involving `ProviderError` (e.g. with `Judge` and `The core judging engine.  :class:`Judge` orchestrates a single pairwise evaluati`) actually correct?**
  _`ProviderError` has 25 INFERRED edges - model-reasoned connections that need verification._