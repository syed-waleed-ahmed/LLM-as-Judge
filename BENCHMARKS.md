# Benchmarks

Two kinds of numbers live here, and they shouldn't be confused:

1. **Framework performance**: what the library itself costs, measured offline.
2. **Sample-benchmark runs**: judgments on the small bundled synthetic
   datasets. They show the pipeline working. They are **not** evidence about
   real models: the datasets have 5-6 rows, and their "systems" are neutral
   labels.

Every number below was measured, and each table states how. Re-run
`python scripts/perf_baseline.py` or `llm-judge benchmark <name>` to reproduce
them on your machine.

## 1. Framework performance

Measured on 2026-11-10 with `scripts/perf_baseline.py` (llm-judge 1.8.0,
Python 3.10.11, Windows, 12 logical CPUs), offline, with a simulated provider.

**Library overhead per pairwise judgment** (3 criteria). This is prompt
building, parsing, validation, scoring and diagnostics, with a provider that
answers instantly:

| median | p95 |
|---|---|
| 0.23 ms | 0.47 ms |

**Dataset throughput** (200 rows, simulated 50 ms provider latency):

| workers | rows/s | speed-up |
|---|---|---|
| 1 | 16.0 | 1.0x |
| 4 | 64.0 | 4.0x |
| 8 | 128.8 | 8.1x |
| 16 | 247.1 | 15.5x |

Throughput scales almost linearly with workers until the provider's rate limit
becomes the bottleneck. The single-worker rate is below the theoretical
20 rows/s because Windows' timer granularity stretches each simulated 50 ms
sleep.

**Position-bias mitigation** (two orderings per judgment, 50 ms latency):

| execution | time per judgment |
|---|---|
| sequential | 125 ms |
| parallel (default) | 63 ms |

## 2. Sample-benchmark runs

### Instruction following: live judge vs. mock heuristic

`llm-judge benchmark instruction_following` was run on 2026-11-10. In the
sample, `system-alpha`'s outputs follow the task's explicit format
constraints, and `system-beta`'s break them.

| judge | system-alpha (mean score, 0-1) | system-beta (mean score, 0-1) |
|---|---|---|
| Groq `openai/gpt-oss-120b` (live) | 1.000 | 0.074 |
| Mock heuristic (offline, not an LLM) | 0.185 | 0.370 |

The live judge separates constraint-following from constraint-breaking
outputs cleanly. The mock heuristic gets them backwards, because word-overlap
features can't check format constraints. That's exactly why mock results are
labelled as not coming from an LLM. Six rows is far too few for any
statistical claim. Treat this as a smoke test of the pipeline.

### Reproducing

```bash
llm-judge datasets samples                         # list the bundled datasets
llm-judge benchmark instruction_following          # configured live judge
llm-judge --provider mock benchmark safety         # offline, fully deterministic
llm-judge benchmark safety --experiment bench      # record the run for comparisons
```
