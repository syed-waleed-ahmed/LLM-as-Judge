"""Reproducible performance baseline (offline, no API key, no network).

Measures what the library itself controls:

1. Overhead per judgment: prompt construction, parsing, validation, scoring and
   diagnostics, measured with a provider that answers instantly.
2. Concurrency: dataset throughput against a provider with a fixed simulated
   latency, at increasing worker counts.
3. Position-bias mitigation: parallel vs sequential execution of the two orderings.

Real-world latency is dominated by the LLM provider, so these numbers bound the
framework's cost; they are not end-to-end provider benchmarks.

    python scripts/perf_baseline.py            # prints a Markdown report
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

from llm_judge import Judge, __version__
from llm_judge.config import Settings
from llm_judge.datasets import DatasetRunner, PairwiseRecord

PAYLOAD = json.dumps(
    {
        "criteria": [
            {
                "criterion": name,
                "output_a_score": 8,
                "output_b_score": 5,
                "output_a_explanation": "Clear and accurate.",
                "output_b_explanation": "Vague.",
                "winner": "A",
            }
            for name in ("Helpfulness", "Conciseness", "Factual Accuracy")
        ],
        "overall_winner": "A",
        "overall_comment": "A is better.",
    }
)
CRITERIA = ["Helpfulness", "Conciseness", "Factual Accuracy"]
TASK = "Explain what an API is to a non-technical reader."
OUTPUT_A = (
    "An API is like a waiter: it takes your request to the kitchen and brings back what "
    "you ordered."
)
OUTPUT_B = "An API is an application programming interface."


class InstantProvider:
    name = "instant"
    model = "instant"

    def complete_json(self, messages: object) -> str:
        return PAYLOAD


class SlowProvider(InstantProvider):
    """Fixed simulated network latency per request."""

    name = "simulated"

    def __init__(self, latency: float) -> None:
        self.latency = latency

    def complete_json(self, messages: object) -> str:
        time.sleep(self.latency)
        return PAYLOAD


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def overhead(n: int = 500) -> tuple[float, float]:
    judge = Judge(provider=InstantProvider(), settings=_settings())
    judge.evaluate(TASK, OUTPUT_A, OUTPUT_B, CRITERIA)  # warm-up
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        judge.evaluate(TASK, OUTPUT_A, OUTPUT_B, CRITERIA)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), statistics.quantiles(samples, n=20)[-1]


def throughput(rows: int, workers: int, latency: float) -> float:
    judge = Judge(provider=SlowProvider(latency), settings=_settings())
    records = [
        PairwiseRecord(id=str(i), task_description=TASK, output_a=OUTPUT_A, output_b=OUTPUT_B)
        for i in range(rows)
    ]
    start = time.perf_counter()
    DatasetRunner(judge, CRITERIA, max_workers=workers).run(records)
    return rows / (time.perf_counter() - start)


def bias_mitigation(n: int, latency: float, parallel: bool) -> float:
    settings = _settings(mitigate_position_bias=True, parallel_bias_runs=parallel)
    judge = Judge(provider=SlowProvider(latency), settings=settings)
    start = time.perf_counter()
    for _ in range(n):
        judge.evaluate(TASK, OUTPUT_A, OUTPUT_B, CRITERIA)
    return (time.perf_counter() - start) / n * 1000


def main() -> None:
    latency = 0.05
    print(f"# Performance baseline (llm-judge {__version__})\n")
    print(
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}, "
        f"{os.cpu_count()} logical CPUs. Offline; simulated provider latency "
        f"{latency * 1000:.0f} ms.\n"
    )

    median, p95 = overhead()
    print("## Library overhead per pairwise judgment (3 criteria, instant provider)\n")
    print("| median | p95 |\n|---|---|")
    print(f"| {median:.2f} ms | {p95:.2f} ms |\n")

    print("## Dataset throughput vs workers (200 rows)\n")
    print("| workers | rows/s | speed-up |\n|---|---|---|")
    base = None
    for workers in (1, 4, 8, 16):
        rate = throughput(200, workers, latency)
        base = base or rate
        print(f"| {workers} | {rate:.1f} | {rate / base:.1f}x |")

    print("\n## Position-bias mitigation (two orderings per judgment, 20 judgments)\n")
    sequential = bias_mitigation(20, latency, parallel=False)
    concurrent = bias_mitigation(20, latency, parallel=True)
    print("| execution | time per judgment |\n|---|---|")
    print(f"| sequential | {sequential:.0f} ms |")
    print(f"| parallel | {concurrent:.0f} ms |")


if __name__ == "__main__":
    sys.exit(main())
