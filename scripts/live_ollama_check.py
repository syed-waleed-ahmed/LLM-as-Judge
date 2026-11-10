"""Live check of the Ollama and OpenAI-compatible backends against a real server.

Ollama also serves an OpenAI-compatible API under ``/v1``, so the same server
verifies the ``openai_compatible`` provider and OpenAI-compatible embeddings.

Run by the "Live Ollama" workflow, which starts the official ``ollama/ollama``
image and pulls a small judge model and an embedding model. To run it locally:

  ollama pull qwen2.5:1.5b && ollama pull nomic-embed-text
  python scripts/live_ollama_check.py [base-url] [judge-model] [embedding-model]

Small local models are weaker judges, so this checks the integration (requests,
parsing, validation, metadata and similarity ordering), not judgment quality.
Each check gets a second attempt, because small models occasionally produce
replies that the judge's repair retry can't save.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from llm_judge.config import Settings
from llm_judge.judge import Judge
from llm_judge.providers import OllamaProvider, OpenAICompatibleProvider
from llm_judge.providers.embeddings import (
    OllamaEmbeddings,
    OpenAICompatibleEmbeddings,
    cosine_similarity,
)

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
JUDGE_MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:1.5b"
EMBED_MODEL = sys.argv[3] if len(sys.argv) > 3 else "nomic-embed-text"

failures: list[str] = []


def check(label: str, action: Callable[[], str], attempts: int = 2) -> None:
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            detail = action()
        except Exception as exc:  # report and retry; the summary decides the exit code
            print(f"  attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"PASS {label} ({time.perf_counter() - started:.1f}s) - {detail}", flush=True)
        return
    print(f"FAIL {label}", flush=True)
    failures.append(label)


settings = Settings(
    _env_file=None,
    provider="ollama",
    ollama_base_url=BASE,
    ollama_model=JUDGE_MODEL,
    max_parse_retries=2,
    max_retries=2,
)
provider = OllamaProvider(JUDGE_MODEL, base_url=BASE, timeout=600)
embedder = OllamaEmbeddings(EMBED_MODEL, BASE, timeout=300)
judge = Judge(provider, settings, embedder=embedder)


def health() -> str:
    status = provider.health()
    assert status.ok, status.detail
    return status.detail


def pairwise() -> str:
    result = judge.evaluate(
        "What is the capital of France? Answer in one sentence.",
        "The capital of France is Paris.",
        "France does not have a capital city.",
        "Accuracy",
    )
    assert result.metadata["judge"] == {"provider": "ollama", "model": JUDGE_MODEL}
    score = result.criteria[0]
    a, b = score.output_a_score, score.output_b_score
    return f"winner {result.overall_winner.value}, A {a:g} vs B {b:g}"


def pointwise() -> str:
    graded = judge.evaluate_pointwise(
        "Explain what DNS does in one sentence.",
        "DNS translates human-readable domain names into IP addresses.",
        ["Accuracy", "Conciseness"],
    )
    assert graded.overall_score is not None and 0 <= graded.overall_score <= 1
    return f"overall {graded.overall_score:.2f}"


def embeddings() -> str:
    close, related, unrelated = embedder.embed(
        [
            "The cat sat on the mat.",
            "A cat was sitting on the rug.",
            "Quarterly revenue grew by twelve percent.",
        ]
    )
    near = cosine_similarity(close, related)
    far = cosine_similarity(close, unrelated)
    assert near > far, f"paraphrase {near:.3f} should beat unrelated {far:.3f}"
    return f"paraphrase {near:.3f} > unrelated {far:.3f} ({len(close)} dims)"


def reference_with_semantic_similarity() -> str:
    graded = judge.evaluate_reference(
        "When did Apollo 11 land on the Moon?",
        "Apollo 11 touched down on the Moon in July 1969.",
        "Apollo 11 landed on the Moon on 20 July 1969.",
    )
    metrics = graded.reference_metrics
    assert metrics is not None and metrics.semantic_similarity is not None, graded.warnings
    assert metrics.embedding_model == EMBED_MODEL
    return f"semantic similarity {metrics.semantic_similarity:.3f}, token F1 {metrics.token_f1:.3f}"


def openai_compatible_judgment() -> str:
    compatible = OpenAICompatibleProvider(f"{BASE}/v1", JUDGE_MODEL, timeout=600)
    result = Judge(compatible, settings).evaluate(
        "What is the capital of France? Answer in one sentence.",
        "The capital of France is Paris.",
        "France does not have a capital city.",
        "Accuracy",
    )
    assert result.metadata["judge"] == {"provider": "openai_compatible", "model": JUDGE_MODEL}
    return f"winner {result.overall_winner.value}"


def openai_compatible_embeddings() -> str:
    compatible = OpenAICompatibleEmbeddings(f"{BASE}/v1", EMBED_MODEL, timeout=300)
    close, related, unrelated = compatible.embed(
        [
            "The cat sat on the mat.",
            "A cat was sitting on the rug.",
            "Quarterly revenue grew by twelve percent.",
        ]
    )
    near = cosine_similarity(close, related)
    far = cosine_similarity(close, unrelated)
    assert near > far, f"paraphrase {near:.3f} should beat unrelated {far:.3f}"
    return f"paraphrase {near:.3f} > unrelated {far:.3f}"


print(f"Ollama at {BASE}: judge {JUDGE_MODEL}, embeddings {EMBED_MODEL}", flush=True)
check("provider health", health, attempts=1)
check("pairwise judgment", pairwise)
check("pointwise grade", pointwise)
check("embeddings rank a paraphrase above unrelated text", embeddings, attempts=1)
check("reference grading with semantic similarity", reference_with_semantic_similarity)
check("openai_compatible provider (Ollama /v1)", openai_compatible_judgment)
check("OpenAI-compatible embeddings (Ollama /v1)", openai_compatible_embeddings, attempts=1)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
