"""Optional embedding-based semantic similarity for reference grading."""

from __future__ import annotations

import json

import httpx
import pytest

from llm_judge.config import Settings
from llm_judge.datasets import DatasetRunner, ReferenceRecord, summarize
from llm_judge.exceptions import ConfigurationError, ProviderError
from llm_judge.judge import Judge
from llm_judge.providers.embeddings import (
    EmbeddingProvider,
    OllamaEmbeddings,
    OpenAICompatibleEmbeddings,
    build_embedder,
    cosine_similarity,
)

from .conftest import FnProvider, make_pointwise_payload


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, groq_api_key="k", **overrides)


class _FakeEmbedder:
    model = "fake-embed"

    def __init__(self, vectors=None, error: Exception | None = None) -> None:
        self.vectors = vectors or {"same": [1.0, 0.0], "near": [0.8, 0.6], "far": [0.0, 1.0]}
        self.error = error
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.error:
            raise self.error
        return [self.vectors.get(t.split()[0], [0.5, 0.5]) for t in texts]


def _grader(embedder) -> Judge:
    provider = FnProvider(
        lambda m: make_pointwise_payload(("Correctness", 8), ("Completeness", 7), missing=[])
    )
    return Judge(provider, _settings(), embedder=embedder)


# --- math and protocol ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [([1, 0], [1, 0], 1.0), ([1, 0], [0, 1], 0.0), ([1, 0], [-1, 0], -1.0), ([0, 0], [1, 1], 0.0)],
)
def test_cosine_similarity(a, b, expected):
    assert cosine_similarity(a, b) == pytest.approx(expected)


def test_cosine_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_fake_embedder_satisfies_the_protocol():
    assert isinstance(_FakeEmbedder(), EmbeddingProvider)


# --- reference grading ------------------------------------------------------------------


def test_reference_grading_adds_semantic_similarity():
    embedder = _FakeEmbedder()
    graded = _grader(embedder).evaluate_reference("t", "near answer", "same reference")
    metrics = graded.reference_metrics
    assert metrics is not None
    assert metrics.semantic_similarity == pytest.approx(0.8)
    assert metrics.embedding_model == "fake-embed"
    assert metrics.token_f1 is not None  # the lexical metrics are still there


def test_no_embedder_means_no_semantic_metric():
    graded = _grader(None).evaluate_reference("t", "near answer", "same reference")
    assert graded.reference_metrics is not None
    assert graded.reference_metrics.semantic_similarity is None
    assert graded.reference_metrics.embedding_model is None


def test_pointwise_without_reference_never_embeds():
    embedder = _FakeEmbedder()
    _grader(embedder).evaluate_pointwise("t", "o", ["Correctness", "Completeness"])
    assert embedder.calls == 0


def test_embedding_failure_is_a_warning_not_an_error():
    embedder = _FakeEmbedder(error=ProviderError("embedding server down"))
    graded = _grader(embedder).evaluate_reference("t", "near", "same")
    assert graded.reference_metrics is not None
    assert graded.reference_metrics.semantic_similarity is None
    [warning] = [w for w in graded.warnings if w.code == "semantic_similarity_unavailable"]
    assert "embedding server down" in warning.message


def test_dataset_runs_report_mean_semantic_similarity():
    records = [
        ReferenceRecord(id="1", task_description="t", output="same a", reference="same r"),
        ReferenceRecord(id="2", task_description="t", output="far a", reference="same r"),
    ]
    report = DatasetRunner(_grader(_FakeEmbedder())).run(records)
    summary = summarize(report)
    assert summary.mean_semantic_similarity == pytest.approx(0.5)


# --- HTTP backends -------------------------------------------------------------------------


def test_ollama_embeddings_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    embedder = OllamaEmbeddings("nomic-embed-text", transport=httpx.MockTransport(handler))
    assert embedder.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert seen["url"].endswith("/api/embed")
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_openai_compatible_embeddings_follow_the_index_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]},
        )

    embedder = OpenAICompatibleEmbeddings(
        "http://local.test/v1",
        "embed-model",
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    assert embedder.embed(["first", "second"]) == [[1.0], [2.0]]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(503, text="busy"), "HTTP 503"),
        (httpx.Response(200, json={"embeddings": [[0.1]]}), "Expected 2 embedding"),
        (httpx.Response(200, json={"embeddings": [["x"], ["y"]]}), "must contain numbers"),
        (httpx.Response(200, text="not json"), "Malformed embedding response"),
    ],
)
def test_bad_embedding_responses_raise_provider_errors(response, message):
    embedder = OllamaEmbeddings("m", transport=httpx.MockTransport(lambda request: response))
    with pytest.raises(ProviderError, match=message):
        embedder.embed(["a", "b"])


# --- configuration -----------------------------------------------------------------------


def test_embeddings_are_off_by_default():
    assert build_embedder(_settings()) is None
    assert Judge(FnProvider(lambda m: {}), _settings()).embedder is None


def test_configured_backends_are_built():
    ollama = build_embedder(
        _settings(embeddings_provider="ollama", embeddings_model="nomic-embed-text")
    )
    assert isinstance(ollama, OllamaEmbeddings) and ollama.base_url == "http://localhost:11434"
    compatible = build_embedder(
        _settings(
            embeddings_provider="openai_compatible",
            embeddings_model="text-embed",
            openai_base_url="http://local.test/v1",
        )
    )
    assert isinstance(compatible, OpenAICompatibleEmbeddings)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embeddings_provider": "groq", "embeddings_model": "x"}, "must be one of"),
        ({"embeddings_provider": "ollama"}, "JUDGE_EMBEDDINGS_MODEL"),
        ({"embeddings_provider": "openai_compatible", "embeddings_model": "x"}, "BASE_URL"),
    ],
)
def test_invalid_embedding_configuration_is_rejected(overrides, message):
    with pytest.raises(ConfigurationError, match=message):
        build_embedder(_settings(**overrides))
