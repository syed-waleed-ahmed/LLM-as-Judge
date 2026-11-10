"""Small, original, synthetic sample datasets bundled with the package.

They exist to demonstrate and smoke-test every evaluation mode offline. With
the mock provider, results are fully reproducible. They are far too small to
draw conclusions about real models. The candidate "systems" are neutral labels
(``system-alpha``, ``system-beta``), not real models.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files

from ..config import Settings
from .io import DatasetError
from .records import DatasetKind, Record
from .validation import load_dataset


@dataclass(frozen=True)
class SampleDataset:
    name: str
    kind: DatasetKind
    description: str
    criteria: tuple[str, ...]

    @property
    def filename(self) -> str:
        return f"{self.name}.jsonl"


SAMPLE_DATASETS: dict[str, SampleDataset] = {
    s.name: s
    for s in (
        SampleDataset(
            "general_qa",
            "reference",
            "Short factual questions with reference answers (some outputs are wrong).",
            ("Correctness", "Completeness"),
        ),
        SampleDataset(
            "code_generation",
            "pairwise",
            "Pairs of small Python functions for the same task.",
            ("Helpfulness", "Code Readability"),
        ),
        SampleDataset(
            "summarization",
            "pairwise",
            "One-sentence summaries, including vague and inaccurate ones.",
            ("Factual Accuracy", "Conciseness"),
        ),
        SampleDataset(
            "rag",
            "reference",
            "Context-grounded answers; some add facts the context doesn't support.",
            ("Correctness", "Factual Consistency"),
        ),
        SampleDataset(
            "instruction_following",
            "pointwise",
            "Outputs that follow or ignore explicit format constraints.",
            ("Instruction Following",),
        ),
        SampleDataset(
            "safety",
            "pairwise",
            "Responses to risky requests: responsible help versus dismissive or unsafe replies.",
            ("Safety", "Helpfulness"),
        ),
    )
}


def get_sample(name: str) -> SampleDataset:
    try:
        return SAMPLE_DATASETS[name]
    except KeyError:
        raise DatasetError(
            f"Unknown sample dataset '{name}'. Available: {', '.join(SAMPLE_DATASETS)}."
        ) from None


def sample_text(name: str) -> str:
    """Raw JSONL content of a bundled sample dataset."""
    sample = get_sample(name)
    return (
        files(__package__).joinpath("samples").joinpath(sample.filename).read_text(encoding="utf-8")
    )


def load_sample(name: str, *, settings: Settings | None = None) -> list[Record]:
    """Load a bundled sample dataset as validated records."""
    sample = get_sample(name)
    resource = files(__package__).joinpath("samples").joinpath(sample.filename)
    with as_file(resource) as path:
        records, report = load_dataset(path, kind=sample.kind, settings=settings)
    if not report.ok:  # pragma: no cover - guarded by the test suite
        raise DatasetError(f"Bundled sample '{name}' failed validation: {report.issues}")
    return records
