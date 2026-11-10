"""Shareable Markdown reports for recorded evaluation runs.

:func:`run_report` turns a stored run into a self-contained Markdown document:
what was evaluated, how it was judged (criteria, instructions, settings),
the results (overall, per criterion, per model, warnings, failures), an
optional comparison against a baseline run, and the reproducibility metadata.

Reports only restate what the store recorded. They add no numbers of their
own, and they label mock-judge runs as heuristic.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from .compare import RunComparison, compare_runs
from .store import ExperimentStore, RunInfo

_VERDICT_TEXT = {
    "regression": "Regression detected",
    "improvement": "Improvement detected",
    "no_significant_change": "No significant change",
    "insufficient_data": "Insufficient data",
}


def _cell(value: object) -> str:
    """Render a table cell: no newlines or pipes, and a dash for missing values."""
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        text = f"{value:.3f}"
    else:
        text = str(value)
    return " ".join(text.split()).replace("|", "\\|")


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines


def _pct(part: int | None, whole: int | None) -> str:
    if part is None or not whole:
        return "-"
    return f"{part} ({part / whole:.0%})"


def _run_section(run: RunInfo) -> list[str]:
    fingerprint = (run.dataset_fingerprint or "")[:12] or None
    lines = ["## Run", ""]
    lines += _table(
        ["Field", "Value"],
        [
            ("Run id", f"`{run.id}`"),
            ("Experiment", run.experiment),
            ("Recorded", run.created_at),
            ("Kind", run.kind),
            ("Status", run.status),
            ("Dataset", run.dataset_source),
            ("Dataset fingerprint", f"`{fingerprint}`" if fingerprint else None),
            ("Judge", f"{run.judge_provider} · {run.judge_model}"),
            ("Prompt version", run.prompt_version),
            ("Duration", f"{run.duration_seconds:.1f} s"),
            ("Rows", f"{run.succeeded} succeeded, {run.failed} failed, {run.total} total"),
        ],
    )
    if run.judge_provider == "mock":
        lines += [
            "",
            "> **Mock judge.** These scores come from a deterministic text heuristic, not an "
            "LLM, and it ignores judge instructions. They show how the pipeline works, not "
            "real quality judgments.",
        ]
    if run.notes:
        lines += ["", f"Notes: {run.notes}"]
    return lines


def _config_section(run: RunInfo) -> list[str]:
    config = run.config
    lines = ["## How it was judged", ""]
    criteria = config.get("criteria") or [{"name": c} for c in run.criteria]
    lines += _table(
        ["Criterion", "Scale", "Weight", "Description"],
        [
            (
                c.get("name"),
                f"{c['scale_min']}-{c['scale_max']}" if "scale_min" in c else None,
                None if c.get("weight") is None else f"{c['weight']:g}",
                c.get("description"),
            )
            for c in criteria
        ],
    )
    instructions = config.get("instructions")
    if instructions:
        lines += ["", "Judge instructions:", ""]
        lines += ["> " + line if line else ">" for line in str(instructions).splitlines()]
    settings = config.get("settings") or {}
    if settings:
        lines += ["", "Settings: " + ", ".join(f"`{k}={v}`" for k, v in sorted(settings.items()))]
    return lines


def _results_section(run: RunInfo) -> list[str]:
    s = run.summary
    lines = ["## Results", ""]
    if run.kind == "pairwise":
        decided = s.get("succeeded")
        lines += _table(
            ["Measure", "Output A", "Output B", "Ties"],
            [
                (
                    "Wins",
                    _pct(s.get("wins_a"), decided),
                    _pct(s.get("wins_b"), decided),
                    _pct(s.get("ties"), decided),
                ),
                (
                    "Mean overall score (0-1)",
                    s.get("mean_overall_score_a"),
                    s.get("mean_overall_score_b"),
                    None,
                ),
            ],
        )
    else:
        rows: list[tuple[str, object]] = [("Mean overall score (0-1)", s.get("mean_overall_score"))]
        if s.get("mean_token_f1") is not None:
            rows.append(("Mean token F1 vs reference", s.get("mean_token_f1")))
        if s.get("mean_semantic_similarity") is not None:
            rows.append(("Mean semantic similarity", s.get("mean_semantic_similarity")))
        if s.get("rows_with_unsupported_claims") is not None:
            rows.append(
                (
                    "Rows with unsupported claims",
                    _pct(s.get("rows_with_unsupported_claims"), s.get("succeeded")),
                )
            )
        lines += _table(["Measure", "Value"], rows)

    means: dict[str, dict[str, float]] = s.get("criteria_means") or {}
    if means:
        sides = sorted({side for values in means.values() for side in values})
        labels = {"a": "Output A", "b": "Output B", "score": "Mean score"}
        lines += ["", "### Per criterion (raw scale)", ""]
        lines += _table(
            ["Criterion", *(labels.get(side, side) for side in sides)],
            [(name, *(values.get(side) for side in sides)) for name, values in means.items()],
        )

    models = s.get("models") or []
    if models:
        lines += ["", "### Per model", ""]
        lines += _table(
            ["Model", "Rows", "Wins", "Losses", "Ties", "Win rate", "Mean score"],
            [
                (
                    m.get("model"),
                    m.get("rows"),
                    m.get("wins"),
                    m.get("losses"),
                    m.get("ties"),
                    None if m.get("win_rate") is None else f"{m['win_rate']:.0%}",
                    m.get("mean_overall_score"),
                )
                for m in models
            ],
        )

    warnings: dict[str, int] = s.get("warning_counts") or {}
    if warnings:
        lines += ["", "### Reliability warnings", ""]
        ordered: list[tuple[str, int]] = sorted(warnings.items(), key=lambda kv: (-kv[1], kv[0]))
        lines += _table(["Warning", "Rows"], ordered)
    return lines


def _failures_section(store: ExperimentStore, run: RunInfo, max_rows: int) -> list[str]:
    if not run.failed:
        return []
    failed = [r for r in store.get_rows(run.id) if not r.ok]
    lines = ["## Failed rows", ""]
    lines += _table(
        ["Row", "Error"], [(r.row_id, (r.error or "")[:200]) for r in failed[:max_rows]]
    )
    if len(failed) > max_rows:
        lines += ["", f"... and {len(failed) - max_rows} more."]
    return lines


def _comparison_section(comparison: RunComparison, max_rows: int) -> list[str]:
    c = comparison
    lines = [
        "## Comparison with the baseline",
        "",
        f"**{_VERDICT_TEXT.get(c.verdict, c.verdict)}** (baseline `{c.baseline_id}`, "
        f"metric: {c.metric}).",
        "",
    ]
    interval = (
        f"{c.ci_low:+.3f} to {c.ci_high:+.3f} ({c.confidence:.0%}, {c.ci_method})"
        if c.ci_low is not None and c.ci_high is not None
        else None
    )
    lines += _table(
        ["Measure", "Value"],
        [
            ("Matched rows", c.matched_rows),
            ("Baseline mean", c.baseline_mean),
            ("Candidate mean", c.candidate_mean),
            ("Mean change", None if c.mean_delta is None else f"{c.mean_delta:+.3f}"),
            ("Confidence interval", interval),
            ("Smallest change that counts", f"{c.min_effect:.3f}"),
            ("Rows that regressed / improved", f"{c.regressed_rows} / {c.improved_rows}"),
            ("Winner flips", c.winner_flips),
            (
                "Failed rows (baseline / candidate)",
                f"{c.baseline_failed_rows} / {c.candidate_failed_rows}",
            ),
        ],
    )
    if c.criteria_deltas:
        lines += ["", "### Change per criterion", ""]
        lines += _table(
            ["Criterion", "Mean change"],
            [(name, f"{delta:+.3f}") for name, delta in c.criteria_deltas.items()],
        )
    if c.regressions:
        lines += ["", "### Largest regressions", ""]
        lines += _table(
            ["Row", "Baseline", "Candidate", "Change"],
            [
                (d.row_id, d.baseline, d.candidate, f"{d.delta:+.3f}")
                for d in c.regressions[:max_rows]
            ],
        )
    if c.warnings:
        lines += ["", "### Caveats", ""]
        lines += [f"- {w}" for w in c.warnings]
    return lines


def _reproducibility_section(run: RunInfo) -> list[str]:
    config = {k: v for k, v in run.config.items() if k != "criteria"}
    return [
        "## Reproducibility",
        "",
        "The judging configuration recorded with this run (criteria are listed above):",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
    ]


def run_report(
    store: ExperimentStore,
    run_id: str,
    *,
    baseline_id: str | None = None,
    max_rows: int = 10,
    min_effect: float = 0.02,
    generated_at: datetime | None = None,
) -> str:
    """Render a Markdown report for a stored run, optionally compared with a baseline.

    ``run_id`` and ``baseline_id`` accept unique prefixes, as elsewhere in the
    store. ``max_rows`` caps the failed-row and regression listings.
    """
    from .. import __version__

    run = store.require_run(run_id)
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Evaluation report: {run.name or run.id}",
        "",
        f"Generated by LLM-as-Judge {__version__} on {when}.",
        "",
    ]
    sections = [
        _run_section(run),
        _config_section(run),
        _results_section(run),
        _failures_section(store, run, max_rows),
    ]
    if baseline_id:
        comparison = compare_runs(store, baseline_id, run.id, min_effect=min_effect)
        sections.append(_comparison_section(comparison, max_rows))
    sections.append(_reproducibility_section(run))
    for section in sections:
        if section:
            lines += section + [""]
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["run_report"]
