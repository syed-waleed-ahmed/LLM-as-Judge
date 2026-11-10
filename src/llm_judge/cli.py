"""Command-line interface for llm_judge.

Examples::

    llm-judge list-criteria
    llm-judge evaluate --task "Summarise the text" \\
        --output-a-file a.txt --output-b-file b.txt \\
        --criteria "Helpfulness,Conciseness"
    llm-judge batch --input data.csv --output results.csv \\
        --criteria Creativity --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from pydantic import ValidationError as PydanticValidationError

from . import __version__
from .config import Settings
from .criteria import (
    CRITERIA_PRESETS,
    DEFAULT_REFERENCE_CRITERIA,
    REFERENCE_CRITERIA_PRESETS,
    resolve_criterion,
)
from .csv_safety import safe_rows
from .exceptions import LLMJudgeError
from .logging_config import configure_logging
from .models import Criterion, EvaluationWarning, JudgeResult, PointwiseResult, RankingResult

if TYPE_CHECKING:
    from .datasets import DatasetReport, DatasetSummary, Record, RunReport
    from .experiments import ExperimentStore, RunComparison
    from .judge import Judge


def _read_text_arg(value: str | None, file_value: str | None, name: str) -> str:
    """Resolve a text argument from an inline value or a file path."""
    if file_value:
        return Path(file_value).read_text(encoding="utf-8")
    if value is not None:
        return value
    raise SystemExit(f"error: provide --{name} or --{name}-file")


def _parse_criteria(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _parse_weights(raw: str | None) -> dict[str, float]:
    """Parse ``"Name=2,Other=0.5"`` into a mapping, exiting with a message on bad input."""
    weights: dict[str, float] = {}
    if not raw:
        return weights
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        name, sep, value = item.rpartition("=")
        try:
            if not sep or not name.strip():
                raise ValueError
            weights[name.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"error: invalid weight '{item}'; expected Name=number") from None
    return weights


def _build_criteria(raw_criteria: str, raw_weights: str | None) -> list[Criterion]:
    """Resolve criteria names and apply any ``--weights``."""
    names = _parse_criteria(raw_criteria)
    weights = _parse_weights(raw_weights)
    unknown = [n for n in weights if n not in names]
    if unknown:
        raise SystemExit(f"error: --weights names criteria not in --criteria: {unknown}")
    criteria: list[Criterion] = []
    for name in names:
        criterion = resolve_criterion(name)
        if name in weights:
            try:
                criterion = Criterion.model_validate(
                    {**criterion.model_dump(), "weight": weights[name]}
                )
            except PydanticValidationError:
                raise SystemExit(
                    f"error: weight for '{name}' must be greater than 0 and at most 100"
                ) from None
        criteria.append(criterion)
    return criteria


def _add_dataset_run_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--kind",
        choices=["pairwise", "pointwise", "reference"],
        default=None,
        help="Dataset kind (default: inferred from the columns).",
    )
    p.add_argument(
        "--criteria",
        default=None,
        help="Comma-separated criteria. Default: Helpfulness, or Correctness,Completeness "
        "for reference datasets (benchmark samples use their own defaults).",
    )
    p.add_argument("--weights", default=None, help='Relative weights, e.g. "Helpfulness=2".')
    p.add_argument("--workers", type=int, default=None, help="Max concurrent requests.")
    p.add_argument(
        "--mitigate-bias",
        action="store_true",
        help="Run each pair in both orders and average to reduce position bias.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows.")
    p.add_argument(
        "--sample", type=int, default=None, help="Evaluate a reproducible random sample of N rows."
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for --sample (default 0).")
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Evaluate the valid rows even if some rows fail validation.",
    )
    p.add_argument("--json", action="store_true", help="Print the run summary as JSON.")
    p.add_argument(
        "--experiment",
        default=None,
        help="Record this run under the named experiment (see `llm-judge experiments`).",
    )
    p.add_argument("--run-name", default=None, help="Label for the recorded run.")
    p.add_argument("--db", default=None, help="Experiment database (default: JUDGE_DB_PATH).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-judge",
        description="LLM-as-Judge: evaluate and compare model outputs.",
    )
    parser.add_argument("--version", action="version", version=f"llm-judge {__version__}")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default from env/INFO.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Judge backend: groq, ollama, openai_compatible or mock (offline, not an LLM). "
        "Default from JUDGE_PROVIDER, else groq.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # evaluate ------------------------------------------------------------
    ev = sub.add_parser("evaluate", help="Judge a single pair of outputs.")
    ev.add_argument("--task", help="Task description (inline).")
    ev.add_argument("--task-file", help="Path to a file with the task description.")
    ev.add_argument("--output-a", help="Output A text (inline).")
    ev.add_argument("--output-a-file", help="Path to Output A.")
    ev.add_argument("--output-b", help="Output B text (inline).")
    ev.add_argument("--output-b-file", help="Path to Output B.")
    ev.add_argument(
        "--criteria",
        default="Helpfulness",
        help="Comma-separated criteria names (presets or custom). Default: Helpfulness.",
    )
    ev.add_argument(
        "--weights",
        default=None,
        help='Relative criterion weights, e.g. "Helpfulness=2,Conciseness=1". Default: 1 each.',
    )
    ev.add_argument("--model-a", default=None, help="Name of the model that produced Output A.")
    ev.add_argument("--model-b", default=None, help="Name of the model that produced Output B.")
    ev.add_argument(
        "--mitigate-bias",
        action="store_true",
        help="Run each pair in both orders and average to reduce position bias.",
    )
    ev.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # batch ---------------------------------------------------------------
    ba = sub.add_parser(
        "batch",
        help="Evaluate a dataset file (pairwise, pointwise or reference) with checkpoints.",
    )
    ba.add_argument("--input", required=True, help="Dataset: .csv, .jsonl, .json or .yaml.")
    ba.add_argument("--output", required=True, help="Results: .csv, .jsonl or .json.")
    _add_dataset_run_options(ba)
    ba.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file (default: <output>.checkpoint.jsonl). Removed after a "
        "fully successful run.",
    )
    ba.add_argument("--no-checkpoint", action="store_true", help="Don't write a checkpoint.")
    ba.add_argument("--resume", action="store_true", help="Continue from an existing checkpoint.")
    ba.add_argument(
        "--no-retry-failed",
        action="store_true",
        help="On --resume, keep previously failed rows instead of retrying them.",
    )

    # datasets ------------------------------------------------------------
    ds = sub.add_parser("datasets", help="Validate datasets and work with bundled samples.")
    ds_sub = ds.add_subparsers(dest="datasets_command", required=True)
    dv = ds_sub.add_parser("validate", help="Check a dataset file and report problems.")
    dv.add_argument("path", help="Dataset file.")
    dv.add_argument("--kind", choices=["pairwise", "pointwise", "reference"], default=None)
    dv.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    ds_sub.add_parser("samples", help="List the bundled sample datasets.")
    dx = ds_sub.add_parser("export", help="Write a bundled sample dataset to a file.")
    dx.add_argument("name", help="Sample name (see `llm-judge datasets samples`).")
    dx.add_argument("--output", required=True, help="Destination .jsonl file.")

    # experiments ---------------------------------------------------------
    ex = sub.add_parser(
        "experiments",
        aliases=["experiment"],
        help="Browse recorded runs, compare them and detect regressions.",
    )
    ex.add_argument("--db", default=None, help="Experiment database (default: JUDGE_DB_PATH).")
    ex_sub = ex.add_subparsers(dest="experiments_command", required=True)
    ex_sub.add_parser("list", help="List experiments.")
    er = ex_sub.add_parser("runs", help="List recorded runs (newest first).")
    er.add_argument("--experiment", default=None, help="Only runs of this experiment.")
    er.add_argument("--limit", type=int, default=20)
    es = ex_sub.add_parser("show", help="Show one run's metadata and summary.")
    es.add_argument("run_id", help="Run id (a unique prefix is enough).")
    es.add_argument("--json", action="store_true")
    ec = ex_sub.add_parser("compare", help="Compare a candidate run against a baseline run.")
    ec.add_argument("baseline", help="Baseline run id.")
    ec.add_argument("candidate", help="Candidate run id.")
    ec.add_argument(
        "--min-effect",
        type=float,
        default=0.02,
        help="Smallest mean change (normalised) counted as a regression/improvement.",
    )
    ec.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with status 1 when the verdict is a regression (for CI gates).",
    )
    ec.add_argument("--json", action="store_true")
    ea = ex_sub.add_parser(
        "agreement", help="How closely two runs over the same rows agree (e.g. two judges)."
    )
    ea.add_argument("run_a")
    ea.add_argument("run_b")
    ea.add_argument("--json", action="store_true")
    ee = ex_sub.add_parser("export", help="Write a stored run's rows to a file.")
    ee.add_argument("run_id")
    ee.add_argument("--output", required=True, help=".csv or .jsonl")
    ed = ex_sub.add_parser("delete", help="Delete a stored run.")
    ed.add_argument("run_id")

    # benchmark -----------------------------------------------------------
    bm = sub.add_parser(
        "benchmark",
        help="Run a bundled sample (or a dataset file) and print per-model statistics.",
    )
    bm.add_argument("dataset", help="Sample name (e.g. general_qa) or a dataset file path.")
    bm.add_argument("--output", default=None, help="Optionally save per-row results.")
    _add_dataset_run_options(bm)

    # grade ---------------------------------------------------------------
    gr = sub.add_parser(
        "grade", help="Grade a single output, optionally against a reference answer."
    )
    gr.add_argument("--task", help="Task description (inline).")
    gr.add_argument("--task-file", help="Path to a file with the task description.")
    gr.add_argument("--output", help="Output text (inline).")
    gr.add_argument("--output-file", help="Path to the output.")
    gr.add_argument("--reference", help="Expected answer (inline); enables reference grading.")
    gr.add_argument("--reference-file", help="Path to the expected answer.")
    gr.add_argument(
        "--criteria",
        default=None,
        help="Comma-separated criteria. Default: Helpfulness, or Correctness,Completeness "
        "with a reference.",
    )
    gr.add_argument("--weights", default=None, help='Relative weights, e.g. "Correctness=2".')
    gr.add_argument("--model", default=None, help="Name of the model that produced the output.")
    gr.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # rank ----------------------------------------------------------------
    rk = sub.add_parser("rank", help="Rank several candidate outputs for one task.")
    rk.add_argument("--task", help="Task description (inline).")
    rk.add_argument("--task-file", help="Path to a file with the task description.")
    rk.add_argument(
        "--candidates-file",
        required=True,
        help='JSON file: {"id": "text", ...}, or a list of {"id", "output", "model"} objects.',
    )
    rk.add_argument(
        "--strategy",
        choices=["round_robin", "pointwise"],
        default="round_robin",
        help="round_robin compares every pair; pointwise grades each candidate (cheaper).",
    )
    rk.add_argument("--criteria", default="Helpfulness", help="Comma-separated criteria names.")
    rk.add_argument("--weights", default=None, help='Relative weights, e.g. "Helpfulness=2".')
    rk.add_argument("--workers", type=int, default=None, help="Max concurrent requests.")
    rk.add_argument(
        "--mitigate-bias",
        action="store_true",
        help="Judge every pair in both orders (round_robin only; doubles the calls).",
    )
    rk.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # rag -----------------------------------------------------------------
    rg = sub.add_parser("rag", help="Evaluate a RAG answer against its retrieved contexts.")
    rg.add_argument("--question", required=True, help="The user question.")
    rg.add_argument(
        "--context",
        action="append",
        default=[],
        help="A retrieved passage (repeat, in rank order).",
    )
    rg.add_argument(
        "--contexts-file",
        default=None,
        help="JSON list of passages, or a text file with passages separated by blank lines.",
    )
    rg.add_argument("--answer", help="The generated answer (inline).")
    rg.add_argument("--answer-file", help="Path to the generated answer.")
    rg.add_argument("--reference", help="Expected answer; enables context recall.")
    rg.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # agent ---------------------------------------------------------------
    ag = sub.add_parser("agent", help="Evaluate an agent trajectory from a JSON trace file.")
    ag.add_argument("--trace-file", required=True, help="AgentTrace JSON (see llm_judge.agents).")
    ag.add_argument(
        "--criteria", default=None, help="Comma-separated criteria (default: agent set)."
    )
    ag.add_argument("--model", default=None, help="Name of the agent's model.")
    ag.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # consistency ---------------------------------------------------------
    cs = sub.add_parser(
        "consistency", help="Judge one input repeatedly and report how stable the verdict is."
    )
    cs.add_argument("--task", help="Task description (inline).")
    cs.add_argument("--task-file", help="Path to a file with the task description.")
    cs.add_argument("--output-a", help="Output A (pairwise check).")
    cs.add_argument("--output-b", help="Output B (pairwise check).")
    cs.add_argument("--output", help="A single output (pointwise check).")
    cs.add_argument("--criteria", default="Helpfulness", help="Comma-separated criteria names.")
    cs.add_argument("--samples", type=int, default=5, help="Number of repeated judgments (2-50).")
    cs.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Judge sampling temperature for this check (default: JUDGE_TEMPERATURE).",
    )
    cs.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # list-criteria -------------------------------------------------------
    sub.add_parser("list-criteria", help="List built-in criteria presets.")

    # serve ---------------------------------------------------------------
    sv = sub.add_parser("serve", help="Run the REST API server (needs the [api] extra).")
    sv.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: localhost).")
    sv.add_argument("--port", type=int, default=8000)

    # providers / health / version -----------------------------------------
    sub.add_parser("providers", help="List judge providers and whether each is configured.")
    sub.add_parser("health", help="Check that the active provider is reachable (no inference).")
    sub.add_parser("version", help="Print the version.")

    return parser


def _settings(args: argparse.Namespace) -> Settings:
    """Load settings and apply command-line overrides (--provider, --mitigate-bias)."""
    from .config import get_settings

    updates: dict[str, object] = {}
    if getattr(args, "provider", None):
        updates["provider"] = args.provider.strip().lower().replace("-", "_")
    if getattr(args, "mitigate_bias", False):
        updates["mitigate_position_bias"] = True
    settings = get_settings()
    return settings.model_copy(update=updates) if updates else settings


def _cmd_list_criteria() -> int:
    print("Available criteria presets:\n")
    for name, crit in CRITERIA_PRESETS.items():
        print(f"  {name}\n    {crit.description}\n")
    print("Reference criteria (for `grade --reference`):\n")
    for name, crit in REFERENCE_CRITERIA_PRESETS.items():
        print(f"  {name}\n    {crit.description}\n")
    return 0


def _optional_text_arg(value: str | None, file_value: str | None) -> str | None:
    if file_value:
        return Path(file_value).read_text(encoding="utf-8")
    return value


def _load_candidates(path: str) -> tuple[dict[str, str], dict[str, str | None]]:
    """Read candidates (and optional models) from a JSON file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SystemExit(f"error: {path} is not valid JSON: {exc}") from None
    if isinstance(data, dict) and all(isinstance(v, str) for v in data.values()):
        return {str(k): v for k, v in data.items()}, {}
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        texts: dict[str, str] = {}
        models: dict[str, str | None] = {}
        for i, item in enumerate(data, start=1):
            cid = str(item.get("id") or i)
            if cid in texts:
                raise SystemExit(f"error: duplicate candidate id '{cid}' in {path}")
            texts[cid] = str(item.get("output", ""))
            models[cid] = item.get("model")
        return texts, models
    raise SystemExit(
        f'error: {path} must hold {{"id": "text"}} or a list of {{"id", "output"}} objects'
    )


def _cmd_grade(args: argparse.Namespace) -> int:
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    output = _read_text_arg(args.output, args.output_file, "output")
    reference = _optional_text_arg(args.reference, args.reference_file)
    default = ",".join(DEFAULT_REFERENCE_CRITERIA) if reference is not None else "Helpfulness"
    criteria = _build_criteria(args.criteria or default, args.weights)

    judge = Judge(settings=_settings(args))
    result = judge.evaluate_pointwise(
        task, output, criteria, reference=reference, candidate_model=args.model
    )
    if args.json:
        print(_json_text(result.model_dump(mode="json")))
    else:
        _print_grade(result)
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    texts, models = _load_candidates(args.candidates_file)
    criteria = _build_criteria(args.criteria, args.weights)
    judge = Judge(settings=_settings(args))
    result = judge.rank(
        task,
        texts,
        criteria,
        strategy=args.strategy,
        candidate_models={k: v for k, v in models.items() if v},
        max_workers=args.workers,
    )
    if args.json:
        print(_json_text(result.model_dump(mode="json")))
    else:
        _print_ranking(result)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    # Imported lazily so `list-criteria` works without an API key.
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    output_a = _read_text_arg(args.output_a, args.output_a_file, "output-a")
    output_b = _read_text_arg(args.output_b, args.output_b_file, "output-b")
    criteria = _build_criteria(args.criteria, args.weights)

    judge = Judge(settings=_settings(args))
    result = judge.evaluate(
        task,
        output_a,
        output_b,
        criteria,
        candidate_a_model=args.model_a,
        candidate_b_model=args.model_b,
    )

    if args.json:
        print(_json_text(result.model_dump(mode="json")))
    else:
        _print_summary(result)
    return 0


def _print_report(report: DatasetReport) -> None:
    kind = report.kind or "unknown"
    print(f"{report.source}: {report.format}, kind {kind}, {report.rows} row(s)")
    print(f"  valid rows: {report.valid_rows}/{report.rows}")
    if report.fingerprint:
        print(f"  fingerprint (dataset version): {report.fingerprint}")
    if report.aliases:
        aliases = ", ".join(f"{k} -> {v}" for k, v in report.aliases.items())
        print(f"  column aliases: {aliases}")
    if report.unknown_columns:
        print(f"  ignored columns: {', '.join(report.unknown_columns)}")
    for issue in report.issues:
        where = f"row {issue.row}" if issue.row else "dataset"
        print(f"  {issue.severity}: {where}: {issue.message}")
    if report.issues_truncated:
        print(f"  ... and {report.issues_truncated} more issue(s)")


def _load_for_run(args: argparse.Namespace, source: str | Path, kind: str | None) -> list[Record]:
    from .datasets import load_dataset, sample_records

    records, report = load_dataset(source, kind=kind, settings=_settings(args))  # type: ignore[arg-type]
    if not report.ok:
        _print_report(report)
        if not (args.skip_invalid and records):
            raise SystemExit(
                "error: the dataset has problems (see above); fix them or pass --skip-invalid"
            )
        print(f"Continuing with {len(records)} valid row(s) (--skip-invalid).")
    if args.sample is not None:
        records = sample_records(records, args.sample, seed=args.seed)
    if args.limit is not None:
        records = records[: max(0, args.limit)]
    if not records:
        raise SystemExit("error: no rows to evaluate")
    return records


def _run_dataset(
    args: argparse.Namespace,
    records: list[Record],
    criteria: list[Criterion] | None,
    checkpoint: Path | None,
) -> tuple[RunReport, Judge]:
    from .datasets import DatasetRunner
    from .judge import Judge

    judge = Judge(settings=_settings(args))
    runner = DatasetRunner(
        judge,
        criteria,
        max_workers=args.workers,
        checkpoint=checkpoint,
        resume=getattr(args, "resume", False),
        retry_failed=not getattr(args, "no_retry_failed", False),
        progress=lambda done, total: print(f"\r  {done}/{total} done", end="", flush=True),
    )
    print(f"Evaluating {len(records)} row(s) with {runner.max_workers} worker(s)...")
    report = runner.run(records)
    print()
    return report, judge


def _store(args: argparse.Namespace) -> ExperimentStore:
    from .experiments import ExperimentStore

    return ExperimentStore(getattr(args, "db", None) or _settings(args).db_path)


def _maybe_record(args: argparse.Namespace, report: RunReport, judge: Judge, source: str) -> None:
    if not getattr(args, "experiment", None):
        return
    from .experiments import record_dataset_run

    with _store(args) as store:
        info = record_dataset_run(
            store,
            report,
            judge,
            experiment=args.experiment,
            name=args.run_name,
            dataset_source=source,
        )
    print(f"Recorded run {info.id} in experiment '{args.experiment}' ({store.path}).")


def _print_summary_stats(summary: DatasetSummary) -> None:
    print(
        f"\n{summary.kind} run: {summary.succeeded}/{summary.total} succeeded, "
        f"{summary.failed} failed" + (f", {summary.resumed} resumed" if summary.resumed else "")
    )
    if summary.kind == "pairwise":
        print(f"  overall winners: A={summary.wins_a} B={summary.wins_b} tie={summary.ties}")
        for m in summary.models:
            print(
                f"  {m.model:<24} win rate {m.win_rate:.3f}  "
                f"(W-L-T {m.wins}-{m.losses}-{m.ties} over {m.rows})"
            )
    else:
        if summary.mean_overall_score is not None:
            print(f"  mean overall score: {summary.mean_overall_score:.3f}")
        if summary.mean_token_f1 is not None:
            print(f"  mean token F1 vs reference: {summary.mean_token_f1:.3f}")
        if summary.rows_with_unsupported_claims is not None:
            print(f"  rows with unsupported claims: {summary.rows_with_unsupported_claims}")
        for m in summary.models:
            print(f"  {m.model:<24} mean score {m.mean_overall_score:.3f} over {m.rows} row(s)")
    for name, means in summary.criteria_means.items():
        values = ", ".join(f"{k}={v:g}" for k, v in means.items())
        print(f"  [{name}] mean {values}")
    if summary.warning_counts:
        counts = ", ".join(f"{k}={v}" for k, v in summary.warning_counts.items())
        print(f"  warnings: {counts}")


def _cmd_batch(args: argparse.Namespace) -> int:
    from .datasets import summarize, write_outcomes

    records = _load_for_run(args, args.input, args.kind)
    criteria = _build_criteria(args.criteria, args.weights) if args.criteria else None
    checkpoint: Path | None = None
    if not args.no_checkpoint:
        checkpoint = Path(args.checkpoint or f"{args.output}.checkpoint.jsonl")

    report, judge = _run_dataset(args, records, criteria, checkpoint)
    write_outcomes(report.outcomes, args.output)
    summary = summarize(report)
    if args.json:
        print(_json_text(summary.model_dump(mode="json")))
    else:
        _print_summary_stats(summary)
    print(
        f"\nWrote {len(report.outcomes)} result(s) to {args.output} "
        f"({report.succeeded} ok, {report.failed} failed)."
    )
    _maybe_record(args, report, judge, str(args.input))

    if report.interrupted:
        print(f"Interrupted. Resume with: --resume (checkpoint: {checkpoint})", file=sys.stderr)
        return 130
    if report.failed:
        if checkpoint:
            print(f"Retry failed rows with --resume (checkpoint: {checkpoint}).", file=sys.stderr)
        return 1
    if checkpoint and checkpoint.exists():
        checkpoint.unlink()
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    from .datasets import SAMPLE_DATASETS, sample_text, validate_dataset

    if args.datasets_command == "validate":
        report = validate_dataset(args.path, kind=args.kind, settings=_settings(args))
        if args.json:
            print(_json_text(report.model_dump(mode="json")))
        else:
            _print_report(report)
            print("OK" if report.ok else "INVALID")
        return 0 if report.ok else 1
    if args.datasets_command == "samples":
        print("Bundled sample datasets (synthetic; for demos and smoke tests):\n")
        for sample in SAMPLE_DATASETS.values():
            print(f"  {sample.name:<22} {sample.kind:<9} {sample.description}")
            print(f"  {'':<22} default criteria: {', '.join(sample.criteria)}")
        return 0
    if args.datasets_command == "export":
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sample_text(args.name), encoding="utf-8")
        print(f"Wrote sample '{args.name}' to {out}")
        return 0
    return 2  # pragma: no cover - argparse enforces the choices


def _print_comparison(c: RunComparison) -> None:
    print(f"Comparing {c.candidate_id} (candidate) against {c.baseline_id} (baseline)")
    print(f"  metric: {c.metric}; matched rows: {c.matched_rows}", end="")
    extra = []
    if c.baseline_only:
        extra.append(f"{c.baseline_only} only in baseline")
    if c.candidate_only:
        extra.append(f"{c.candidate_only} only in candidate")
    print(f" ({', '.join(extra)})" if extra else "")
    if c.mean_delta is not None:
        print(
            f"  mean: {c.baseline_mean:.3f} -> {c.candidate_mean:.3f} "
            f"(change {c.mean_delta:+.3f}, 95% CI [{c.ci_low:+.3f}, {c.ci_high:+.3f}], "
            f"{c.ci_method})"
        )
    print(f"  verdict: {c.verdict.replace('_', ' ').upper()}")
    if c.winner_flips is not None:
        print(f"  winner flips: {c.winner_flips}")
    if c.regressed_rows or c.improved_rows:
        print(f"  rows regressed: {c.regressed_rows}; rows improved: {c.improved_rows}")
    for d in c.regressions[:5]:
        print(f"    worse  {d.row_id}: {d.baseline:.3f} -> {d.candidate:.3f}")
    for name, delta in c.criteria_deltas.items():
        print(f"  [{name}] mean change {delta:+.3f}")
    for warning in c.warnings:
        print(f"  warning: {warning}")


def _cmd_experiments(args: argparse.Namespace) -> int:
    from .datasets import DatasetSummary
    from .experiments import compare_runs

    command = args.experiments_command
    with _store(args) as store:
        if command == "list":
            experiments = store.list_experiments()
            if not experiments:
                print(f"No experiments in {store.path} yet.")
            for e in experiments:
                print(f"  {e.name:<30} {e.runs:>4} run(s)   created {e.created_at}")
            return 0
        if command == "runs":
            runs = store.list_runs(experiment=args.experiment, limit=args.limit)
            if not runs:
                print("No runs recorded yet.")
            for r in runs:
                label = f"{r.name} " if r.name != r.id else ""
                print(
                    f"  {r.id}  {label}[{r.experiment or '-'}] {r.kind:<9} {r.status:<22} "
                    f"{r.succeeded}/{r.total} ok  judge {r.judge_provider}/{r.judge_model}"
                )
            return 0
        if command == "show":
            run = store.require_run(args.run_id)
            if args.json:
                print(_json_text(run.model_dump(mode="json")))
                return 0
            print(f"Run {run.id} ({run.name}) in experiment '{run.experiment or '-'}'")
            print(f"  created {run.created_at}; status {run.status}; {run.duration_seconds}s")
            print(f"  dataset {run.dataset_source} (fingerprint {run.dataset_fingerprint})")
            print(
                f"  judge {run.judge_provider}/{run.judge_model}; prompt v{run.prompt_version}; "
                f"llm-judge {run.config.get('llm_judge_version')}"
            )
            _print_summary_stats(DatasetSummary.model_validate(run.summary))
            return 0
        if command == "compare":
            comparison = compare_runs(
                store, args.baseline, args.candidate, min_effect=args.min_effect
            )
            if args.json:
                print(_json_text(comparison.model_dump(mode="json")))
            else:
                _print_comparison(comparison)
            regression = comparison.verdict == "regression"
            return 1 if regression and args.fail_on_regression else 0
        if command == "agreement":
            from .reliability import judge_agreement

            agreement = judge_agreement(store, args.run_a, args.run_b)
            if args.json:
                print(_json_text(agreement.model_dump(mode="json")))
                return 0
            print(f"{agreement.judge_a} vs {agreement.judge_b} on {agreement.matched_rows} row(s)")
            if agreement.cohen_kappa is not None:
                print(
                    f"  winner agreement {agreement.raw_agreement:.0%}, "
                    f"Cohen's kappa {agreement.cohen_kappa:+.3f}"
                )
            if agreement.mean_absolute_difference is not None:
                rho = "n/a" if agreement.spearman is None else f"{agreement.spearman:+.3f}"
                print(
                    f"  Spearman rho {rho}, mean absolute score difference "
                    f"{agreement.mean_absolute_difference:.3f}"
                )
            for note in agreement.notes:
                print(f"  note: {note}")
            return 0
        if command == "export":
            rows = store.get_rows(args.run_id)
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix.lower() == ".csv":
                import csv

                fieldnames: list[str] = []
                for row in rows:
                    fieldnames.extend(k for k in row.flat if k not in fieldnames)
                with out.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(safe_rows(row.flat for row in rows))
            else:
                with out.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        fh.write(
                            json.dumps({**row.flat, "result": row.result}, ensure_ascii=False)
                            + "\n"
                        )
            print(f"Wrote {len(rows)} row(s) to {out}")
            return 0
        if command == "delete":
            run = store.require_run(args.run_id)
            store.delete_run(run.id)
            print(f"Deleted run {run.id}")
            return 0
    return 2  # pragma: no cover - argparse enforces the choices


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .api import create_app
    except ImportError:
        raise SystemExit(
            "error: the API server needs FastAPI and Uvicorn: pip install 'llm-as-judge[api]'"
        ) from None
    settings = _settings(args)
    if args.host not in _LOCAL_HOSTS and not settings.api_keys:
        print(
            f"warning: serving on {args.host} without JUDGE_API_KEYS. Anyone who can reach "
            "this server can use (and exhaust) your provider quota.",
            file=sys.stderr,
        )
    print(f"LLM-as-Judge API on http://{args.host}:{args.port} (docs at /docs)")
    uvicorn.run(
        create_app(settings), host=args.host, port=args.port, log_level=settings.log_level.lower()
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .datasets import SAMPLE_DATASETS, load_sample, summarize, write_outcomes

    if args.dataset in SAMPLE_DATASETS:
        sample = SAMPLE_DATASETS[args.dataset]
        records = load_sample(args.dataset, settings=_settings(args))
        if args.sample is not None:
            from .datasets import sample_records

            records = sample_records(records, args.sample, seed=args.seed)
        if args.limit is not None:
            records = records[: max(0, args.limit)]
        criteria: list[Criterion] | None = _build_criteria(
            args.criteria or ",".join(sample.criteria), args.weights
        )
    else:
        records = _load_for_run(args, args.dataset, args.kind)
        criteria = _build_criteria(args.criteria, args.weights) if args.criteria else None

    report, judge = _run_dataset(args, records, criteria, checkpoint=None)
    summary = summarize(report)
    if args.output:
        write_outcomes(report.outcomes, args.output)
    if args.json:
        print(_json_text(summary.model_dump(mode="json")))
    else:
        settings = _settings(args)
        print(f"Benchmark '{args.dataset}' with provider {settings.provider}")
        _print_summary_stats(summary)
        if args.output:
            print(f"\nPer-row results written to {args.output}")
    source = f"sample:{args.dataset}" if args.dataset in SAMPLE_DATASETS else str(args.dataset)
    _maybe_record(args, report, judge, source)
    if report.interrupted:
        return 130
    return 0 if not report.failed else 1


def _cmd_providers(args: argparse.Namespace) -> int:
    from .providers.registry import configuration_status, provider_info, provider_names

    settings = _settings(args)
    print("Judge providers:\n")
    for name in provider_names():
        info = provider_info(name)
        ready, detail = configuration_status(name, settings)
        marker = "*" if name == settings.provider else " "
        status = "configured" if ready else "not configured"
        print(f"{marker} {name:<18} {status:<15} {info.cost}")
        print(f"    {info.description}")
        print(f"    {detail}\n")
    print("* active provider (set with JUDGE_PROVIDER or --provider).")
    print('  "configured" means the settings are present; run `llm-judge health` to test.')
    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    from .providers.registry import build_provider

    settings = _settings(args)
    provider = build_provider(settings)
    check = getattr(provider, "health", None)
    if check is None:
        print(f"{settings.provider}: no health check available")
        return 0
    health = check()
    print(
        f"{health.provider} ({health.model}): {'OK' if health.ok else 'FAILED'} - {health.detail}"
    )
    return 0 if health.ok else 1


def _read_contexts(args: argparse.Namespace) -> list[str]:
    contexts = list(args.context)
    if args.contexts_file:
        text = Path(args.contexts_file).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list) and all(isinstance(c, str) for c in data):
            contexts.extend(data)
        elif data is not None:
            raise SystemExit("error: --contexts-file JSON must be a list of strings")
        else:
            contexts.extend(p.strip() for p in text.split("\n\n") if p.strip())
    if not contexts:
        raise SystemExit("error: provide --context (repeatable) or --contexts-file")
    return contexts


def _cmd_rag(args: argparse.Namespace) -> int:
    from .judge import Judge

    contexts = _read_contexts(args)
    answer = _read_text_arg(args.answer, args.answer_file, "answer")
    result = Judge(settings=_settings(args)).evaluate_rag(
        args.question, contexts, answer, reference=args.reference
    )
    if args.json:
        print(_json_text(result.model_dump(mode="json")))
        return 0

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    print(f"\nFaithfulness:      {pct(result.faithfulness)} of claims supported by the contexts")
    print(f"Answer relevance:  {pct(result.answer_relevance)}")
    print(f"Context precision: {pct(result.context_precision)} (relevant contexts ranked early)")
    if result.context_recall is not None:
        print(f"Context recall:    {pct(result.context_recall)} of the reference covered")
    for claim in result.unsupported_claims:
        print(f"  unsupported: {claim}")
    for verdict in result.contexts:
        mark = "relevant" if verdict.relevant else "not relevant"
        print(f"  context {verdict.context_id}: {mark}")
    _print_warnings(result.warnings)
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    from .agents import AgentTrace
    from .judge import Judge

    try:
        trace = AgentTrace.model_validate_json(Path(args.trace_file).read_text(encoding="utf-8"))
    except PydanticValidationError as exc:
        raise SystemExit(f"error: invalid trace file: {exc}") from None
    criteria = _parse_criteria(args.criteria) if args.criteria else None
    result = Judge(settings=_settings(args)).evaluate_agent(
        trace, criteria, candidate_model=args.model
    )
    if args.json:
        print(_json_text(result.model_dump(mode="json")))
        return 0
    m = result.metrics
    print(
        f"\nSteps {m.steps}, tool calls {m.tool_calls} "
        f"({m.invalid_tool_calls} invalid, {m.failed_tool_calls} failed, "
        f"{m.repeated_calls} repeated)"
    )
    if m.expected_tool_recall is not None:
        print(f"Expected tools used: {m.expected_tool_recall:.0%}")
    for check in result.checks:
        where = f"step {check.step}: " if check.step else ""
        print(f"  {check.severity}: {where}{check.message}")
    _print_grade(result.grade)
    return 0


def _cmd_consistency(args: argparse.Namespace) -> int:
    from .judge import Judge
    from .reliability import measure_consistency

    task = _read_text_arg(args.task, args.task_file, "task")
    settings = _settings(args)
    if args.temperature is not None:
        settings = settings.model_copy(update={"temperature": args.temperature})
    report = measure_consistency(
        Judge(settings=settings),
        task,
        output_a=args.output_a,
        output_b=args.output_b,
        output=args.output,
        criteria=_parse_criteria(args.criteria),
        samples=args.samples,
    )
    if args.json:
        print(_json_text(report.model_dump(mode="json")))
        return 0
    print(
        f"\n{report.mode} consistency over {report.succeeded}/{report.samples} sample(s) "
        f"at temperature {report.temperature:g}"
    )
    if report.agreement_rate is not None:
        print(
            f"  modal verdict {report.modal_winner} in {report.agreement_rate:.0%} of samples "
            f"{report.winner_counts}"
        )
    if report.mean is not None:
        print(
            f"  {report.metric}: mean {report.mean:+.3f}, sd {report.std:.3f}, "
            f"95% CI [{report.ci_low:+.3f}, {report.ci_high:+.3f}]"
        )
    for error in report.errors:
        print(f"  failed sample: {error}")
    for note in report.notes:
        print(f"  note: {note}")
    return 0


def _print_warnings(warnings: list[EvaluationWarning]) -> None:
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning.code}: {warning.message}")


def _print_grade(result: PointwiseResult) -> None:
    label = "Reference grading" if result.mode == "reference" else "Pointwise grading"
    overall = "n/a" if result.overall_score is None else f"{result.overall_score:.3f}"
    print(f"\n{label}: overall score {overall} (weighted, 0-1)")
    for score in result.criteria:
        print(f"  [{score.criterion}] {score.score:g} - {score.explanation}")
    if result.feedback:
        print(f"\nFeedback: {result.feedback}")
    for title, items in (
        ("Missing information", result.missing_information),
        ("Unsupported claims", result.unsupported_claims),
    ):
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")
    if result.reference_metrics:
        m = result.reference_metrics
        print(f"\nReference metrics: token F1 {m.token_f1:.3f}, exact match {m.exact_match}")
    _print_warnings(result.warnings)


def _print_ranking(result: RankingResult) -> None:
    print(f"\nLeaderboard ({result.strategy}, criteria: {', '.join(result.criteria)}):")
    for row in result.leaderboard:
        record = ""
        if result.strategy == "round_robin":
            record = f"  W-L-T {row.wins}-{row.losses}-{row.ties}"
        print(f"  {row.rank}. {row.candidate}  score {row.score:.3f}{record}")
    _print_warnings(result.warnings)


def _print_summary(result: JudgeResult) -> None:
    print(f"\nOverall winner: {result.overall_winner.value.upper()}")
    if result.overall_comment:
        print(f"Comment: {result.overall_comment}\n")
    if result.overall_score_a is not None and result.overall_score_b is not None:
        print(
            f"Overall score (weighted, 0-1): A={result.overall_score_a:.3f} "
            f"B={result.overall_score_b:.3f}"
        )
    for score in result.criteria:
        print(
            f"  [{score.criterion}] A={score.output_a_score} "
            f"B={score.output_b_score} -> winner {score.winner.value}"
        )
    _print_warnings(result.warnings)
    model = result.metadata.get("model")
    if model:
        print(f"\n(model: {model}, latency: {result.metadata.get('latency_seconds')}s)")


def _make_console_safe(stream: TextIO) -> None:
    """Never crash on characters the console can't encode.

    LLM output often contains characters such as U+202F (narrow no-break space)
    that legacy consoles (e.g. cp1252 on Windows) can't represent. Replacing them
    is better than a traceback after a successful, paid-for evaluation.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass


def _json_text(payload: object) -> str:
    """Serialise JSON losslessly for the current stdout.

    Non-ASCII is written as-is on UTF-8 streams and as ``\\uXXXX`` escapes
    otherwise, so redirected JSON never loses characters.
    """
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    return json.dumps(payload, indent=2, ensure_ascii=encoding != "utf8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    _make_console_safe(sys.stdout)
    _make_console_safe(sys.stderr)
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level or "INFO")

    try:
        if args.command == "list-criteria":
            return _cmd_list_criteria()
        if args.command == "version":
            print(f"llm-judge {__version__}")
            return 0
        if args.command == "providers":
            return _cmd_providers(args)
        if args.command == "health":
            return _cmd_health(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "batch":
            return _cmd_batch(args)
        if args.command == "datasets":
            return _cmd_datasets(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args)
        if args.command in ("experiments", "experiment"):
            return _cmd_experiments(args)
        if args.command == "serve":
            return _cmd_serve(args)
        if args.command == "rag":
            return _cmd_rag(args)
        if args.command == "agent":
            return _cmd_agent(args)
        if args.command == "consistency":
            return _cmd_consistency(args)
        if args.command == "grade":
            return _cmd_grade(args)
        if args.command == "rank":
            return _cmd_rank(args)
    except LLMJudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable, keeps type checkers happy


if __name__ == "__main__":
    raise SystemExit(main())
