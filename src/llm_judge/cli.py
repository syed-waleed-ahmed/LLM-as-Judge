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

from . import __version__
from .criteria import CRITERIA_PRESETS
from .exceptions import LLMJudgeError
from .logging_config import configure_logging
from .models import JudgeResult


def _read_text_arg(value: str | None, file_value: str | None, name: str) -> str:
    """Resolve a text argument from an inline value or a file path."""
    if file_value:
        return Path(file_value).read_text(encoding="utf-8")
    if value is not None:
        return value
    raise SystemExit(f"error: provide --{name} or --{name}-file")


def _parse_criteria(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


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
        "--mitigate-bias",
        action="store_true",
        help="Run each pair in both orders and average to reduce position bias.",
    )
    ev.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")

    # batch ---------------------------------------------------------------
    ba = sub.add_parser("batch", help="Judge many pairs from a CSV/JSONL file.")
    ba.add_argument("--input", required=True, help="Input .csv or .jsonl file.")
    ba.add_argument("--output", required=True, help="Output .csv or .jsonl file.")
    ba.add_argument("--criteria", default="Helpfulness", help="Comma-separated criteria names.")
    ba.add_argument("--workers", type=int, default=None, help="Max concurrent requests.")
    ba.add_argument(
        "--mitigate-bias",
        action="store_true",
        help="Run each pair in both orders and average to reduce position bias.",
    )

    # list-criteria -------------------------------------------------------
    sub.add_parser("list-criteria", help="List built-in criteria presets.")

    return parser


def _cmd_list_criteria() -> int:
    print("Available criteria presets:\n")
    for name, crit in CRITERIA_PRESETS.items():
        print(f"  {name}\n    {crit.description}\n")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    # Imported lazily so `list-criteria` works without an API key.
    from .config import get_settings
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    output_a = _read_text_arg(args.output_a, args.output_a_file, "output-a")
    output_b = _read_text_arg(args.output_b, args.output_b_file, "output-b")

    settings = get_settings()
    if args.mitigate_bias:
        settings = settings.model_copy(update={"mitigate_position_bias": True})

    judge = Judge(settings=settings)
    result = judge.evaluate(task, output_a, output_b, _parse_criteria(args.criteria))

    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_summary(result)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from .batch import BatchEvaluator, read_records, write_results
    from .config import get_settings
    from .judge import Judge

    records = read_records(args.input)
    settings = get_settings()
    if args.mitigate_bias:
        settings = settings.model_copy(update={"mitigate_position_bias": True})

    judge = Judge(settings=settings)
    evaluator = BatchEvaluator(judge, max_workers=args.workers)

    print(f"Judging {len(records)} record(s) with {evaluator.max_workers} worker(s)...")

    def _progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} done", end="", flush=True)

    results = evaluator.run(records, _parse_criteria(args.criteria), progress=_progress)
    print()
    write_results(results, args.output)

    ok = sum(1 for r in results if r.ok)
    print(f"Wrote {len(results)} result(s) to {args.output} ({ok} ok, {len(results) - ok} failed).")
    return 0 if ok == len(results) else 1


def _print_summary(result: JudgeResult) -> None:
    print(f"\nOverall winner: {result.overall_winner.value.upper()}")
    if result.overall_comment:
        print(f"Comment: {result.overall_comment}\n")
    for score in result.criteria:
        print(
            f"  [{score.criterion}] A={score.output_a_score} "
            f"B={score.output_b_score} -> winner {score.winner.value}"
        )
    model = result.metadata.get("model")
    if model:
        print(f"\n(model: {model}, latency: {result.metadata.get('latency_seconds')}s)")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level or "INFO")

    try:
        if args.command == "list-criteria":
            return _cmd_list_criteria()
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        if args.command == "batch":
            return _cmd_batch(args)
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
