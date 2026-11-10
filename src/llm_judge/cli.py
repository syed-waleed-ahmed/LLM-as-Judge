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

from pydantic import ValidationError as PydanticValidationError

from . import __version__
from .criteria import (
    CRITERIA_PRESETS,
    DEFAULT_REFERENCE_CRITERIA,
    REFERENCE_CRITERIA_PRESETS,
    resolve_criterion,
)
from .exceptions import LLMJudgeError
from .logging_config import configure_logging
from .models import Criterion, EvaluationWarning, JudgeResult, PointwiseResult, RankingResult


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
    ba = sub.add_parser("batch", help="Judge many pairs from a CSV/JSONL file.")
    ba.add_argument("--input", required=True, help="Input .csv or .jsonl file.")
    ba.add_argument("--output", required=True, help="Output .csv or .jsonl file.")
    ba.add_argument("--criteria", default="Helpfulness", help="Comma-separated criteria names.")
    ba.add_argument("--weights", default=None, help='Relative weights, e.g. "Helpfulness=2".')
    ba.add_argument("--workers", type=int, default=None, help="Max concurrent requests.")
    ba.add_argument(
        "--mitigate-bias",
        action="store_true",
        help="Run each pair in both orders and average to reduce position bias.",
    )

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

    # list-criteria -------------------------------------------------------
    sub.add_parser("list-criteria", help="List built-in criteria presets.")

    return parser


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
    except json.JSONDecodeError as exc:
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
    from .config import get_settings
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    output = _read_text_arg(args.output, args.output_file, "output")
    reference = _optional_text_arg(args.reference, args.reference_file)
    default = ",".join(DEFAULT_REFERENCE_CRITERIA) if reference is not None else "Helpfulness"
    criteria = _build_criteria(args.criteria or default, args.weights)

    judge = Judge(settings=get_settings())
    result = judge.evaluate_pointwise(
        task, output, criteria, reference=reference, candidate_model=args.model
    )
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_grade(result)
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    from .config import get_settings
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    texts, models = _load_candidates(args.candidates_file)
    criteria = _build_criteria(args.criteria, args.weights)
    settings = get_settings()
    if args.mitigate_bias:
        settings = settings.model_copy(update={"mitigate_position_bias": True})

    judge = Judge(settings=settings)
    result = judge.rank(
        task,
        texts,
        criteria,
        strategy=args.strategy,
        candidate_models={k: v for k, v in models.items() if v},
        max_workers=args.workers,
    )
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_ranking(result)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    # Imported lazily so `list-criteria` works without an API key.
    from .config import get_settings
    from .judge import Judge

    task = _read_text_arg(args.task, args.task_file, "task")
    output_a = _read_text_arg(args.output_a, args.output_a_file, "output-a")
    output_b = _read_text_arg(args.output_b, args.output_b_file, "output-b")
    criteria = _build_criteria(args.criteria, args.weights)

    settings = get_settings()
    if args.mitigate_bias:
        settings = settings.model_copy(update={"mitigate_position_bias": True})

    judge = Judge(settings=settings)
    result = judge.evaluate(
        task,
        output_a,
        output_b,
        criteria,
        candidate_a_model=args.model_a,
        candidate_b_model=args.model_b,
    )

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
    criteria = _build_criteria(args.criteria, args.weights)
    settings = get_settings()
    if args.mitigate_bias:
        settings = settings.model_copy(update={"mitigate_position_bias": True})

    judge = Judge(settings=settings)
    evaluator = BatchEvaluator(judge, max_workers=args.workers)

    print(f"Judging {len(records)} record(s) with {evaluator.max_workers} worker(s)...")

    def _progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} done", end="", flush=True)

    results = evaluator.run(records, criteria, progress=_progress)
    print()
    write_results(results, args.output)

    ok = sum(1 for r in results if r.ok)
    print(f"Wrote {len(results)} result(s) to {args.output} ({ok} ok, {len(results) - ok} failed).")
    return 0 if ok == len(results) else 1


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
