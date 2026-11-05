"""Command-line entry point. Importing the package never starts a computation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .config import COUNTRIES, MODEL_NAMES, SEEDS, load_config, run_matrix


def parser():
    result = argparse.ArgumentParser(
        description="Short-term national-level Solar Generation forecasting"
    )
    subs = result.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    common.add_argument("--workspace", type=Path, default=Path.cwd())
    prepare = subs.add_parser(
        "prepare",
        parents=[common],
        help="Verify supplied CSVs and construct complete hourly windows",
    )
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    subs.add_parser(
        "plan",
        parents=[common],
        help="Print the fixed 90-run matrix without running it",
    )
    subs.add_parser(
        "status",
        parents=[common],
        help="Inspect manifests, histories and best checkpoints",
    )
    subs.add_parser(
        "verify-initialization",
        parents=[common],
        help="Verify all matched random initialization hashes",
    )
    one = subs.add_parser(
        "train", parents=[common], help="Train one explicitly selected run"
    )
    one.add_argument("--country", choices=list(COUNTRIES), required=True)
    one.add_argument("--model", choices=MODEL_NAMES, required=True)
    one.add_argument("--seed", type=int, choices=SEEDS, required=True)
    one.add_argument("--attempt", type=int, default=1)
    many = subs.add_parser(
        "train-plan",
        parents=[common],
        help="Serially call the protected single-run entry point",
    )
    many.add_argument(
        "--group", choices=("all", "fusion", "supervision"), default="all"
    )
    for command in (one, many):
        command.add_argument(
            "--execute-training",
            action="store_true",
            help="Explicitly opt into actual Train/Validation training",
        )
        command.add_argument("--device", default="cuda")
    evaluation = subs.add_parser(
        "evaluate", parents=[common], help="Evaluate accepted best checkpoints"
    )
    evaluation.add_argument(
        "--split", choices=("Validation", "Test"), default="Validation"
    )
    evaluation.add_argument(
        "--group", choices=("all", "fusion", "supervision"), default="all"
    )
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--allow-test", action="store_true")
    evaluation.add_argument("--comparators", type=Path)
    audit = subs.add_parser(
        "audit-evaluation",
        help="Recompute metrics from saved arrays without any model forward",
    )
    audit.add_argument("directory", type=Path)
    plots = subs.add_parser(
        "plot", help="Render saved evaluation results without inference"
    )
    plots.add_argument("directory", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "audit-evaluation":
        from .evaluation import audit_saved

        output = audit_saved(args.directory)
    elif args.command == "plot":
        from .plotting import plot_saved

        output = plot_saved(args.directory)
    else:
        config = load_config(args.config)
        config_dir = args.config.resolve().parent
        workspace = args.workspace.resolve()
        if args.command == "prepare":
            from .workflow import prepare_data

            output = prepare_data(args.raw_dir, workspace, config_dir)
        elif args.command == "plan":
            output = {"training_started": False, "tasks": run_matrix()}
        elif args.command == "verify-initialization":
            from .workflow import verify_all_initializations

            output = verify_all_initializations(config_dir)
        elif args.command == "status":
            from .workflow import inspect_task

            tasks = [
                inspect_task(
                    workspace, config, row["country"], row["model"], row["seed"]
                )
                for row in run_matrix()
            ]
            output = {
                "counts": dict(Counter(r["status"] for r in tasks)),
                "tasks": tasks,
            }
        elif args.command == "train":
            from .training import train

            output = train(
                workspace,
                config,
                config_dir,
                args.country,
                args.model,
                args.seed,
                execute_training=args.execute_training,
                device=args.device,
                attempt=args.attempt,
            )
        elif args.command == "train-plan":
            from .training import train
            from .workflow import inspect_task

            if not args.execute_training:
                raise PermissionError(
                    "Serial training requires --execute-training; use plan for read-only inspection"
                )
            completed = []
            for row in run_matrix():
                if args.group != "all" and row["group"] != args.group:
                    continue
                state = inspect_task(
                    workspace, config, row["country"], row["model"], row["seed"]
                )
                if state["status"] == "COMPLETE":
                    completed.append(state)
                    continue
                if state["status"] != "NOT_STARTED":
                    raise RuntimeError(
                        "Failed/partial task encountered; stop without automatic retry"
                    )
                completed.append(
                    train(
                        workspace,
                        config,
                        config_dir,
                        row["country"],
                        row["model"],
                        row["seed"],
                        execute_training=True,
                        device=args.device,
                    )
                )
            output = {"completed": len(completed), "tasks": completed}
        elif args.command == "evaluate":
            from .evaluation import evaluate

            output = evaluate(
                workspace,
                config,
                config_dir,
                split=args.split,
                group=args.group,
                device=args.device,
                allow_test=args.allow_test,
                comparator_file=args.comparators,
            )
        else:
            raise ValueError("Unknown command")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0
