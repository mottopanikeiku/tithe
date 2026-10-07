from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .analysis import analyze_run
from .calibration_stage import run_calibration_stage
from .data import TRUTHFULQA_CSV_URL, prepare_data
from .scoring import score_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alignmenttax",
        description="Run TruthfulQA honesty and calibration experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Build binary TruthfulQA JSONL.")
    prepare.add_argument("--out", required=True, help="Output JSONL path.")
    prepare.add_argument("--seed", type=int, default=20260420, help="A/B randomization seed.")
    prepare.add_argument("--source-csv", help="Optional local TruthfulQA CSV path.")
    prepare.add_argument("--cache-path", default="data/raw/TruthfulQA.csv", help="Raw CSV cache path.")
    prepare.add_argument("--url", default=TRUTHFULQA_CSV_URL, help="CSV URL (default: pinned TruthfulQA revision).")
    prepare.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")

    score = subparsers.add_parser("score", help="Score models on processed TruthfulQA JSONL.")
    score.add_argument("--config", required=True, help="YAML or JSON config path.")
    score.add_argument("--out", required=True, help="Output scores JSONL path.")
    score.add_argument("--fake", action="store_true", help="Use deterministic fake scores.")
    score.add_argument("--limit", type=int, help="Optional question limit for smoke tests.")
    score.add_argument(
        "--resume",
        action="store_true",
        help="Append missing score rows and skip rows already present in the output JSONL.",
    )

    analyze = subparsers.add_parser("analyze", help="Analyze a score run.")
    analyze.add_argument("--run", required=True, help="Run directory, or a plain/gzipped scores JSONL file.")
    analyze.add_argument("--out", required=True, help="Report output directory.")
    analyze.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override configured bootstrap iteration count.",
    )
    analyze.add_argument("--no-plots", action="store_true", help="Skip plot generation.")

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Run held-out post-hoc calibration analysis from existing scores.",
    )
    calibrate.add_argument("--run", required=True, help="Run directory containing plain/gzipped scores JSONL.")
    calibrate.add_argument("--out", required=True, help="Report output directory.")
    calibrate.add_argument("--seed", type=int, default=20260420, help="Deterministic split seed.")
    calibrate.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help="Fraction of questions used to fit temperatures.",
    )
    calibrate.add_argument("--ece-bins", type=int, default=10, help="ECE bins for summaries and plots.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare-data":
        count = prepare_data(
            out=args.out,
            seed=args.seed,
            source_csv=args.source_csv,
            cache_path=args.cache_path,
            url=args.url,
            limit=args.limit,
        )
        print(f"Wrote {count} TruthfulQA binary rows to {Path(args.out)}")
        return 0

    if args.command == "score":
        count = score_run(
            config_path=args.config,
            out=args.out,
            fake=args.fake,
            limit=args.limit,
            resume=args.resume,
        )
        print(f"Wrote {count} score rows to {Path(args.out)}")
        return 0

    if args.command == "analyze":
        manifest = analyze_run(
            run=args.run,
            out=args.out,
            bootstrap_iterations=args.bootstrap_iterations,
            no_plots=args.no_plots,
        )
        print(f"Wrote analysis outputs to {Path(args.out)}")
        print(f"Summary: {manifest['summary_path']}")
        return 0

    if args.command == "calibrate":
        manifest = run_calibration_stage(
            run_dir=args.run,
            report_dir=args.out,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
            ece_bins=args.ece_bins,
        )
        print(f"Wrote stage 2 calibration outputs to {Path(args.out)}")
        print(f"Summary: {manifest['summary']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
