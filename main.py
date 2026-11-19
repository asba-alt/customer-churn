"""Customer Retention & Revenue Intelligence Platform — command line entry point.

Usage
-----
    python main.py all          # full pipeline (default)
    python main.py pipeline     # data ingestion, cleaning, validation, database
    python main.py sql          # execute the SQL analytics library
    python main.py train        # train and select the churn model
    python main.py evaluate     # test-set evaluation and curves
    python main.py score        # score customers and build the priority list
    python main.py explain      # SHAP global and local explanations
    python main.py report       # EDA figures and business reports
    python main.py check        # verify the environment and dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import config


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)


def cmd_check() -> int:
    """Environment and dataset readiness check."""
    from src.data.ingestion import dataset_is_available, file_fingerprint, missing_dataset_message

    logger = logging.getLogger(__name__)
    logger.info("Python         : %s", sys.version.split()[0])
    for module in ("pandas", "numpy", "sklearn", "xgboost", "shap", "matplotlib", "seaborn"):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            logger.info("%-14s : %s", module, version)
        except ImportError:
            logger.error("%-14s : MISSING", module)

    if dataset_is_available():
        fingerprint = file_fingerprint()
        logger.info("Dataset        : FOUND (%s, %s bytes)", fingerprint["file_name"], fingerprint["size_bytes"])
        logger.info("SHA-256        : %s", fingerprint["sha256"][:32] + "...")
        return 0

    logger.error(missing_dataset_message())
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Customer Retention & Revenue Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["all", "pipeline", "sql", "train", "evaluate", "score", "explain", "report", "check"],
        help="pipeline stage to run (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    logger = logging.getLogger(__name__)
    config.ensure_dirs()

    if args.command == "check":
        return cmd_check()

    started = time.perf_counter()

    try:
        from src.pipeline import (
            PipelineState,
            run_all,
            stage_data,
            stage_evaluate,
            stage_explain,
            stage_report,
            stage_score,
            stage_sql,
            stage_train,
        )
        from src.data.cleaning import load_clean

        if args.command == "all":
            state = run_all()
        elif args.command == "pipeline":
            stage_data(PipelineState())
        elif args.command == "sql":
            stage_sql(PipelineState(clean=load_clean()))
        elif args.command == "train":
            stage_train(PipelineState(clean=load_clean()))
        elif args.command == "evaluate":
            stage_evaluate(PipelineState(clean=load_clean()))
        elif args.command == "score":
            stage_score(PipelineState(clean=load_clean()))
        elif args.command == "explain":
            stage_explain(PipelineState(clean=load_clean()))
        elif args.command == "report":
            stage_report(PipelineState(clean=load_clean()))
        else:  # pragma: no cover - argparse restricts choices
            parser.error(f"unknown command: {args.command}")

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    elapsed = time.perf_counter() - started
    logger.info("Finished '%s' in %.1fs", args.command, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
