"""Run the reproducible NGSIM I-80 car-following experiment end to end."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import time


# Steps 0-6 are the reproducibility pipeline.  Step 7 is optional: it calibrates
# the models on the already-selected case, which is an in-sample demonstration
# and never held-out generalisation evidence.  Run it explicitly with
# ``--start-step 7`` (or ``--end-step 7``).
PIPELINE = [
    (0, "Dataset inspection", "00_inspect_data.py"),
    (1, "Pair reconstruction", "01_find_following_pairs.py"),
    (2, "Case selection", "02_select_following_case.py"),
    (3, "Trajectory preparation", "03_prepare_trajectories.py"),
    (4, "Model self-check", "04_car_following_models.py"),
    (5, "Numerical simulation", "05_simulate.py"),
    (6, "Evaluation and plots", "06_evaluate_and_plot.py"),
    (7, "Optional in-sample calibration", "07_calibrate.py"),
]
DEFAULT_END_STEP = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML configuration file (default: project config.yaml)",
    )
    parser.add_argument("--start-step", type=int, default=0, choices=range(0, 8))
    parser.add_argument(
        "--end-step",
        type=int,
        default=DEFAULT_END_STEP,
        choices=range(0, 8),
        help=(
            "Last step to run (default: 6). Use 7 to include the optional "
            "in-sample parameter calibration, which stage is excluded by default."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands without executing them"
    )
    return parser.parse_args()


def configure_logging(project_root: Path) -> logging.Logger:
    log_dir = project_root / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ngsim_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "run_pipeline.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_step(command: list[str], project_root: Path, logger: logging.Logger) -> None:
    logger.info("Command: %s", subprocess.list2cmdline(command))
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("  %s", line.rstrip())
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if args.start_step > args.end_step:
        raise ValueError("--start-step must be less than or equal to --end-step")

    logger = configure_logging(project_root)
    selected = [item for item in PIPELINE if args.start_step <= item[0] <= args.end_step]
    logger.info("Starting NGSIM pipeline: steps %d through %d", args.start_step, args.end_step)
    pipeline_start = time.perf_counter()

    for number, label, filename in selected:
        script = project_root / "src" / filename
        if not script.is_file():
            raise FileNotFoundError(f"Pipeline script not found: {script}")
        command = [sys.executable, "-u", str(script), "--config", str(config_path)]
        logger.info("STEP %d — %s", number, label)
        if args.dry_run:
            logger.info("Dry run; command not executed: %s", subprocess.list2cmdline(command))
            continue
        started = time.perf_counter()
        run_step(command, project_root, logger)
        logger.info("STEP %d completed in %.2f s", number, time.perf_counter() - started)

    logger.info("Pipeline completed in %.2f s", time.perf_counter() - pipeline_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

