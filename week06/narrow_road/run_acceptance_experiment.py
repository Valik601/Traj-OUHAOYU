"""一键复现实验 2：速度相关个人空间 / 间隙接受模型。

用法：
    python run_acceptance_experiment.py                 # 仿真 + 指标 + 表 + 图
    python run_acceptance_experiment.py --no-figures    # 只跑仿真与指标
    python run_acceptance_experiment.py --out outputs_acceptance_dev

产物写入 outputs_acceptance/：
    metrics.csv / ped_metrics.csv / pass_events.csv
    trajectories/*.csv（关键情景）
    predictions.json / prediction_consistency.json / summary.json
    tables/A1_acceptance_model.md / A2_results.md / A3_boundaries.md
    figures/figA1...figA5, summary2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from narrowroad.acceptance import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="week06 experiment 2: speed-dependent acceptance")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 96)
    print("experiment 2: speed-dependent personal space (pedestrians squeeze past slow vehicles)")
    print("=" * 96)
    out_dir = Path(args.out) if args.out else ROOT / "outputs_acceptance"
    summary = run_experiment(args.config, out_dir, verbose=True)
    print("-" * 96)
    print(
        f"done: {summary['meta']['n_runs']} runs; prediction consistency "
        f"{summary['prediction_consistency']['n_agree']}/"
        f"{summary['prediction_consistency']['n_checked']} = "
        f"{summary['prediction_consistency']['all_agree']}"
    )
    print("boundaries:")
    for b in summary["boundaries"]:
        print(
            f"  {b['rule']:8s} {b['model']:10s} predicted L_crit={b['predicted_l_crit_m']:.2f} m, "
            f"observed min pass width={b['observed_min_pass_width_m']}"
        )
    if not args.no_figures:
        from narrowroad.acceptance_plots import make_acceptance_figures

        make_acceptance_figures(out_dir, summary)
        print(f"figures written to {out_dir / 'figures'}")
    print(f"total wall time {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
