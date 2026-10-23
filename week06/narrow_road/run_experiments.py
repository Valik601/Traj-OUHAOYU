"""一键复现 week06 窄路相撞实验。

用法：
    python run_experiments.py                # 全流程（仿真 + 指标 + 图 + 参数表）
    python run_experiments.py --no-figures   # 只跑仿真与指标
    python run_experiments.py --out outputs_dev

产物写入 outputs/：
    metrics.csv                 每次仿真的汇总指标
    ped_metrics.csv             每个行人的指标
    trajectories/*.csv          逐时刻轨迹（仅主实验与失败/消融情景）
    initial_conditions.json     冻结的初始条件
    predictions.json            解析可通行判据（“先预测”）
    prediction_consistency.json 预测 vs 仿真结果的一致性检查
    summary.json                报告用关键数字
    tables/*.md                 规则与参数表（Markdown）
    figures/*.png|pdf           论文/汇报图
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from narrowroad.experiments import run_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="week06 窄路相撞实验")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    parser.add_argument("--out", default=None, help="输出目录（默认 outputs/）")
    parser.add_argument("--no-figures", action="store_true", help="跳过绘图")
    parser.add_argument("--no-tables", action="store_true", help="跳过参数表生成")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("week06 narrow-road encounter: unified pedestrian/vehicle agent model (SFM / Wiedemann)")
    print("=" * 92)
    out_dir = Path(args.out) if args.out else ROOT / "outputs"
    summary = run_all(args.config, out_dir, verbose=True)
    print("-" * 92)
    print(f"simulation done: {summary['meta']['n_runs']} runs; analytic-prediction consistency = "
          f"{summary['prediction_consistency']['all_agree']}")

    if not args.no_tables:
        from narrowroad.tables import write_tables

        write_tables(summary, out_dir / "tables")
        print(f"tables written to {out_dir / 'tables'}")

    if not args.no_figures:
        from narrowroad.plotting import make_all_figures

        make_all_figures(out_dir, summary)
        print(f"figures written to {out_dir / 'figures'}")

    print(f"total wall time {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

