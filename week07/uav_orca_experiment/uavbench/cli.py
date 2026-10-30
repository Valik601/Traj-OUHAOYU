"""uavbench 命令行入口（handoff §9.2）。

示例：
    python -m uavbench.cli validate-config --config configs/default.yaml
    python -m uavbench.cli generate-scenes --profile full
    python -m uavbench.cli run --model orca --scenario B --seed 1000
    python -m uavbench.cli batch --profile smoke
    python -m uavbench.cli batch --profile full --resume --jobs 24
    python -m uavbench.cli batch --profile sensitivity --resume --jobs 24
    python -m uavbench.cli audit --results results
    python -m uavbench.cli summarize --results results
    python -m uavbench.cli render --scenario B --seed 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
SCENES_DIR = ROOT / "scenes"
RESULTS_DIR = ROOT / "results"


def _cmd_validate_config(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if not path.is_absolute():
        path = (CONFIGS_DIR / path) if (CONFIGS_DIR / path).exists() else (ROOT / path)
    try:
        cfg = load_config(path)
        errors = validate_config(cfg)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 2
    print(f"OK config={path}")
    print(f"  config_hash = {cfg.hash()}")
    print(f"  n_agents={cfg.simulation.n_agents} dt={cfg.experiment.dt} steps={cfg.n_steps}")
    print(f"  formal_seeds={list(cfg.scenes.formal_seeds)}")
    print(f"  models.A={list(cfg.models.A)} models.B={list(cfg.models.B)}")
    return 0


def _cmd_generate_scenes(args: argparse.Namespace) -> int:
    from .initialization import generate_scenes
    from .runner import config_path_for

    profile = args.profile
    cfg = load_config(config_path_for(profile, CONFIGS_DIR))
    scenes = generate_scenes(cfg, profile, SCENES_DIR)
    print(f"generated/reused {len(scenes)} scenes in {SCENES_DIR}")
    for sc in scenes:
        print(f"  {sc.scene_id:>16s} seed={sc.seed:>5d} sha256={sc.sha256[:16]} min_start={sc.meta.get('min_start_separation'):.3f}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .initialization import build_scene, load_scene, save_scene, scene_path
    from .runner import execute_run

    cfg_path = args.config or "default.yaml"
    cfg = load_config(CONFIGS_DIR / cfg_path)
    scene_id = str(int(args.seed))
    p = scene_path(SCENES_DIR, scene_id)
    if not p.exists():
        scene = build_scene(cfg, int(args.seed))
        save_scene(scene, SCENES_DIR)
        print(f"generated scene {p}")
    spec = {
        "experiment_id": args.experiment_id or "single",
        "scenario": args.scenario,
        "model": args.model,
        "scene_id": scene_id,
        "seed": int(args.seed),
        "config_file": cfg_path,
        "overrides": {},
        "note": "single run via CLI",
    }
    rec = execute_run(spec, root_dir=str(ROOT))
    print(json.dumps({k: rec.get(k) for k in ("status", "valid", "runtime_s", "out_dir", "error")}, indent=2, ensure_ascii=False))
    return 0 if rec.get("status") != "numerical_error" else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    from .runner import run_batch

    jobs = args.jobs if args.jobs is not None else 1
    summary = run_batch(
        args.profile,
        jobs=jobs,
        resume=bool(args.resume),
        limit=args.limit,
        models=args.models,
        scenarios=args.scenarios,
        seeds=[int(s) for s in args.seeds.split(",")] if args.seeds else None,
        blocks=args.blocks.split(",") if args.blocks else None,
        freeze=not args.no_freeze,
        force_manifest=bool(args.force_manifest),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import audit_results

    summary = audit_results(
        Path(args.results),
        experiments=args.experiments.split(",") if args.experiments else None,
        max_runs=args.max_runs,
        write=not args.no_write,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2, ensure_ascii=False))
    if summary["failures"]:
        print("FIRST FAILURES:")
        for f in summary["failures"][:10]:
            print(f"  {f['run_dir']}: {f['problems'] or f['error']}")
    return 0 if summary["n_failed"] == 0 else 1


def _cmd_summarize(args: argparse.Namespace) -> int:
    from .statistics import summarize_results

    result = summarize_results(
        Path(args.results),
        experiments=args.experiments.split(",") if args.experiments else None,
        config_path=CONFIGS_DIR / (args.config or "default.yaml"),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .visualization import render_scene_seed

    outputs = render_scene_seed(
        scenario=args.scenario,
        seed=int(args.seed),
        results_dir=Path(args.results),
        out_dir=Path(args.out or (ROOT / "reports" / "figures")),
        models=args.models.split(",") if args.models else None,
    )
    print(json.dumps([str(p) for p in outputs], indent=2))
    return 0


def _cmd_render_all(args: argparse.Namespace) -> int:
    from .visualization import render_all

    outputs = render_all(
        results_dir=Path(args.results),
        out_dir=Path(args.out or (ROOT / "reports" / "figures")),
        configs_dir=CONFIGS_DIR,
    )
    print(json.dumps([str(p) for p in outputs], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uavbench", description="100 UAV 3D VO/RVO/ORCA benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-config", help="校验配置")
    p.add_argument("--config", default="configs/default.yaml")
    p.set_defaults(func=_cmd_validate_config)

    p = sub.add_parser("generate-scenes", help="按 profile 生成/复用场景")
    p.add_argument("--profile", default="full", choices=["smoke", "full", "development", "sensitivity", "stress"])
    p.add_argument("--config", default=None)
    p.set_defaults(func=_cmd_generate_scenes)

    p = sub.add_parser("run", help="单次运行")
    p.add_argument("--model", required=True)
    p.add_argument("--scenario", required=True, choices=["A", "B"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--config", default="default.yaml")
    p.add_argument("--experiment-id", default="single")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("batch", help="批量运行")
    p.add_argument("--profile", required=True, choices=["smoke", "full", "development", "sensitivity", "stress"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--seeds", default=None, help="逗号分隔种子列表")
    p.add_argument("--blocks", default=None, help="逗号分隔敏感性 block")
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--force-manifest", action="store_true")
    p.set_defaults(func=_cmd_batch)

    p = sub.add_parser("audit", help="离线审计")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--experiments", default=None, help="逗号分隔 experiment_id")
    p.add_argument("--max-runs", type=int, default=None)
    p.add_argument("--no-write", action="store_true")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("summarize", help="汇总与配对统计")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--experiments", default=None)
    p.add_argument("--config", default="default.yaml")
    p.set_defaults(func=_cmd_summarize)

    p = sub.add_parser("render", help="渲染单场景图表/动画")
    p.add_argument("--scenario", required=True, choices=["A", "B"])
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--models", default=None)
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_render)

    p = sub.add_parser("render-all", help="渲染全部报告图")
    p.add_argument("--results", default=str(RESULTS_DIR))
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_render_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
