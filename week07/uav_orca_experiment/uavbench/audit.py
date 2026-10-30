"""离线审计（handoff §8 结果审计、§9.3）。

从已保存的 ``trajectories.npz`` 重算指标，与 ``metrics.json`` 逐字段比对；同时校验：
- ``config.json`` 的配置 hash == ``provenance.config_hash``；
- 场景文件 hash == ``provenance.scene_hash``；
- ``comparison.csv`` 每行可追溯到一个实际运行目录且数值与 ``metrics.json`` 一致；
- 运行目录资产齐全、数值数组有限（无效运行除外）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Config, ConfigError
from .initialization import load_scene, scene_path
from .metrics import compute_metrics
from .statistics import COMPARISON_COLUMNS, discover_run_dirs

ROOT = Path(__file__).resolve().parent.parent
FLOAT_RTOL = 1e-9
FLOAT_ATOL = 1e-12


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_metrics(saved: dict[str, Any], recomputed: dict[str, Any]) -> list[str]:
    """逐字段比较两个指标字典；返回不一致描述列表。"""
    problems: list[str] = []
    fa, fb = _flatten(saved), _flatten(recomputed)
    keys = sorted(set(fa) | set(fb))
    for key in keys:
        if key not in fa:
            problems.append(f"missing in metrics.json: {key}")
            continue
        if key not in fb:
            problems.append(f"missing in recomputed: {key}")
            continue
        va, vb = fa[key], fb[key]
        if va is None and vb is None:
            continue
        if isinstance(va, bool) or isinstance(vb, bool):
            if bool(va) != bool(vb):
                problems.append(f"{key}: {va} != {vb}")
            continue
        if _is_number(va) and _is_number(vb):
            a, b = float(va), float(vb)
            if not (np.isfinite(a) and np.isfinite(b)):
                problems.append(f"{key}: non-finite value ({a}, {b})")
                continue
            if not np.isclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
                problems.append(f"{key}: {a!r} != {b!r}")
            continue
        if isinstance(va, list) and isinstance(vb, list):
            if len(va) != len(vb):
                problems.append(f"{key}: list length {len(va)} != {len(vb)}")
            continue
        if va != vb:
            problems.append(f"{key}: {va!r} != {vb!r}")
    return problems


def _config_from_run(run_dir: Path) -> Config:
    cfg_json = _load_json(run_dir / "config.json")
    # config.json 为有效配置（含敏感性 override），重新校验
    cfg = Config.from_json_dict(cfg_json)
    from .config import validate_config

    errs = validate_config(cfg)
    if errs:
        raise ConfigError("; ".join(errs))
    return cfg


def _scene_file_for(prov: dict[str, Any], run_dir: Path, results_dir: Path) -> Path:
    rel = prov.get("scene_file")
    candidates: list[Path] = []
    if rel:
        base = rel
        candidates.append(ROOT / base)
        # 兼容绝对路径与 results 相对路径
        p = Path(base)
        if p.is_absolute():
            candidates.append(p)
        for root in (run_dir, run_dir.parent, results_dir, results_dir.parent):
            candidates.append(root / base)
        candidates.append(ROOT / "scenes" / Path(base).name)
    candidates.append(scene_path(ROOT / "scenes", str(prov.get("scene_id"))))
    for cand in candidates:
        try:
            if cand.exists():
                return cand
        except OSError:
            continue
    raise FileNotFoundError(f"cannot locate scene file for provenance scene_file={rel!r}")


def audit_run(run_dir: Path, results_dir: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "run_dir": str(run_dir),
        "experiment_id": run_dir.parents[3].name if len(run_dir.parents) > 3 else "",
        "scenario": run_dir.parents[2].name if len(run_dir.parents) > 2 else "",
        "model": run_dir.parents[1].name if len(run_dir.parents) > 1 else "",
        "scene_id": run_dir.parent.name.replace("scene_", "") if run_dir.parent.name.startswith("scene_") else "",
        "ok": False,
        "n_problems": 0,
        "problems": "",
        "error": "",
    }
    # 目录深挖：results/<exp>/<scenario>/<model>/scene_<id>/<hash>
    parts = run_dir.relative_to(results_dir).parts if results_dir in run_dir.parents else ()
    if len(parts) >= 5:
        rec.update({"experiment_id": parts[0], "scenario": parts[1], "model": parts[2], "scene_id": parts[3][6:]})
    try:
        prov = _load_json(run_dir / "provenance.json")
        saved = _load_json(run_dir / "metrics.json")
        cfg = _config_from_run(run_dir)
        problems: list[str] = []
        if cfg.hash() != prov.get("config_hash"):
            problems.append(
                f"config hash mismatch: config.json={cfg.hash()} provenance={prov.get('config_hash')}"
            )
        if saved.get("config_hash") != prov.get("config_hash"):
            problems.append("metrics config_hash != provenance config_hash")
        scene = load_scene(_scene_file_for(prov, run_dir, results_dir))
        if scene.sha256 != prov.get("scene_hash"):
            problems.append(
                f"scene hash mismatch: file={scene.sha256} provenance={prov.get('scene_hash')}"
            )
        if saved.get("scene_hash") != prov.get("scene_hash"):
            problems.append("metrics scene_hash != provenance scene_hash")
        with np.load(run_dir / "trajectories.npz", allow_pickle=False) as npz:
            traj = {k: np.array(npz[k]) for k in npz.files}
        # 轨迹完整性
        n = cfg.simulation.n_agents
        if traj["positions"].shape != (cfg.n_steps + 1, n, 3):
            problems.append(f"positions shape {traj['positions'].shape} unexpected")
        if traj["velocities"].shape != (cfg.n_steps, n, 3):
            problems.append(f"velocities shape {traj['velocities'].shape} unexpected")
        if not np.all(np.isfinite(traj["positions"])):
            problems.append("non-finite positions in trajectory")
        speed = np.linalg.norm(traj["velocities"], axis=2)
        if np.max(speed) > cfg.simulation.v_max + 1e-8:
            problems.append(f"trajectory exceeds v_max: {float(np.max(speed))}")
        status = saved.get("status") or {}
        recomputed = compute_metrics(
            traj,
            scene,
            cfg,
            model=prov.get("model", ""),
            scenario=prov.get("scenario", ""),
            seed=int(prov.get("seed", -1)),
            scene_hash=prov.get("scene_hash", ""),
            config_hash=prov.get("config_hash", ""),
            center_entry_times=traj["center_entry_times"],
            center_entry_steps=traj["center_entry_steps"],
            release_planned=traj["release_planned"],
            release_actual=traj["release_actual"],
            status=status,
            experiment_id=prov.get("experiment_id"),
        )
        problems.extend(compare_metrics(saved, recomputed.metrics))
        # 对照 agents.csv 行数与列
        agents = pd.read_csv(run_dir / "agents.csv")
        if len(agents) != n:
            problems.append(f"agents.csv rows {len(agents)} != {n}")
        rec["n_problems"] = len(problems)
        rec["problems"] = " | ".join(problems[:20])
        rec["ok"] = not problems
    except Exception as exc:  # noqa: BLE001 - 审计失败必须记录
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def audit_comparison_tables(results_dir: Path) -> list[dict[str, Any]]:
    """校验每个 comparison.csv 行都能追溯到运行目录且数值一致。"""
    results_dir = Path(results_dir)
    records: list[dict[str, Any]] = []
    for comp_path in sorted(results_dir.glob("*/comparison.csv")):
        exp = comp_path.parent.name
        df = pd.read_csv(comp_path)
        missing_cols = [c for c in COMPARISON_COLUMNS if c not in df.columns]
        if missing_cols:
            records.append(
                {"experiment_id": exp, "ok": False, "problems": f"missing columns: {missing_cols}"}
            )
            continue
        problems: list[str] = []
        for _, row in df.iterrows():
            run_dir = (
                results_dir
                / exp
                / str(row["scenario"])
                / str(row["model"])
                / f"scene_{row['scene_id']}"
                / str(row["config_hash"])
            )
            if not (run_dir / "metrics.json").exists():
                problems.append(f"unresolvable row: {run_dir}")
                continue
            metrics = _load_json(run_dir / "metrics.json")
            checks = [
                ("arrival_rate", metrics.get("arrival_rate")),
                ("safe_completion_rate", metrics.get("safe_completion_rate")),
                ("min_separation", metrics.get("min_separation")),
                ("violating_pair_count", metrics.get("violating_pair_count")),
            ]
            for col, actual in checks:
                expected = row[col]
                if pd.isna(expected) and actual is None:
                    continue
                if pd.isna(expected) != (actual is None):
                    problems.append(f"row {row['model']}/{row['scene_id']}: {col} null mismatch")
                elif actual is not None and not np.isclose(
                    float(expected), float(actual), rtol=FLOAT_RTOL, atol=FLOAT_ATOL
                ):
                    problems.append(
                        f"row {row['model']}/{row['scene_id']}: {col} {expected} != {actual}"
                    )
        records.append({"experiment_id": exp, "ok": not problems, "problems": " | ".join(problems[:20])})
    return records


def audit_results(
    results_dir: Path,
    *,
    experiments: Iterable[str] | None = None,
    write: bool = True,
    max_runs: int | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    if experiments is None:
        exp_list = None
    else:
        exp_list = [str(e) for e in experiments]
    run_dirs = discover_run_dirs(results_dir, None)
    if exp_list is not None:
        wanted = set(exp_list)
        run_dirs = [
            d for d in run_dirs if d.relative_to(results_dir).parts[0] in wanted
        ]
    if max_runs is not None:
        run_dirs = run_dirs[: int(max_runs)]
    records = [audit_run(d, results_dir) for d in run_dirs]
    table_records = audit_comparison_tables(results_dir)
    n_ok = sum(1 for r in records if r["ok"])
    summary = {
        "n_runs_audited": len(records),
        "n_ok": n_ok,
        "n_failed": len(records) - n_ok,
        "comparison_tables": table_records,
        "failures": [r for r in records if not r["ok"]][:50],
    }
    if write:
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(results_dir / "audit_report.csv", index=False)
        with (results_dir / "audit_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True, allow_nan=False)
    return summary
