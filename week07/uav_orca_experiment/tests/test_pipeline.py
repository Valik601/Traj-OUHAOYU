# OWNER: lead
"""端到端流水线验收：mini 配置下 run → 落盘 → 指标 → 审计 → 汇总 → resume。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from uavbench.audit import audit_results
from uavbench.config import load_config
from uavbench.initialization import build_base_fibonacci_scene, build_scene, save_scene
from uavbench.runner import code_source_hash, execute_run, run_batch
from uavbench.statistics import COMPARISON_COLUMNS, load_runs, summarize_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_mini_root(root: Path) -> tuple[Path, Path, Path, Path]:
    configs = root / "configs"
    scenes = root / "scenes"
    results = root / "results"
    for p in (configs, scenes, results):
        p.mkdir(parents=True, exist_ok=True)
    cfg_dict = load_config(PROJECT_ROOT / "configs" / "default.yaml").to_json_dict()
    cfg_dict["experiment"]["name"] = "uav_antipodal_v2_mini"
    cfg_dict["experiment"]["observation_time"] = 30.0
    cfg_dict["simulation"]["n_agents"] = 4
    cfg_dict["scenes"]["n_formal_scenes"] = 1
    cfg_dict["scenes"]["formal_seeds"] = [0]
    cfg_dict["scenes"]["development_seeds"] = [0]
    cfg_dict["scenes"]["sensitivity_seeds"] = [0]
    cfg_dict["metrics"]["stall_window"] = 0.5
    cfg_dict["metrics"]["bootstrap_resamples"] = 200
    (configs / "default.yaml").write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for name in ("full", "smoke", "development", "sensitivity"):
        (configs / f"{name}.yaml").write_text("extends: default.yaml\n", encoding="utf-8")
    return configs, scenes, results, root


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("mini_root")
    configs, scenes, results, root = _make_mini_root(root)
    cfg = load_config(configs / "default.yaml")
    save_scene(build_base_fibonacci_scene(cfg), scenes)
    save_scene(build_scene(cfg, 0), scenes)
    return {"root": root, "configs": configs, "scenes": scenes, "results": results, "cfg": cfg}


def _run_kwargs(mini):
    return {
        "root_dir": str(PROJECT_ROOT),
        "configs_dir": str(mini["configs"]),
        "scenes_dir": str(mini["scenes"]),
        "results_dir": str(mini["results"]),
    }


def test_execute_run_writes_complete_artifacts(mini):
    spec = {
        "experiment_id": "mini",
        "scenario": "B",
        "model": "direct",
        "scene_id": "0",
        "seed": 0,
        "config_file": "default.yaml",
        "overrides": {},
        "note": "unit",
    }
    rec = execute_run(spec, **_run_kwargs(mini))
    assert rec["status"] == "completed"
    assert rec["valid"] is True
    assert rec["audit_ok"] is True
    out = Path(rec["out_dir"])
    expected = {
        "config.json",
        "provenance.json",
        "trajectories.npz",
        "agents.csv",
        "center_events.csv",
        "violation_events.csv",
        "min_distance_curve.csv",
        "arrival_curve.csv",
        "planner_diagnostics.csv",
        "metrics.json",
        "audit.json",
        "run.log",
    }
    assert expected <= {p.name for p in out.iterdir()}
    with (out / "metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    assert metrics["status"]["valid"] is True
    assert metrics["arrival_rate"] == pytest.approx(1.0)
    assert metrics["safe_completion_rate"] == pytest.approx(0.0)
    assert metrics["violating_pair_count"] == 6  # 4 机全部穿过中心 → 所有 6 个机对
    with (out / "provenance.json").open("r", encoding="utf-8") as fh:
        prov = json.load(fh)
    assert prov["source_hash"] == code_source_hash(PROJECT_ROOT)
    assert prov["config_hash"] == load_config(mini["configs"] / "default.yaml").hash()
    with np.load(out / "trajectories.npz", allow_pickle=False) as npz:
        assert npz["positions"].shape == (301, 4, 3)
        assert npz["t"].shape == (301,)
        assert "diag_t" in npz.files
        assert "diag_sample_exhausted" in npz.files


def test_batch_resume_audit_and_summarize(mini):
    summary = run_batch(
        "smoke",
        jobs=2,
        resume=False,
        configs_dir=mini["configs"],
        scenes_dir=mini["scenes"],
        results_dir=mini["results"],
        freeze=False,
    )
    assert summary["n_specs"] == 10  # 4 A + 6 B
    assert summary["n_executed"] == 10
    assert summary["n_completed"] >= 8
    assert summary["n_valid"] == 10
    assert (mini["results"] / "smoke" / "run_manifest.csv").exists()

    # resume 全部命中，不再执行
    summary2 = run_batch(
        "smoke",
        jobs=2,
        resume=True,
        configs_dir=mini["configs"],
        scenes_dir=mini["scenes"],
        results_dir=mini["results"],
        freeze=False,
    )
    assert summary2["n_resumed"] == 10
    assert summary2["n_executed"] == 0

    # 汇总（只统计 smoke 实验，前面 execute_run 的 mini 运行单独存在）
    res = summarize_results(
        mini["results"], experiments=["smoke"], config_path=mini["configs"] / "default.yaml"
    )
    assert res["n_runs"] == 10
    comp = mini["results"] / "smoke" / "comparison.csv"
    assert comp.exists()
    df = load_runs(mini["results"], ["smoke"])
    assert set(COMPARISON_COLUMNS) <= set(df.columns)
    assert df["safe_completion_rate"].notna().all()
    assert (mini["results"] / "smoke" / "summary_statistics.csv").exists()
    assert (mini["results"] / "numbers.json").exists()

    # 审计：全部通过且对照表可追溯
    audit = audit_results(mini["results"], experiments=["smoke"], write=True)
    assert audit["n_runs_audited"] == 10
    assert audit["n_failed"] == 0
    assert (mini["results"] / "audit_report.csv").exists()
    assert all(t["ok"] for t in audit["comparison_tables"])


def test_audit_detects_tampered_metrics(mini, tmp_path):
    # 复制一个运行目录并篡改 metrics.json
    src = next((mini["results"] / "smoke").rglob("metrics.json")).parent
    copied_results = tmp_path / "results"
    target = copied_results / "tampered" / "B" / "direct" / "scene_0" / "hash"
    target.mkdir(parents=True)
    shutil.copytree(src, target, dirs_exist_ok=True)
    # 修正 provenance 场景路径可解析性：沿用原 provenance（scene_file 相对项目根）
    with (target / "metrics.json").open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    metrics["arrival_rate"] = 0.123
    with (target / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True, allow_nan=False)
    audit = audit_results(copied_results, write=False)
    assert audit["n_runs_audited"] == 1
    assert audit["n_failed"] == 1
    assert "arrival_rate" in audit["failures"][0]["problems"]
