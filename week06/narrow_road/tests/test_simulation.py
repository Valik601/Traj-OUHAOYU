"""仿真层测试：结果判定、安全不变量、确定性、产物一致性。"""
from __future__ import annotations

import math

import pytest

from narrowroad.common import draw_initial_conditions, load_config
from narrowroad.experiments import build_specs, run_all
from narrowroad.simulate import RunSpec, simulate

CASES = [
    # width, rule, model, expected outcome
    (4.0, "R0", "sfm", "PASS"),
    (4.0, "R0", "wiedemann", "PASS"),
    (4.0, "R1", "sfm", "PASS"),
    (4.0, "R1", "wiedemann", "PASS"),
    (3.5, "R0", "sfm", "PASS"),
    (3.5, "R1", "wiedemann", "PASS"),
    (3.0, "R0", "sfm", "DEADLOCK"),
    (3.0, "R0", "wiedemann", "DEADLOCK"),
    (3.0, "R1", "sfm", "PASS"),
    (3.0, "R1", "wiedemann", "PASS"),
    (2.8, "R0", "sfm", "DEADLOCK"),
    (2.8, "R1", "wiedemann", "DEADLOCK"),
]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def ics(cfg):
    return draw_initial_conditions(cfg)


@pytest.mark.parametrize("width, rule, model, expected", CASES)
def test_outcomes(cfg, ics, width, rule, model, expected):
    spec = RunSpec(f"t_{width}_{rule}_{model}", width, rule, model, ics, save_trace=False)
    res, _, _ = simulate(spec, cfg)
    assert res.outcome == expected, f"{spec.run_id}: {res.outcome} != {expected}"
    # 不重叠、无硬约束事件
    assert res.min_body_distance_m > -1e-6
    assert res.collision_events == 0
    # 舒适减速度足够：从未超过紧急制动上限
    assert res.car_max_decel_mps2 <= cfg["car"]["b_max_mps2"] + 1e-6
    if expected == "PASS":
        assert res.car_finish_t_s is not None
        assert res.car_delay_s is not None
        assert res.car_delay_s >= -0.5      # 允许 0.05 s 的离散误差
        assert res.peds_finished == res.peds_total
        assert res.car_stop_count <= 1
    else:
        assert res.car_finish_t_s is None
        assert res.deadlock_t_s is not None
        assert res.standoff_gap_m is not None and res.standoff_gap_m > 0.5


@pytest.mark.parametrize(
    "width, rule, model",
    [(4.0, "R0", "sfm"), (3.0, "R1", "wiedemann"), (2.8, "R1", "sfm"), (3.0, "R0", "wiedemann")],
)
def test_kinematic_invariants(cfg, ics, width, rule, model):
    spec = RunSpec(f"inv_{width}_{rule}_{model}", width, rule, model, ics, save_trace=True)
    _, trace, _ = simulate(spec, cfg)
    assert trace is not None
    ped_lim = width / 2 - cfg["pedestrians"]["width_m"] / 2 + 1e-6
    for name, df in trace.groupby("agent"):
        if name == "CAR":
            assert (df["vx"] <= 1e-9).all()                  # 车辆不倒车
            assert (df["vy"].abs() <= 1e-12).all()           # 车辆不侧移
            assert (df["y"].abs() <= 1e-12).all()            # 车辆固定在 y = 0
        else:
            assert (df["vx"] >= -1e-9).all()                 # 行人不倒退
            assert (df["y"].abs() <= ped_lim).all()          # 行人不出路缘
            assert (df["speed"] <= cfg["pedestrians"]["v_max_mps"] + 1e-6).all()


def test_determinism(cfg, ics):
    spec = RunSpec("det", 3.0, "R1", "sfm", ics, save_trace=True)
    r1, t1, p1 = simulate(spec, cfg)
    r2, t2, p2 = simulate(spec, cfg)
    assert r1.as_dict() == r2.as_dict()
    assert p1 == p2
    assert t1 is not None and t2 is not None
    assert t1.equals(t2)


def test_prediction_consistency_and_artifacts(cfg, tmp_path):
    """完整跑一遍实验矩阵，检查解析预测与仿真结果一致、产物相互一致。"""
    summary = run_all(out_dir=tmp_path, verbose=False)
    assert summary["prediction_consistency"]["all_agree"] is True
    assert summary["prediction_consistency"]["n_checked"] == 16

    import pandas as pd

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    specs = build_specs(cfg, draw_initial_conditions(cfg))
    assert len(metrics) == len(specs)
    assert set(metrics["run_id"]) == {s.run_id for s in specs}

    # 指标内部一致
    for _, r in metrics.iterrows():
        assert r["c_avail_m"] == pytest.approx(
            r["width_m"] / 2 - cfg["car"]["width_m"] / 2 - cfg["pedestrians"]["width_m"],
            abs=1e-5,
        )
        assert r["margin_m"] == pytest.approx(r["c_avail_m"] - r["c_req_m"], abs=1e-5)
        if r["outcome"] == "PASS":
            assert r["car_delay_s"] == pytest.approx(
                r["car_finish_t_s"] - r["car_free_flow_t_s"], abs=1e-5
            )
        else:
            assert math.isnan(r["car_delay_s"]) or r["car_delay_s"] is None
            assert r["deadlock_t_s"] > 0

    # 轨迹与指标一致：由轨迹重算车辆通过时刻（0.1 s 采样误差内）
    traces_dir = tmp_path / "trajectories"
    checked = 0
    for _, r in metrics.iterrows():
        path = traces_dir / f"{r['run_id']}.csv"
        if not path.exists():
            continue
        tr = pd.read_csv(path)
        car = tr[tr["agent"] == "CAR"].sort_values("t")
        if r["outcome"] != "PASS":
            continue
        finish = car[car["x"] < cfg["car"]["finish_x_m"]]["t"]
        assert len(finish) > 0
        assert finish.iloc[0] == pytest.approx(r["car_finish_t_s"], abs=0.11)
        checked += 1
    assert checked >= 8

    # summary.json 与 predictions.json 一致
    preds = pd.read_json(tmp_path / "predictions.json")
    rows = summary["prediction_consistency"]["rows"]
    assert len(rows) == 16
    for row in rows:
        sub = preds[preds["width_m"] == row["width_m"]]
        assert not sub.empty
        assert bool(sub.iloc[0][f"{row['rule']}_feasible"]) == row["predicted_pass"]


def test_ablation_outcome_is_attributable_to_vehicle_rule(cfg, ics):
    """L=3.0：只改车规则(A1)即可恢复通行；只改行人规则(A2)仍僵持。"""
    results = {}
    for rule in ("R0", "A1", "A2", "R1"):
        res, _, _ = simulate(RunSpec(f"abl_{rule}", 3.0, rule, "sfm", ics, save_trace=False), cfg)
        results[rule] = res.outcome
    assert results["R0"] == "DEADLOCK"
    assert results["A2"] == "DEADLOCK"
    assert results["A1"] == "PASS"
    assert results["R1"] == "PASS"
