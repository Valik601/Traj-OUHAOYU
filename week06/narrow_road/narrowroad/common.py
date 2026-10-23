"""公共定义：配置读取、带种子的初始条件、几何工具、可通行性解析判据。

坐标约定
--------
x : 沿道路方向（行人向 +x 走，车辆向 -x 驶来）
y : 横向，路中线 y = 0
    行人的期望方向 e = (+1, 0)，车辆 e = (-1, 0)
    行人的“右侧”为 -y（对向车辆行驶方向的右侧为 +y），但本实验只要求“就近避让”，
    因此侧向目标取行人初始位置所在一侧的路缘。

所有函数都是纯函数 / 无副作用，便于单元测试。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML 配置。"""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)!r}")


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def lateral_clearance(y_a: float, y_b: float, width_a: float, width_b: float) -> float:
    """两个物体横向的身体间隙（负值 = 横向投影重叠）。"""
    return abs(y_a - y_b) - 0.5 * (width_a + width_b)


def longitudinal_body_gap(
    x_a: float, x_b: float, length_a: float, length_b: float, heading_a: float
) -> float:
    """沿 a 的行驶方向，从 a 的车头到 b 的车尾的净距；正值表示 b 在前方。"""
    return heading_a * (x_b - x_a) - 0.5 * (length_a + length_b)


def clearance_available(width_m: float, car_width_m: float, ped_width_m: float) -> float:
    """解析判据：行人贴到路缘时，车辆与行人之间能得到的最大横向余量。

    行人圆心最大 |y| = L/2 - Wp/2；车辆半宽 Wc/2。
    c_avail = (L/2 - Wp/2) - (Wc/2 + Wp/2) = L/2 - Wc/2 - Wp
    """
    return 0.5 * width_m - 0.5 * car_width_m - ped_width_m


def required_clearance(rule: dict[str, Any], speed_mps: float = 0.0) -> float:
    """给定规则下车辆允许通过所需的横向余量。"""
    if rule["car_clearance"] == "fixed":
        return float(rule["car_c_req_m"])
    return float(rule["car_c_min_m"]) + float(rule["car_c_per_mps"]) * max(speed_mps, 0.0)


def clearance_speed_cap(rule: dict[str, Any], clearance_m: float, v0: float) -> float:
    """给定规则下，车辆在横向余量为 c 时允许的最高车速。

    - fixed 规则：c >= c_req 才允许以 v0 通过，否则为 0（必须停车等待）。
    - speed 规则：v_allow = (c - c_min) / k，c_min 为几何最小余量、k 为每 m/s 所需余量。
    """
    if rule["car_clearance"] == "fixed":
        return float(v0) if clearance_m >= float(rule["car_c_req_m"]) else 0.0
    return max(0.0, (clearance_m - float(rule["car_c_min_m"])) / float(rule["car_c_per_mps"]))


# ---------------------------------------------------------------------------
# 初始条件（只抽一次，所有情景共用，保证“只改变道路宽度”）
# ---------------------------------------------------------------------------
def draw_initial_conditions(cfg: dict[str, Any], seed: int | None = None) -> list[dict[str, Any]]:
    """抽取并冻结行人初始条件（x, y, 期望步速）。

    x 在 [x_min, x_max] 内均匀抽取，任意两人间距 >= min_spacing；
    y 在 [y_min, y_max] 内均匀抽取；v0 在 [v_min, v_max] 内均匀抽取。
    按 x 从大到小命名为 P1..Pn（P1 最先与车辆相遇）。
    """
    ped = cfg["pedestrians"]
    rng = np.random.default_rng(cfg["meta"]["seed"] if seed is None else seed)
    n = int(ped["count"])
    x_lo, x_hi = ped["x_range_m"]
    min_spacing = float(ped["min_spacing_m"])
    xs: list[float] = []
    while len(xs) < n:
        x = float(rng.uniform(x_lo, x_hi))
        if all(abs(x - others) >= min_spacing for others in xs):
            xs.append(x)
    ys = [float(v) for v in rng.uniform(*ped["y_range_m"], size=n)]
    vs = [float(v) for v in rng.uniform(*ped["v0_range_mps"], size=n)]
    order = np.argsort(xs)[::-1]  # x 从大到小：最先遇到车辆的人排第一
    out = []
    for rank, idx in enumerate(order, start=1):
        out.append(
            {
                "name": f"P{rank}",
                "x_m": round(xs[int(idx)], 6),
                "y_m": round(ys[int(idx)], 6),
                "v0_mps": round(vs[int(idx)], 6),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 解析预测
# ---------------------------------------------------------------------------
def analytic_predictions(cfg: dict[str, Any], widths: list[float]) -> list[dict[str, Any]]:
    """用解析判据给出“先预测”表：哪些宽度在哪个规则下可通过。

    重要：这里只是几何 + 规则阈值的解析判据，不含动力学仿真结果。
    """
    car_w = float(cfg["car"]["width_m"])
    ped_w = float(cfg["pedestrians"]["width_m"])
    rows = []
    for width in widths:
        c_avail = clearance_available(width, car_w, ped_w)
        row: dict[str, Any] = {"width_m": width, "c_avail_m": round(c_avail, 4)}
        for rname, rule in cfg["rules"].items():
            c_req = required_clearance(rule, 0.0)
            feasible = c_avail > c_req + 1e-9
            v_allow = clearance_speed_cap(rule, c_avail, float(cfg["car"]["v0_mps"]))
            row[f"{rname}_c_req_m"] = round(c_req, 4)
            row[f"{rname}_feasible"] = bool(feasible)
            row[f"{rname}_v_allow_mps"] = round(float(v_allow), 4)
        rows.append(row)
    return rows
