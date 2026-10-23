"""仿真器：统一积分、道路边界、不重叠安全约束、僵持检测与指标统计。

一步的时间推进顺序（与报告“每个时间步怎样更新位置”一致）：
    1. 决策：所有 Agent 从同一时刻状态计算 (mode, v_target, y_target)
    2. 控制：所有 Agent 从同一时刻状态计算加速度 a(t)
    3. 积分：v(t+dt) = clip(v(t) + a(t)dt)；x(t+dt) = x(t) + v(t+dt)dt（半隐式欧拉）
    4. 约束：道路边界、行人不得后退、不重叠投影（记录违规事件）
    5. 记录与判定：指标、僵持、结束条件
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .agent import Agent, World
from .common import clearance_available, required_clearance
from .controllers import control
from .decision import decide_car, decide_pedestrian


@dataclass
class RunSpec:
    """一次仿真的输入。"""

    run_id: str
    width_m: float
    rule: str
    model: str
    initial_conditions: list[dict[str, Any]]
    save_trace: bool = True


@dataclass
class RunResult:
    """一次仿真的汇总结果（一行 metrics）。"""

    run_id: str
    width_m: float
    rule: str
    model: str
    outcome: str                     # PASS / DEADLOCK / TIMEOUT
    t_end_s: float
    car_finish_t_s: float | None
    car_free_flow_t_s: float
    car_delay_s: float | None
    car_min_speed_mps: float
    car_max_decel_mps2: float
    car_stop_time_s: float
    car_stop_count: int
    min_pass_clearance_m: float      # 擦身而过时人车横向身体余量最小值
    min_body_distance_m: float       # 人车 2D 身体距离最小值（负 = 重叠）
    collision_events: int
    deadlock_t_s: float | None
    standoff_gap_m: float | None
    c_avail_m: float
    c_req_m: float
    margin_m: float
    peds_finished: int
    peds_total: int
    ped_delay_mean_s: float | None
    ped_delay_max_s: float | None
    ped_stop_time_total_s: float
    trace_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        for k, v in list(d.items()):
            if isinstance(v, np.floating):
                d[k] = float(v)
            elif isinstance(v, np.integer):
                d[k] = int(v)
        return d


# ---------------------------------------------------------------------------
# 构建世界
# ---------------------------------------------------------------------------
def build_world(spec: RunSpec, cfg: dict[str, Any]) -> World:
    car_cfg = cfg["car"]
    ped_cfg = cfg["pedestrians"]
    car = Agent(
        name="CAR",
        role="car",
        x=float(car_cfg["start_x_m"]),
        y=float(car_cfg["y_m"]),
        vx=-float(car_cfg["start_speed_mps"]),
        vy=0.0,
        length=float(car_cfg["length_m"]),
        width=float(car_cfg["width_m"]),
        v0=float(car_cfg["v0_mps"]),
        tau=float(car_cfg["tau_s"]),
        a_max=float(car_cfg["a_max_mps2"]),
        b_comfort=float(car_cfg["b_comfort_mps2"]),
        b_max=float(car_cfg["b_max_mps2"]),
        alpha_lat=0.0,
    )
    peds: list[Agent] = []
    for ic in spec.initial_conditions:
        y0 = float(ic["y_m"])
        peds.append(
            Agent(
                name=str(ic["name"]),
                role="ped",
                x=float(ic["x_m"]),
                y=y0,
                vx=float(ic["v0_mps"]),
                vy=0.0,
                length=float(ped_cfg["length_m"]),
                width=float(ped_cfg["width_m"]),
                v0=float(ic["v0_mps"]),
                tau=float(ped_cfg["tau_s"]),
                a_max=float(ped_cfg["a_max_mps2"]),
                b_comfort=float(ped_cfg["b_comfort_mps2"]),
                b_max=float(ped_cfg["b_max_mps2"]),
                alpha_lat=1.0,
                v_lat_max=float(ped_cfg["v_lat_max_mps"]),
                a_lat_max=float(ped_cfg["a_lat_max_mps2"]),
                tau_lat=float(ped_cfg["tau_lat_s"]),
                side_sign=1.0 if y0 >= 0.0 else -1.0,
                started_x=float(ic["x_m"]),
            )
        )
    return World(
        width=spec.width_m,
        x_min=float(cfg["road"]["x_min_m"]),
        x_max=float(cfg["road"]["x_max_m"]),
        car=car,
        peds=peds,
    )


# ---------------------------------------------------------------------------
# 几何：点到车辆矩形的有符号距离
# ---------------------------------------------------------------------------
def _point_rect_signed_distance(
    px: float, py: float, cx: float, cy: float, half_len: float, half_wid: float
) -> float:
    dx = abs(px - cx) - half_len
    dy = abs(py - cy) - half_wid
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def _ped_body_distance_to_car(ped: Agent, car: Agent) -> float:
    """行人外接圆到车辆矩形的身体距离（负值 = 重叠）。"""
    d = _point_rect_signed_distance(
        ped.x, ped.y, car.x, car.y, 0.5 * car.length, 0.5 * car.width
    )
    return d - ped.radius


# ---------------------------------------------------------------------------
# 积分与约束
# ---------------------------------------------------------------------------
def _integrate(world: World, dt: float, cfg: dict[str, Any]) -> None:
    ped_cfg = cfg["pedestrians"]
    car_cfg = cfg["car"]
    v_lat_max = float(ped_cfg["v_lat_max_mps"])
    v_ped_max = float(ped_cfg["v_max_mps"])
    v_car_max = 1.6 * float(car_cfg["v0_mps"])

    for a in world.agents:
        a.vx += a.a_world[0] * dt
        a.vy += a.a_world[1] * dt
        if a.role == "ped":
            a.vx = max(a.vx, 0.0)            # 行人只前进 / 侧移 / 停止，不后退
            a.vy = _clip(a.vy, -v_lat_max, v_lat_max)
            sp = math.hypot(a.vx, a.vy)
            if sp > v_ped_max:
                scale = v_ped_max / sp
                a.vx *= scale
                a.vy *= scale
        else:
            a.vy = 0.0
            a.vx = _clip(a.vx, -v_car_max, 0.0)   # 车辆不倒车
        a.x += a.vx * dt
        a.y += a.vy * dt

    # 道路边界：行人不越出路缘
    y_lim = world.y_limit
    for ped in world.peds:
        if ped.y > y_lim:
            ped.y = y_lim
            ped.vy = 0.0
        elif ped.y < -y_lim:
            ped.y = -y_lim
            ped.vy = 0.0
    world.car.y = float(car_cfg["y_m"])


def _resolve_overlaps(world: World, cfg: dict[str, Any]) -> int:
    """硬安全约束：保证身体不重叠；返回触发的约束事件数。

    正常规则下不应触发；触发说明规则失效或参数不合理，会被计入 collision_events。
    """
    events = 0
    car = world.car
    g_stop = float(cfg["car"]["g_stop_m"])

    # 1) 车辆与横向重叠的行人：纵向净距不得小于 g_stop（车头不压人）
    for ped in world.peds:
        if car.clearance_to(ped) < 0.0:
            gap = car.gap_to(ped)
            if gap < g_stop:
                # gap = car.x - ped.x - (Lc+Lp)/2  ->  解出 car.x
                car.x = ped.x + 0.5 * (car.length + ped.length) + g_stop
                car.vx = 0.0
                events += 1

    # 2) 行人外接圆与车辆矩形：若两个方向都重叠，把行人横向推到车侧
    for ped in world.peds:
        if abs(ped.x - car.x) < 0.5 * car.length + ped.radius and abs(
            ped.y - car.y
        ) < 0.5 * car.width + ped.radius:
            side = 1.0 if ped.y >= car.y else -1.0
            ped.y = car.y + side * (0.5 * car.width + ped.radius)
            ped.vy = 0.0
            events += 1

    # 3) 行人与行人
    for i in range(len(world.peds)):
        for j in range(i + 1, len(world.peds)):
            a, b = world.peds[i], world.peds[j]
            d = math.hypot(a.x - b.x, a.y - b.y)
            r_sum = 0.5 * (a.width + b.width)
            if d < r_sum - 1e-12:
                events += 1
                ux = (a.x - b.x) / d if d > 1e-9 else 1.0
                uy = (a.y - b.y) / d if d > 1e-9 else 0.0
                push = 0.5 * (r_sum - d)
                a.x += ux * push
                a.y += uy * push
                b.x -= ux * push
                b.y -= uy * push
                a.vy = b.vy = 0.0
    y_lim = world.y_limit
    for ped in world.peds:
        ped.y = _clip(ped.y, -y_lim, y_lim)
    return events


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _snapshot(world: World, spec: RunSpec, t: float) -> list[dict[str, Any]]:
    car = world.car
    rows = []
    for a in world.agents:
        rows.append(
            {
                "run_id": spec.run_id,
                "t": round(t, 6),
                "agent": a.name,
                "role": a.role,
                "x": round(a.x, 6),
                "y": round(a.y, 6),
                "vx": round(a.vx, 6),
                "vy": round(a.vy, 6),
                "speed": round(a.speed, 6),
                "mode": a.mode,
                "v_target": round(a.v_target, 6),
                "a_long": round(a.a_long, 6),
                "gap_to_car": round(a.gap_to(car), 6) if a is not car else float("nan"),
                "clearance_to_car": (
                    round(a.clearance_to(car), 6) if a is not car else float("nan")
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def simulate(
    spec: RunSpec, cfg: dict[str, Any]
) -> tuple[RunResult, pd.DataFrame | None, list[dict[str, Any]]]:
    dt = float(cfg["meta"]["dt_s"])
    t_max = float(cfg["meta"]["t_max_s"])
    record_every = float(cfg["meta"]["record_every_s"])
    rule = cfg["rules"][spec.rule]
    metrics_cfg = cfg["metrics"]
    ped_finish_dist = float(metrics_cfg["ped_finish_distance_m"])
    stop_speed = float(metrics_cfg["stop_speed_mps"])
    dl_speed = float(metrics_cfg["deadlock_speed_mps"])
    dl_hold = float(metrics_cfg["deadlock_hold_s"])
    passing_zone = float(metrics_cfg["passing_zone_m"])
    dl_window = float(metrics_cfg["deadlock_window_m"])

    world = build_world(spec, cfg)
    car = world.car
    car_free_flow_t = (
        float(cfg["car"]["start_x_m"]) - float(cfg["car"]["finish_x_m"])
    ) / float(cfg["car"]["v0_mps"])
    c_avail = clearance_available(
        spec.width_m, float(cfg["car"]["width_m"]), float(cfg["pedestrians"]["width_m"])
    )
    c_req = required_clearance(rule, 0.0)

    records: list[dict[str, Any]] = []
    if spec.save_trace:
        records.extend(_snapshot(world, spec, 0.0))
    next_record = record_every

    # 指标累计
    min_body_distance = math.inf
    min_pass_clearance = math.inf
    min_clearance_per_ped: dict[str, float] = {p.name: math.inf for p in world.peds}
    collision_events = 0
    car_stop_time = 0.0
    car_stop_count = 0
    car_was_moving = True
    car_min_speed = math.inf
    car_max_decel = 0.0
    deadlock_t: float | None = None
    standoff_gap: float | None = None
    car_finish_t: float | None = None
    ped_finish_t: dict[str, float] = {}
    ped_stop_time: dict[str, float] = {p.name: 0.0 for p in world.peds}

    # 僵持检测用位置历史（每 hist_dt 秒一帧，窗口 dl_hold 秒）
    pos_hist: list[tuple[float, float, dict[str, tuple[float, float]]]] = []
    hist_dt = 0.25
    next_hist = 0.0

    t = 0.0
    outcome = "TIMEOUT"
    while t <= t_max + 1e-9:
        world.t = t

        # ---- 1. 决策 -----------------------------------------------------
        decisions = {}
        for ped in world.peds:
            decisions[ped.name] = decide_pedestrian(ped, world, cfg, rule)
        decisions[car.name] = decide_car(car, world, cfg, rule)

        # ---- 2. 控制 -----------------------------------------------------
        for a in world.agents:
            ax, ay, a_long = control(a, world, decisions[a.name], spec.model, cfg)
            a.a_world = (ax, ay)
            a.a_long = a_long
            a.mode = decisions[a.name].mode
            a.v_target = decisions[a.name].v_target
            a.y_target = decisions[a.name].y_target

        # ---- 3. 积分 -----------------------------------------------------
        _integrate(world, dt, cfg)

        # ---- 4. 安全约束 -------------------------------------------------
        collision_events += _resolve_overlaps(world, cfg)

        # ---- 5. 指标 -----------------------------------------------------
        for ped in world.peds:
            d_body = _ped_body_distance_to_car(ped, car)
            min_body_distance = min(min_body_distance, d_body)
            clearance = car.clearance_to(ped)
            min_clearance_per_ped[ped.name] = min(
                min_clearance_per_ped[ped.name], clearance
            )
            if abs(ped.x - car.x) <= passing_zone:
                min_pass_clearance = min(min_pass_clearance, clearance)
            if ped.speed < stop_speed:
                ped_stop_time[ped.name] += dt

        car_speed = car.speed
        car_min_speed = min(car_min_speed, car_speed)
        car_max_decel = min(car_max_decel, -float(car.a_world[0]))  # 车头朝 -x
        if car_speed < stop_speed:
            if car_was_moving:
                car_stop_count += 1
            car_stop_time += dt
            car_was_moving = False
        else:
            car_was_moving = True

        # ---- 6. 结束状态 -------------------------------------------------
        t += dt
        world.t = t
        if car_finish_t is None and car.x < float(cfg["car"]["finish_x_m"]):
            car_finish_t = t
        for ped in world.peds:
            # 行人“完成”= 前进 60 m 且已被车辆完全驶过（否则会把人堵在车头前、
            # 但因累计前进已超过 60 m 而误判为完成）。
            already_passed = (car.x + 0.5 * car.length) < ped.x
            if (
                ped.name not in ped_finish_t
                and already_passed
                and ped.x - ped.started_x >= ped_finish_dist
            ):
                ped_finish_t[ped.name] = t
                ped.finish_time = t

        # 僵持判定：车停住 + 前方存在“按当前规则仍不可通过”的行人，
        # 且过去 dl_hold 秒内双方净位移都小于 0.25 m（允许微小蠕行/抖动）。
        # 注意：c_req 在 v=0 处取值；对“刚好卡在阈值上”的宽度加 0.05 m 容差，
        # 否则 L=2.9（c_avail=0.05）这类边界情形会因为浮点比较被判成“可通过”。
        blocking: list[Agent] = []
        for ped in world.peds:
            gap = car.gap_to(ped)
            if -5.0 < gap < dl_window and car.clearance_to(ped) < c_req + 0.05:
                blocking.append(ped)

        if t >= next_hist - 1e-9:
            pos_hist.append((t, car.x, {p.name: (p.x, p.y) for p in world.peds}))
            next_hist += hist_dt
            while pos_hist and t - pos_hist[0][0] > 2.0 * dl_hold:
                pos_hist.pop(0)

        if len(blocking) > 0 and pos_hist:
            ref = None
            for entry in pos_hist:
                if t - entry[0] >= dl_hold:
                    ref = entry
                else:
                    break
            if ref is not None:
                d_car = abs(car.x - ref[1])
                d_peds = [
                    math.hypot(p.x - ref[2][p.name][0], p.y - ref[2][p.name][1])
                    for p in blocking
                    if p.name in ref[2]
                ]
                no_progress = d_car < 0.25 and d_peds and all(d < 0.25 for d in d_peds)
                if (
                    no_progress
                    and car.speed < max(dl_speed * 3.0, 0.3)
                    and car_finish_t is None
                ):
                    deadlock_t = t
                    standoff_gap = min(car.gap_to(p) for p in blocking)
                    outcome = "DEADLOCK"
                    break

        # ---- 7. 记录 -----------------------------------------------------
        if spec.save_trace and t >= next_record - 1e-9:
            records.extend(_snapshot(world, spec, t))
            next_record += record_every

        if car_finish_t is not None and len(ped_finish_t) == len(world.peds):
            outcome = "PASS"
            break
        if t > t_max:
            outcome = "TIMEOUT"
            break

    # ------------------------------------------------------------------
    # 收尾指标
    # ------------------------------------------------------------------
    if outcome != "DEADLOCK":
        outcome = "PASS" if car_finish_t is not None else "TIMEOUT"
    car_delay = None if car_finish_t is None else car_finish_t - car_free_flow_t
    ped_delays = [
        ped_finish_t[p.name] - ped_finish_dist / p.v0
        for p in world.peds
        if p.name in ped_finish_t
    ]
    ped_metrics: list[dict[str, Any]] = []
    for p in world.peds:
        ped_metrics.append(
            {
                "run_id": spec.run_id,
                "width_m": spec.width_m,
                "rule": spec.rule,
                "model": spec.model,
                "ped": p.name,
                "v0_mps": p.v0,
                "start_x_m": p.started_x,
                "start_y_m": p.y,
                "finished": p.name in ped_finish_t,
                "finish_t_s": ped_finish_t.get(p.name),
                "delay_s": (
                    ped_finish_t[p.name] - ped_finish_dist / p.v0
                    if p.name in ped_finish_t
                    else None
                ),
                "stop_time_s": ped_stop_time[p.name],
                "final_y_m": p.y,
                "side": "left(+y)" if p.y >= 0 else "right(-y)",
                "min_clearance_to_car_m": (
                    round(min_clearance_per_ped[p.name], 6)
                    if math.isfinite(min_clearance_per_ped[p.name])
                    else None
                ),
            }
        )

    trace_df = pd.DataFrame(records) if records else None
    result = RunResult(
        run_id=spec.run_id,
        width_m=spec.width_m,
        rule=spec.rule,
        model=spec.model,
        outcome=outcome,
        t_end_s=round(t, 6),
        car_finish_t_s=car_finish_t,
        car_free_flow_t_s=car_free_flow_t,
        car_delay_s=car_delay,
        car_min_speed_mps=round(car_min_speed, 6) if math.isfinite(car_min_speed) else None,
        car_max_decel_mps2=round(-car_max_decel, 6) if car_max_decel < 0 else 0.0,
        car_stop_time_s=round(car_stop_time, 6),
        car_stop_count=car_stop_count,
        min_pass_clearance_m=(
            round(min_pass_clearance, 6) if math.isfinite(min_pass_clearance) else float("nan")
        ),
        min_body_distance_m=round(min_body_distance, 6),
        collision_events=collision_events,
        deadlock_t_s=deadlock_t,
        standoff_gap_m=(round(standoff_gap, 6) if standoff_gap is not None else None),
        c_avail_m=round(c_avail, 6),
        c_req_m=round(c_req, 6),
        margin_m=round(c_avail - c_req, 6),
        peds_finished=len(ped_finish_t),
        peds_total=len(world.peds),
        ped_delay_mean_s=(round(float(np.mean(ped_delays)), 6) if ped_delays else None),
        ped_delay_max_s=(round(float(np.max(ped_delays)), 6) if ped_delays else None),
        ped_stop_time_total_s=round(sum(ped_stop_time.values()), 6),
    )
    return result, trace_df, ped_metrics

