"""决策层：显式的交互规则（作业第 2 步）。

同一套规则引擎同时作用于行人与车辆，只是角色参数不同：
    行人：什么时候继续走 / 侧移 / 等待（速度可以降到 0），往哪一侧让
    车辆：什么时候减速 / 停车 / 恢复前进

规则集在 config.yaml 的 rules 中定义：

    实验 1（固定个人空间）
    R0  基准：车辆固定安全余量（横向余量 < 0.30 m 就停车等待）+ 行人 WAIT 时完全僵住
    R1  修改：车辆速度相关安全余量（余量足够时可低速蹭行）+ 行人等待时继续移向路缘
    A1/A2 单因素消融

    实验 2（速度相关个人空间 / 间隙接受）
    S_cons / S_mod / S_lib：共享接受关系 c_acc(v) = c0 + k·v。
        行人：c >= c_acc(v_car) 时接受、继续贴着车走（PASS）；
              c <  c_acc(v_car) 时让行，纵向期望速度直接置 0（STOP），
              速度可以一直降到 0（停在路缘等待），横向继续移向路缘。
        车辆：v_clear = speed_factor·(c − c0)/k，即“行人当前能接受的间隙”对应的速度。
    S_mod_nf：去掉社会力风险缩放的模型消融。

决策层只输出 (mode, v_target, y_target)，不直接产生加速度；加速度由控制层
（社会力 / Wiedemann）根据这组目标与感知结果生成。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .agent import Agent, World
from .common import acceptance_clearance, clearance_speed_cap


@dataclass
class Decision:
    mode: str
    v_target: float
    y_target: float


# ---------------------------------------------------------------------------
# 行人
# ---------------------------------------------------------------------------
def decide_pedestrian(
    ped: Agent, world: World, cfg: dict[str, Any], rule: dict[str, Any]
) -> Decision:
    """行人决策。

    分支 A —— 固定个人空间（R0/R1/A1/A2，ped_acceptance = fixed）：
      车辆未完全驶过、横向余量 < c_pass(0.30 m)、车距 <= d_alert 时：
        车距 > d_wait -> SIDESTEP（以 0.85·v0 继续前进并移向路缘）；
        车距 <= d_wait -> WAIT（停止前进；R0 同时冻结侧移，R1 继续移向路缘）。

    分支 B —— 速度相关接受（S_*，ped_acceptance = speed）：
      c_acc(v_car) 是行人当前愿意接受的横向余量：
        - clearance >= c_acc：接受，保持 v0 贴着车走（PASS）；
        - clearance <  c_acc：让行，v_target = 0（STOP，速度可降到 0），
                              横向继续移向路缘；车距 <= d_wait 时也停车。
      这一分支显式允许“行人速度下降到 0”。

    行人之间保持最小跟驰净距（跟驰安全速度约束）。
    """
    ped_cfg = cfg["pedestrians"]
    car = world.car
    y_edge = ped.side_sign * world.y_limit

    gap = ped.gap_to(car)                       # 车头到行人车尾的净距
    clearance = ped.clearance_to(car)           # 横向身体间隙
    car_rear = car.x + 0.5 * car.length         # 车辆尾部（x 较大一侧）
    ped_front = ped.x + 0.5 * ped.length
    car_passed = car_rear < ped_front - 0.5
    d_alert = float(ped_cfg["d_alert_m"])
    d_wait = float(ped_cfg["d_wait_m"])

    if rule.get("ped_acceptance") == "speed":
        # 行人观察车辆的“计划通过速度”（car.v_target，由仿真每步先算车辆决策）；
        # 用计划速度而不是瞬时速度，可以避免车辆还在减速时行人过早让行。
        v_ref = car.v_target
        c_accept = acceptance_clearance(rule, cfg, v_ref)
        in_conflict = (not car_passed) and clearance < c_accept and gap <= d_alert
        if in_conflict:
            ped.has_stepped = True
            y_target = y_edge
            mode = "STOP"                      # 让行：纵向速度降到 0
            v_target = 0.0
        else:
            if (not car_passed) and gap <= d_alert:
                mode = "PASS"                  # 接受小间隙，贴着车继续走
            else:
                mode = "WALK"
            v_target = ped.v0
            y_target = y_edge if ped.has_stepped else ped.y
    else:
        in_conflict = (
            (not car_passed)
            and clearance < float(ped_cfg["c_pass_m"])
            and gap <= d_alert
        )
        if in_conflict:
            if gap <= d_wait:
                mode = "WAIT"
                v_target = 0.0
                y_target = ped.y if rule["ped_freeze_on_wait"] else y_edge
            else:
                mode = "SIDESTEP"
                v_target = float(ped_cfg["sidestep_speed_factor"]) * ped.v0
                y_target = y_edge
            ped.has_stepped = True
        else:
            mode = "WALK"
            v_target = ped.v0
            y_target = y_edge if ped.has_stepped else ped.y

    # ---- 行人-行人跟驰（防止在路缘排队时互相挤压）-------------------------
    for other in world.peds:
        if other is ped:
            continue
        gap_other = ped.gap_to(other)
        if gap_other <= 0.0 or gap_other > float(ped_cfg["follow_range_m"]):
            continue
        if abs(other.y - ped.y) > float(ped_cfg["follow_width_m"]):
            continue
        free_gap = max(gap_other - float(ped_cfg["follow_gap_m"]), 0.0)
        v_safe = math.sqrt(2.0 * ped.b_max * free_gap)
        v_target = min(v_target, v_safe)

    return Decision(mode=mode, v_target=v_target, y_target=y_target)


# ---------------------------------------------------------------------------
# 车辆
# ---------------------------------------------------------------------------
def decide_car(
    car: Agent, world: World, cfg: dict[str, Any], rule: dict[str, Any]
) -> Decision:
    """车辆决策：

    对每一个前方（含并行）行人，取三类约束的最小值：
      a) 物理约束：若行人位于纵向行驶路径内（横向身体重叠，clearance < 0），
         车速不得超过“在 g_stop 处停住”的可行减速度曲线
             v_kin(gap) = sqrt(2*b*max(gap - g_stop, 0)) - v_ped；
      b) 规则约束：按横向余量给出允许通过速度 v_clear
           R0: 余量 >= c_req 才放行 v0，否则 0；
           R1 / S_*: v_clear = speed_factor·(clearance - c_min) / k
          - 若 v_clear = 0（规则不允许通过）：必须在 d_standoff 处停住，
                v_cap = sqrt(2*b*max(gap - d_standoff, 0)) - v_ped
          - 若 0 < v_clear < v0（允许低速蹭行）：按可行动力学剖面接近
                v_cap = sqrt(v_clear^2 + 2*b*max(gap - d_near, 0))
          - 若 v_clear >= v0：不加约束
      c) 正并行时（gap < 0）保持 v_clear，直到完全超过该行人。
    这样得到的速度曲线是“按舒适减速度可跟踪”的，不会出现规则要求停车、
    但车辆因惯性冲过停车线的现象。
    """
    car_cfg = cfg["car"]
    v0 = car.v0
    b = float(car_cfg["plan_decel_mps2"])
    d_conflict = float(car_cfg["d_conflict_m"])
    d_near = float(rule.get("car_approach_m", car_cfg["d_near_m"]))
    d_standoff = float(car_cfg["d_standoff_m"])
    g_stop = float(car_cfg["g_stop_m"])
    v_target = v0

    for ped in world.peds:
        gap = car.gap_to(ped)                  # 正值 = 行人在车前方
        if gap > d_conflict:
            continue
        if gap < -(car.length + 2.0):
            continue

        clearance = car.clearance_to(ped)
        # 行人正朝车辆走来时，间距闭合速度 = 车速 + 行人沿 x 的速度，
        # 停车/让行速度曲线必须把这一项算进去，否则会冲过停车线。
        ped_vx = max(0.0, ped.vx)

        # a) 路径内的物理障碍：紧急安全停车曲线
        if clearance < 0.0:
            v_kin = math.sqrt(2.0 * b * max(gap - g_stop, 0.0)) - ped_vx
            v_target = min(v_target, max(v_kin, 0.0))

        # b) 规则允许通过速度
        v_clear = clearance_speed_cap(rule, clearance, v0)
        if v_clear < v0 - 1e-9:
            if gap < 0.0:
                v_cap = v_clear                 # 正并行通过：保持低速直到完全超过
            elif v_clear <= 1e-9:
                v_cap = math.sqrt(2.0 * b * max(gap - d_standoff, 0.0)) - ped_vx
            else:
                v_cap = math.sqrt(v_clear * v_clear + 2.0 * b * max(gap - d_near, 0.0))
            v_target = min(v_target, max(v_cap, 0.0))

    v_target = max(v_target, 0.0)
    mode = "YIELD" if v_target < v0 - 1e-9 else "FREE"
    return Decision(mode=mode, v_target=v_target, y_target=0.0)
