# uavbench：100 架无人机三维球面对跖点穿越避碰对比实验

> **Week 级一页总览：[`../results_all.png`](../results_all.png)** ·
> **Week 级 README：[`../README.md`](../README.md)**（核心结论、实验设计、敏感性、局限与复现）；
> 生成脚本 [`../make_overview.py`](../make_overview.py)。
>
> **仓库提交范围**：本 README、`CONTRACTS.md`、`THIRD_PARTY_NOTICES.md`、`pyproject.toml`、
> `requirements-lock.txt`、`configs/`、`uavbench/`、`tests/`，以及 week 级 README/results_all.png。
> `results/`（逐次运行数据）、`scenes/`（生成场景）、`reports/`（图/日志/审计明细）、`.venv/` 与缓存
> 均由本地生成，已在仓库根 `.GITIGNORE` 排除；按本文命令可完整重建。

实现规范见上级目录的 `HANDOFF_UAV_ANTIPODAL_V2.md`（本地输入，不提交），内部接口见
[`CONTRACTS.md`](CONTRACTS.md)。本项目比较 100 架球形无人机（半径 0.5 m，安全距离 2r = 1.0 m）在
理想运动学条件下完成**对跖点导航（A，辅助）**与**必须先访问中心区域的对跖点导航（B，主实验）**时，
三维 VO、RVO、ORCA 以及固定分时放行（ORCA + Staggered Release / Direct + Release 消融）的表现。

本实验是**理想速度控制仿真**：每步命令三维速度、瞬时改变、无加速度/姿态/传感器/通信误差，
无外部障碍与地面/墙面；碰撞后继续运动学积分以便完整量化风险（不代表真实碰撞后仍可飞行）。
安全阈值 1.0 m 等于两球半径之和，因此 `distance < 1.0 m` 即模型中球体已重叠。

## 安装

推荐使用与系统包共享的虚拟环境（离线可复现，Windows PowerShell）：

```powershell
cd uav_orca_experiment
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]" --no-build-isolation
```

依赖版本锁定见 `requirements-lock.txt`（numpy 1.26.4 / scipy 1.16.1 / pandas 1.5.3 /
matplotlib 3.7.4 / PyYAML 6.0 / pytest 7.4.0；可选 numba 0.57.0 加速 ORCA 求解后端）。
Python 需 ≥3.10（实测 3.11.4）。

## 快速开始

```powershell
python -m uavbench.cli validate-config --config configs/default.yaml
python -m uavbench.cli generate-scenes --profile full
python -m uavbench.cli run --model orca --scenario B --seed 1000
python -m uavbench.cli batch --profile smoke --jobs 8
python -m uavbench.cli batch --profile full --resume --jobs 24
python -m uavbench.cli batch --profile sensitivity --resume --jobs 24
python -m uavbench.cli batch --profile stress --jobs 8
python -m uavbench.cli audit --results results
python -m uavbench.cli summarize --results results
python -m uavbench.cli render --scenario B --seed 1000
python -m uavbench.cli render-all --results results --out reports/figures
python -m pytest -q
```

`configs/{default,full,smoke,development,sensitivity}.yaml` 通过顶层 `extends` 继承，只改需要改的键。
smoke 使用开发种子 0/1，仅用于联调与运行成本估计，不冒充正式结果。

## 实验协议（冻结）

| 项 | 值 |
|---|---|
| 无人机数 N | 100 |
| 半径 r / 安全距离 d_safe / 规划距离 d_plan | 0.5 m / 1.0 m / 1.0 m |
| 初始球面半径 R | 20 m；起点在球面，目标严格为对应起点的负值 |
| v_max / dt / 预测窗 tau | 2.0 m/s / 0.1 s / 5.0 s |
| 观察窗 T / 步数 | 120 s / 1200 |
| 目标容差 / 中心容差 | 0.3 m / 0.1 m |
| 主实验邻居 | 其余全部 99 架（22 m 为感知范围敏感性） |
| 分时放行间隔 Delta | 0.75 s（量化到 0.1 s 网格；最后实际放行 74.3 s） |
| 场景 | 基础 Fibonacci 确定性场景 + 20 个切向扰动正式场景（种子 1000—1019） |
| 模型 | A: direct, vo, rvo, orca；B: 再加 orca_release, direct_release |
| 正式重复 | 20 场景 × (4 + 6) = 200 次运行 |

时间推进规则、冻结体语义（活动机对冻结体承担全部避碰责任）、事件时序与五项课程指标见 handoff §4—§6。
指标由 `uavbench/metrics.py` 从保存轨迹离线重算，不信任算法自报安全。

## 目录

```text
configs/            default/full/smoke/development/sensitivity YAML
uavbench/           config, initialization, geometry, navigation, dynamics,
                    simulation(唯一状态更新者), scheduling, metrics(离线指标),
                    statistics(配对 bootstrap), runner(批量/恢复/落盘),
                    audit(离线重算审计), cli, visualization
uavbench/planners/  base, direct, candidates, vo3d, rvo3d, orca3d
tests/              验收测试（见 handoff §8）
scenes/             scene_*.npz + manifest.json（场景 SHA-256）
results/            <experiment_id>/<scenario>/<model>/scene_<id>/<config_hash>/
reports/            实验报告与图
```

每次运行目录包含 `config.json`、`provenance.json`（源码 hash、场景 hash、配置 hash、环境、
线程配置、命令、时间）、`trajectories.npz`、`agents.csv`、`center_events.csv`、
`violation_events.csv`、`boundary_contacts.csv`、`min_distance_curve.csv`、`arrival_curve.csv`、
`planner_diagnostics.csv`、`metrics.json`、`audit.json`、`run.log`。

## 复现与审计

- 场景、候选方向集旋转、放行顺序分别使用独立随机流（`uavbench/rng.py`）。
- `--resume` 仅复用场景 hash、配置 hash、代码源 hash、后端均匹配且 `audit.json` 为 ok 的运行。
- `audit` 从 `trajectories.npz` 离线重算指标并与 `metrics.json` 逐字段比对，同时校验
  `comparison.csv` 每行可追溯到实际运行目录。
- 正式运行开始前写出 `results/formal/experiment_manifest.json`；清单变化默认拒绝。

## 第三方来源

三维 ORCA 求解器忠实移植自 [RVO2-3D](https://github.com/snape/RVO2-3D)（Apache-2.0），
commit、修改点与适配说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 局限

理想速度控制、完美状态、有限场景族、静止终点策略、候选采样器质量、后端性能差异、
碰撞后继续积分；本研究不证明任意配置下的全局收敛或实机安全。零侵犯但长期停滞不等于任务成功，
失败、超时、停滞和间隔侵犯都如实保留在结果中。

## 实测结果摘要（2026-10-30，详见 `reports/experiment_report.md`）

- 实际完成 320 次运行：正式 200（A 4 模型 + B 6 模型 × 20 场景）、敏感性 90（5 个 block）、
  确定性压力 10、smoke 20；**全部 valid，0 次 numerical_error**；离线审计 **320/320 通过**。
- **Scenario A**（绕行可行）：四模型均 100% 到达；安全完成率 RVO 0.883 > ORCA 0.562 > VO 0.236 > Direct 0
  （ORCA − RVO 配对差 −0.321，95% CI [−0.368, −0.272]）。
- **Scenario B**（强制过中心）：局部避碰无法同时取得安全与进展——VO 完成 100% 但安全完成率 0.6%，
  ORCA 平均只到达 0.95/100 且 20/20 场景超时（停滞），RVO 到达 46.9%、安全完成率 0.9%。
  分时放行把 ORCA 的安全完成率提升到 0.166（配对差 +0.1635，CI [0.138, 0.191]），
  但 Direct+Release 也达到 0.173：**调度收益几乎全部来自降低同时在途数量，ORCA 在其上没有可测增量**
  （ORCA+Release − Direct+Release = −0.0065，CI [−0.031, 0.021]）。
- 无任何场景实现“100 架全部完成且零侵犯”（team_safe_success 全 0）；安全与活性必须分开讨论。

## 复现要点

- 场景、候选方向集旋转、放行顺序使用独立随机流；所有模型读取同一 `scenes/scene_*.npz`（SHA-256 记录）。
- 正式运行前冻结 `results/formal/experiment_manifest.json`；`--resume` 只复用三 hash 匹配且审计完成的运行。
- 每次运行的 `provenance.json` 记录代码源 hash、场景/配置 hash、环境、线程、命令与时间；
  `audit` 从 `trajectories.npz` 离线重算指标并与 `metrics.json` 逐字段比对。
