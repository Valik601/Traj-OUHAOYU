# uavbench 内部接口契约（冻结版 v1）

本文件是 `HANDOFF_UAV_ANTIPODAL_V2.md` 的工程实现契约。所有实现者先读本文件，再读 handoff 对应章节。
规范冲突时以 handoff 为准，并立即通知 Lead 修订本文件。**未写入本文件的接口不得擅自改动**；
新增字段只能追加，不能改名。

数值约定：全部几何/指标用 float64；长度容差 `1e-8 m`；速度容差 `1e-8 m/s`；
JSON 中缺失值一律 `null`，禁止 NaN/Infinity；随机流按用途分离。

---

## 1. 目录与所有权

```text
uav_orca_experiment/
  pyproject.toml                 Lead
  requirements-lock.txt          Lead
  README.md                      Lead
  THIRD_PARTY_NOTICES.md         orca-dev 起草 → Lead 终审
  CONTRACTS.md                   Lead
  configs/{default,smoke,full,sensitivity}.yaml   Lead
  uavbench/__init__.py           Lead
  uavbench/config.py             Lead
  uavbench/geometry.py           Lead          （共享几何/连续安全原语）
  uavbench/initialization.py     Lead
  uavbench/navigation.py         Lead
  uavbench/dynamics.py           Lead
  uavbench/simulation.py         Lead          （唯一状态更新者）
  uavbench/scheduling.py         Lead
  uavbench/metrics.py            metrics-dev
  uavbench/statistics.py         Lead
  uavbench/runner.py             Lead
  uavbench/audit.py              Lead
  uavbench/cli.py                Lead
  uavbench/visualization.py      viz-dev
  uavbench/planners/__init__.py  Lead
  uavbench/planners/base.py      Lead
  uavbench/planners/direct.py    Lead
  uavbench/planners/candidates.py vo-rvo-dev
  uavbench/planners/vo3d.py      vo-rvo-dev
  uavbench/planners/rvo3d.py     vo-rvo-dev
  uavbench/planners/orca3d.py    orca-dev
  tests/conftest.py              Lead
  tests/test_*.py                按文件头注释的 OWNER 行归属；其余 Lead
  scenes/  results/  reports/    Lead 调度
```

写作用域规则：只写自己名下的文件；需要改别人的文件先发消息。测试文件各自独立命名，避免冲突。
运行 pytest 前确认没有别人正在写同目录（各模块自测时用 `pytest tests/test_<own>.py`）。

---

## 2. 包级常量与枚举（`uavbench/geometry.py`, `uavbench/navigation.py`）

```python
# navigation.py
STAGE_WAITING: int = 0
STAGE_TO_CENTER: int = 1
STAGE_TO_GOAL: int = 2
STAGE_ARRIVED: int = 3
STAGE_NAMES = {0: "waiting", 1: "to_center", 2: "to_goal", 3: "arrived"}

# geometry.py
GEOM_EPS: float = 1e-8          # 长度容差 m
VEL_EPS: float = 1e-8           # 速度容差 m/s

def segment_point_min_dist_sq(p0, p1, origin) -> float
    # 线段 p0->p1 到原点的最近距离平方；p0,p1 形状 (...,3)，支持广播。

def ray_sphere_first_hit(p, d, radius, t_max) -> float | None
    # 射线 p + t*d, t in [0,t_max]，返回首次与球面相交的 t（浮点），无交返回 None。
    # 起点在球内返回 0.0。严格按 float64。

def point_segment_distance(...) -> ...

def segment_sphere_intervals(p, dp, radius) -> (t_enter, t_exit) | None
    # 单步位移 dp 下 ||p + t*dp|| < radius 的 t 区间（t in [0,1]），
    # 起点在球内 → (0.0, t_exit)。用于中心访问/连续侵犯事件求根。
```

`metrics.py` 可调用这些原语，也允许在 `metrics.py` 内实现等价向量化版本（离线重算优先性能），
但必须与 `geometry.py` 的标量版本在测试中一致。

---

## 3. 配置（`configs/*.yaml` + `uavbench/config.py`）

YAML 顶层键（schema_version=1）：

```yaml
schema_version: 1
experiment:
  name: uav_antipodal_v2
  observation_time: 120.0     # T
  dt: 0.1
simulation:
  n_agents: 100
  radius: 0.5
  d_safe: 1.0                 # 评价阈值
  d_plan: 1.0                 # 规划距离
  init_sphere_radius: 20.0
  v_max: 2.0
  goal_tolerance: 0.3
  center_tolerance: 0.1
  initial_velocity: zero
  neighbor_mode: all          # all | range
  neighbor_range: null        # range 模式下的感知半径 m
  perturbation:
    mode: tangential_gaussian # none | tangential_gaussian
    max_angle_deg: 2.0
    resample_max: 100
planners:
  candidate:
    directions: 128
    speed_layers: [0.5, 1.0, 2.0]
    include_zero: true
    include_current: true
    include_pref: true
  vo:   {tau: 5.0}
  rvo:  {tau: 5.0}
  orca: {tau: 5.0}
  backend: auto               # auto | python | numba（ORCA 求解后端）
scheduling:
  delta: 0.75
  order: random_stream_0      # fixed_id | random_stream_0 | random_stream_1
metrics:
  eps_distance: 1e-8
  stall_window: 5.0
  stall_path_threshold: 0.1
  stall_waypoint_threshold: 0.1
  bootstrap_resamples: 10000
  bootstrap_seed: 20261030
scenes:
  base_seed: 1000
  n_formal_scenes: 20
  development_seeds: [0, 1, 2, 3, 4]
  formal_seeds: [1000, ..., 1019]
  sensitivity_seeds: [1000, 1001, 1002, 1003, 1004]
models:
  A: [direct, vo, rvo, orca]
  B: [direct, vo, rvo, orca, orca_release, direct_release]
```

`uavbench/config.py`：

```python
@dataclass(frozen=True)
class Config:            # 嵌套 dataclass，字段与 YAML 一一对应
    ...
    def to_json_dict(self) -> dict
    def hash(self) -> str       # sha256 of canonical JSON (sort_keys, separators=(',',':'))

def load_config(path: str | Path) -> Config
def apply_overrides(cfg: Config, overrides: dict) -> Config   # 敏感性实验用，点号路径→值
def validate_config(cfg: Config) -> list[str]                 # 返回错误列表，空=通过
```

`config_hash = sha256(canonical_json(effective_config_including_overrides))`，用于结果目录与 resume 判据。
敏感性运行把 override 写入 config.json 与目录名（config_hash 自动区分）。

---

## 4. 场景（`uavbench/initialization.py`）

`scenes/scene_<id>.npz` 键（全部 float64，除非注明）：

```text
scene_id (str)          e.g. "1000" / "base_fibonacci"
seed (int64)            -1 表示确定性基础场景
starts (N,3)            p_i(0)
goals  (N,3)            g_i = -p_i(0)
agent_ids (N,) int64    arange(N) 稳定 ID
meta_json (str)         JSON：mode、角度上限、重采样次数、min_start_sep、min_goal_sep、
                        min_start_other_goal、cross_pairs_below_safety、hash 计算说明
```

`scenes/manifest.json`：`{"scenes": {"<scene_id>": {"seed":..., "sha256":..., "min_start_separation":...,
"min_goal_separation":..., "min_start_other_goal":..., "scene_file": "scene_<id>.npz",
"resample_count":...}, ...}}`。
`scene_hash` = npz 文件字节的 SHA-256（16 进制小写）。所有模型从同一 scene 文件读取，不得各自生成。

接口：

```python
def fibonacci_sphere(n: int, radius: float) -> np.ndarray        # (n,3)
def build_scene(cfg: Config, seed: int | None) -> Scene          # None → 确定性基础场景
@dataclass
class Scene:
    scene_id: str; seed: int; starts: np.ndarray; goals: np.ndarray
    agent_ids: np.ndarray; meta: dict; sha256: str
def save_scene(scene: Scene, scenes_dir: Path) -> Path
def load_scene(path: Path) -> Scene
def generate_scenes(cfg: Config, profile: str, scenes_dir: Path) -> list[Scene]
```

`profile`：`smoke` → 开发种子 0,1 加 base_fibonacci；`full` → formal_seeds（1000—1019）加 base_fibonacci；
`development` → 0—4；`sensitivity` → 1000—1004 加 base_fibonacci。重复生成时若 hash 与 manifest 一致则复用。

扰动规则严格按 handoff §3.2；退化重采样；起点/终点最小间距 < 1 m 整体重采样最多 100 次，失败抛
`InitializationError`。随机流：`scene_rng = default_rng([seed, 0])`。

---

## 5. 导航与动力学（`navigation.py`, `dynamics.py`）

```python
# navigation.py —— 纯函数，无状态
def desired_velocity(waypoint, position, v_max, dt) -> np.ndarray        # handoff §4.5
def waypoint_for_stage(stage, goal) -> np.ndarray | None                 # TO_CENTER→[0,0,0]; TO_GOAL→goal; WAITING/ARRIVED→None
def update_stages(...) -> ...   # 供 simulation 使用；状态机唯一定义处

# dynamics.py —— 恒速段推进
def advance_positions(positions, commands, active_mask, dt) -> np.ndarray
def step_segment_min_distances(...) -> ...   # 可选，sim 内不做全机对检测
```

状态机（handoff §4.2）：
- A：t=0 全部 `TO_GOAL`（无 WAITING）。
- B：M0—M3 及 C1/M4 由放行表决定 `WAITING→TO_CENTER`。C1 是 `direct`+放行表。
- 中心访问：步内首次 `||p(t)|| <= center_tolerance`，记录真实事件时刻；本步末 `k+1` 切 `TO_GOAL`。
- 到达：步末 `||p-g|| <= goal_tolerance` 且（B）已完成中心子任务 → `ARRIVED`，记录 `t_{k+1}`；下一步冻结。
- `WAITING`/`ARRIVED` 速度恒为 0，保留在邻居与安全检测中。

---

## 6. 规划器接口（`uavbench/planners/base.py`，冻结）

```python
@dataclass(frozen=True)
class PlanState:
    t: float                       # 当前网格时刻 t_k
    dt: float
    positions: np.ndarray          # (N,3) float64 全体
    velocities: np.ndarray         # (N,3) float64 当前速度（冻结者=0）
    stages: np.ndarray             # (N,) int8
    frozen: np.ndarray             # (N,) bool
    active_ids: np.ndarray         # (A,) int64 升序总体索引，A>=0
    waypoints: np.ndarray          # (A,3) 活动机当前 waypoint
    v_pref: np.ndarray             # (A,3) 期望速度
    neighbor_ids: np.ndarray       # (A,M) int64，无效位=-1
    neighbor_mask: np.ndarray      # (A,M) bool
    # 规划器不得访问 goals/scene 其它信息

@dataclass
class Diagnostics:
    sample_exhausted: np.ndarray        # (A,) bool
    used_fallback: np.ndarray           # (A,) bool
    solver_infeasible: np.ndarray       # (A,) bool  仅可证实不可行
    constraint_violation_max: np.ndarray# (A,) float64，本模型约束残差（见各模型说明）
    candidate_count_tested: np.ndarray  # (A,) int32
    initial_overlap_pairs: np.ndarray   # (A,) int32
    numerical_error: np.ndarray         # (A,) bool
    extra: dict[str, np.ndarray] | None # 追加可诊断字段，值形状 (A,)

@dataclass
class PlanResult:
    velocities: np.ndarray         # (A,3) float64，须有限且 ||v||<=v_max+tol
    diagnostics: Diagnostics

class Planner:
    name: str                      # "direct" | "vo" | "rvo" | "orca"
    def __init__(self, cfg: Config, candidate_set: CandidateSet | None = None): ...
    def plan(self, state: PlanState) -> PlanResult: ...
```

约束：规划器只读快照，**不得**修改位置/阶段/累计指标；不得自行推进时间；不得有隐藏可调参数。
`candidate_set` 由场景级 `candidate_rng` 旋转后构建一次，M1/M2 共享同一对象（`CandidateSet.dirs` 只读）。
`plan()` 必须可重复：同一输入两次调用输出逐位一致（ORCA 后端一致时）。

---

## 7. 候选集合与共享搜索（`planners/candidates.py`，vo-rvo-dev）

```python
@dataclass(frozen=True)
class CandidateSet:
    dirs: np.ndarray       # (D,3) float64 单位方向，场景级旋转后
    speeds: tuple[float,...]
    def build(self, v_max, v_current, v_pref) -> np.ndarray   # (C,3)，固定顺序
    def build_index_map(self, n_cand) -> ...
```

- 基础方向：Fibonacci 球面 D=128（公式与 handoff §3.1 同构，但用于方向）。
- 速度层 `{0.5,1.0,2.0}`（层过滤：仅保留 <=v_max 的层；若层值 > v_max 则按 v_max 截断并去重）。
- 候选构造顺序（固定索引，tie-break 用）：先零速度、当前速度、v_pref，再 (方向×速度层) 按方向索引升序、层升序；
  去重容差 1e-9 m/s（保留首次出现索引）。总数 = min(387, ...)。
- M1/M2 使用**同一个** CandidateSet 与候选数组（v_current/v_pref 随代理不同，故 build 每步每机不同）。

危险判定（handoff §5.2，闭合线段 [0,tau] 定义，起点重叠必然判危险并计入 `initial_overlap_pairs`）：

```text
p_ij = p_j - p_i
VO : w = c - v_j
RVO: w = 2c - v_i - v_j
冻结邻机: w = c
危险 ⇔ min_{t in [0,tau]} ||p_ij - t*w|| < d_plan
```

- 共享搜索：候选按固定索引测试；选择规则：acceptable 中最小化 `||c-v_pref||²`，
  再 `||c-v_i||²`，再固定索引。实现可用“先测 v_pref 再批量”的等价加速，但必须与全量暴力结果逐位一致
  （测试覆盖）。
- 无可接受候选：`sample_exhausted=True`，回退排序（handoff §5.3）：对每个候选按
  邻机当前速度保持的物理预测计算全邻居最早 TTC（`w=c-v_j`，与模型无关；已重叠记 0；无接触 +inf），
  最大化最小 TTC → 最小化 `||c-v_pref||²` → 固定索引。`used_fallback=True`。
- 诊断：`candidate_count_tested` = 实际测试候选数（可选近似，但必须确定）；`initial_overlap_pairs` = 当前
  与冻结/活动邻机重叠（`||p_ij||<d_plan`）的数量；`constraint_violation_max` = 返回候选相对全部邻机的
  `max(0, d_plan - min_t dist)`（无邻机=0）。
- 性能目标：N=100、M=99、1200 步单模型单 run < 180 s（32 核单线程）；给出 `tests/test_planners_bench.py`
  可选基准（skip unless `--bench`）。禁止用 numba 掩盖未验证的几何。

---

## 8. 3D-ORCA（`planners/orca3d.py`，orca-dev）

来源：`https://github.com/snape/RVO2-3D`（Apache-2.0），`src/Agent.cc` 的 `linearProgram1/2/3/4`、
`Agent::computeNewVelocity`。必须忠实移植（float64、numpy），并在 `THIRD_PARTY_NOTICES.md` 记录 commit
（用一次 `git ls-remote` 或 API 查询确认后写入；同时记录 URL、license、修改点）。

约束构造（handoff §5.4）：

```text
活动—活动：alpha=0.5，相对速度 v_i - v_j，combinedRadius=d_plan，timeHorizon=tau
活动—冻结：alpha=1.0，相对速度 v_i - 0，冻结邻机不产生反向约束
冻结—冻结：不规划
plane.point = v_i + alpha * u;  plane.normal = n（单位，指向允许侧）
约束：n · (v - point) >= 0
```

- 重叠分支（`distSq <= combinedRadiusSq` 用 `invTimeStep` 恢复）忠实保留并测试；`alpha` 同样按上表。
- `u` 的截球/锥面两分支与官方一致。
- 求解：`linearProgram3`（可行）→ `linearProgram4`（最小最大违反回退）；`solver_fallback = (fail_index < m)`。
- 不可行证据（仅在可证实时置 `solver_infeasible=True`）：
  (a) 某平面 `n·point > v_max + VEL_EPS`；或 (b) 两约束反平行且 `n·point + n'·point' > VEL_EPS`
  （需用统一口径推导，并写入注释与测试）。求解器未收敛≠不可行。
- `constraint_violation_max` = 返回速度对全部约束的最大残差 `max(0, n·point - n·v)`（m/s）。
- 退化：完全重合/零相对速度时使用由 `(i,j)` 对 ID 决定、双向一致的确定性方向；官方 KD-tree 邻居顺序
  改为稳定 ID 升序（同步性测试要求遍历顺序无关）。
- 性能：可选 numba 后端（`planners.backend: auto|python|numba`），必须提供 python/numba 双后端一致性测试
  （容差 1e-9 m/s 或逐位一致），并在 provenance 记录实际后端。N=100、M=99、1200 步 < 180 s（python 后端即须达标）。

---

## 9. 仿真器（`simulation.py`，Lead）

唯一状态更新者；严格按 handoff §4.4 步序。每步：

1. 放行表推进（M4/C1）；2. 建快照；3. 活动机 waypoint/v_pref；4. 调用 `planner.plan`；
5. `p_next = p + v_cmd*dt`（冻结者 v=0）；6. 连续中心进入检测（真实事件时刻）；
7. `t_{k+1}` 切换 TO_GOAL；8. `t_{k+1}` 到达判定；9. 记录轨迹/事件/诊断/墙钟。

```python
@dataclass
class RunConfig:      # 由 Config + model + scenario + seed + overrides 派生
    ...

def run_simulation(cfg: Config, scene: Scene, model: str, scenario: str,
                   release_table: ReleaseTable | None) -> SimulationResult
```

`SimulationResult` 含：`t`, `positions`, `velocities`, `stages`, `frozen`, `active`、
`center_events`（首次访问者列表：agent_id, time, step_index）、到达记录、
`release_actual/release_planned`、`planner_diagnostics` 记录、每步规划墙钟、
`numerical_error` 标记、`valid` 标记。

---

## 10. 分时放行（`scheduling.py`，Lead）

```python
def build_release_table(cfg: Config, scene: Scene) -> ReleaseTable
# ReleaseTable: ranks (N,), planned (N,), release_tick (N,), actual (N,),
#               scenario_applied (bool)
```
- 排列：`schedule_rng = default_rng([seed, 1])`（random_stream_0）或 `[seed, 2]`（stream_1）；
  fixed_id 顺序 = 稳定 ID 升序。同一场景 M4/C1 复用完全相同表。
- 量化：`release_tick = ceil(planned / dt)`（用整数/十进制可靠实现，如 `Fraction`），
  `actual = tick*dt`；最后计划 74.25 s → actual 74.3 s（测试断言）。
- 时隙 0（Delta=0 或单机）退化为同时放行；所有指标从共同 t=0 起算。

---

## 11. 指标（`metrics.py`，metrics-dev）

输入：`trajectories.npz` + `scene.npz` + 仿真提供的中心首次进入时刻 + 有效配置 + 模型/场景/种子元信息。
输出（同一目录）：`metrics.json`、`agents.csv`、`violation_events.csv`、`boundary_contacts.csv`、
`min_distance_curve.csv`、`arrival_curve.csv`。全部从轨迹离线重算，禁止使用算法自报安全。

```python
# 鸭子类型输入：scene 只需有 .starts (N,3), .goals (N,3), .agent_ids (N,), .scene_id, .seed 属性
def compute_metrics(traj, scene, cfg: Config, *, model: str, scenario: str, seed: int,
                    scene_hash: str, config_hash: str,
                    center_entry_times: np.ndarray,       # (N,) float64, np.nan=未访问
                    center_entry_steps: np.ndarray | None = None,  # (N,) int64, -1=未访问
                    release_planned: np.ndarray | None = None,     # (N,) float64, nan=A
                    release_actual: np.ndarray | None = None,      # (N,) float64, nan=A
                    status: dict | None = None) -> MetricsResult
def write_metrics(result_dir: Path, result: MetricsResult) -> None
```
`traj` 是 `trajectories.npz` 读入后的 dict（键见 §9.3：`t, positions, velocities, stages, frozen,
active, dt, n_agents, observation_time`）。到达时刻由 `stages` 在网格上首次变为 `ARRIVED` 推出；
中心进入时刻只能由仿真提供（步内事件），metrics 用实际轨迹校验其一致性。
`MetricsResult` 至少含 `.metrics: dict`（§11.2）、`.agents: pandas.DataFrame`（§11.3 表头）、
`.violation_events`/`.boundary_contacts`/`.min_distance_curve`/`.arrival_curve` 均为 DataFrame。
`valid` 由调用方传入：数值错误/崩溃的运行在 metrics.json 中 `valid=false` 且 `status.numerical_error=true`，
指标仍尽量写出。

### 11.1 连续安全（handoff §6.1，统一 `d_eval = d_safe - eps_distance`）
- 遍历全部 `N(N-1)/2` 机对；每步 `r=p_i-p_j`, `w=v_i-v_j`；`t*=clip(-r·w/|w|²,0,dt)`，
  `d_min_step=||r+t* w||`；`w≈0` 用静态距离。
- 侵犯事件：`||r+t w||² < d_eval²` 的 t 区间（与 [0,dt] 相交），跨步合并；记录机对、起止、持续时间、
  最小距离及发生时刻；首侵犯时间=区间下确界。`d_safe` 与 `d_eval` 两侧都要单测。
- 边界接触：全程最小距离落入 `[d_safe-eps, d_safe+eps]` 的机对单独输出（不含严格侵犯者）。
- 中心首次进入：用 §2 原语从实际轨迹求最早进入时刻；仅第一次。

### 11.2 metrics.json 键（全部必须出现；缺失用 null）
```text
experiment_id, model, scenario, scene_id, seed, scene_hash, config_hash,
status: {completed, timeout, numerical_error, valid},
n_agents, arrival_count, arrival_rate, completed_all,
safe_completion_count, safe_completion_rate, center_crossing_count, center_crossing_rate(A:null),
team_safe_success,
min_separation, min_separation_time,
violating_pair_count, violation_event_count, violation_pair_seconds,
first_violation_time,
all_arrival_time, t90, mean_completion_time_limited,
average_path_length, mean_path_efficiency_excess, mean_path_over_40_minus_1,
completed_count_for_path_metrics,
waiting: {mean_wait, total_wait, mean_post_release_travel},
planning: {mean_ms, p95_ms, max_ms, total_s, calls, per_active_call_ms},
planner_diagnostics: {sample_exhausted_count, solver_fallback_count, solver_infeasible_count,
                      constraint_violation_max, initial_overlap_pairs_total,
                      numerical_error_count, active_planning_calls},
stall: {stalled_agent_count, low_progress_mobile_count},
uncompleted: {not_released, not_center, center_done_not_arrived}
```
- `safe_completion_rate`：120 s 内完成且 [0,T] 从未参与侵犯的机数/N（B 主要终点；A 同定义但无中心要求）。
- A 的 `center_crossing_rate=null`；`all_arrival_time` 未全部完成 → null；`t90` 不足 90 架 → null。
- `mean_path_efficiency_excess` 仅完成者：`mean(L_i/(2R-eps_goal) - 1)`；`mean_path_over_40_minus_1=mean_path/40-1`。
- `mean_completion_time_limited = mean(min(T_i,120))`（全部 N，含未完成→120），另标注完成数。
- 停滞规则严格按 handoff §6.3（5 s 窗口、0.1 m 路程、0.1 m waypoint 距离；排除计划等待/已到达；阶段切换重置）。

### 11.3 CSV 表头（冻结）
```text
agents.csv:
agent_id, release_rank, release_planned, release_actual, wait_duration,
center_entry_time, center_crossing, arrival_time, post_release_travel_time,
phase_wait_duration, phase_to_center_duration, phase_to_goal_duration,
path_length, final_goal_distance, final_stage,
violated, safe_completed, stalled, low_progress_mobile

violation_events.csv:
pair_id, agent_i, agent_j, start_time, end_time, duration, min_distance, min_distance_time

boundary_contacts.csv:
pair_id, agent_i, agent_j, min_distance, min_distance_time, min_time_inside_band

min_distance_curve.csv:
step, t, min_distance, violating_pair_count, boundary_pair_count

arrival_curve.csv:
step, t, cumulative_arrivals, cumulative_center_visits, cumulative_safe_completions_known_later
```

`agents.csv` 的 `safe_completed` 使用全窗口回溯（报告与图中要注明不是实时量）。
`path_length` = Σ||p_{k+1}-p_k|| 实际位移（含等待 0 位移）。

---

## 12. 结果目录与汇总（Lead）

```text
results/<experiment_id>/<scenario>/<model>/scene_<id>/<config_hash>/...
results/<experiment_id>/comparison.csv
results/<experiment_id>/summary_statistics.csv
results/<experiment_id>/paired_effects.csv
results/<experiment_id>/run_manifest.csv
```
`experiment_id`：`smoke` / `formal` / `sensitivity_<block>` / `stress`。
`experiment_id` 必须由 CLI profile/block 决定，禁止目录扫描混合开发与正式结果。
`comparison.csv` 列（冻结，handoff §9.3，顺序一致）：
```text
experiment_id, model, scenario, scene_id, seed, scene_hash, config_hash,
status, valid, arrival_rate, safe_completion_rate, center_crossing_rate,
min_separation, violating_pair_count, violation_event_count,
all_arrival_time, t90, average_path_length, completed,
team_safe_success, planning_time_mean, planning_time_p95,
sample_exhausted_count, solver_fallback_count
```
`status` ∈ {completed, timeout, numerical_error}；`valid` 为布尔。
`run_manifest.csv`：experiment_id, scenario, model, scene_id, seed, status, valid, config_hash,
scene_hash, runtime_s, source(cold|warm|resumed), error。
`--resume` 复用条件：scene_hash、config_hash、代码源 hash、后端均匹配且 audit 通过（metrics.json 存在且
audit 记录 ok）。无效运行保留记录、重跑后覆盖同目录并保留 `attempts` 计数于 manifest。

---

## 13. 可视化（`visualization.py`，viz-dev）

必须实现 handoff §10.1 的 9 类图；入口：
```python
def render_all(results_dir: Path, out_dir: Path, configs_dir: Path | None = None) -> list[Path]
def render_scene_seed(scenario: str, seed: int, results_dir: Path, out_dir: Path,
                      models: Sequence[str] | None = None) -> list[Path]   # CLI render
def render_animation(run_dir: Path, out_path: Path, fps: int = 25,
                     want_mp4: bool = False) -> Path                      # GIF 优先
```
（2026-10-30 修订：`render_scene_seed` 参数与 `cli.py render` 对齐；fig03 使用
`interval_min_separation` 从 npz 重算的区间连续最小值。）
- 主展示固定 seed 1000；另选一个预先声明规则确定的失败/中位案例（写进报告）。
- 缺 ffmpeg 不阻塞静态图；GIF 为最低要求。
- 轴等比例；图中标注 1 m 阈值、区间最小值含义、有效/完成样本数；缺失时间不画 0。
- 图片写到 `reports/figures/`，命名 `fig01_...png` … `fig09_...png`、`anim_<model>_B_seed1000.gif`。

---

## 14. 测试清单（各 OWNER 负责）

handoff §8 表全部覆盖。额外要求：
- 每个测试文件第一行注释 `# OWNER: <name>`。
- 测试必须**实际断言数值**；禁止 `pytest.skip` 掩盖失败（缺乏依赖的可选项除外，需在 README 披露）。
- 回归数值：精确端点 A/B：goal_tolerance=1e-8，A/B 到达率 1.0、T_all≈20 s、均路程≈40 m、
  d_min≈0、侵犯机对 4950；默认任务：dt=0.1、tol=0.3、T_all≈19.9 s、均路程≈39.8 m、
  B 首次中心进入≈9.95 s、控制阶段切换 10.0 s。
- 性能/基准测试标记 `@pytest.mark.bench`，默认跳过的用 `--bench` 开关（在 conftest 注册）。
- 测试不得依赖网络；不得写 `results/formal`。

---

## 15. 交付与核对（Lead）

最终核对：`python -m pytest -q` 全绿；`audit` 全部 ok；报告含真实数字；未执行项明确标注；
第三方来源、commit、license、适配点齐全；预定/有效/错误运行数、五模型对比、C1 消融、敏感性完成情况、
图/动画路径全部列出。任何声称必须有对应文件或日志证据。
