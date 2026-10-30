# Third-Party Notices

本文件记录 `uav_orca_experiment` 中第三方代码的来源、不可变版本、许可证与改动。
起草：orca-dev；终审：Lead。

---

## 1. RVO2-3D（三维 ORCA 官方参考实现）

| 项目 | 内容 |
|---|---|
| 名称 | RVO2-3D — Reciprocal Velocity Obstacles / ORCA for 3-D multi-agent navigation |
| 仓库 URL | <https://github.com/snape/RVO2-3D> |
| 不可变 commit | `f940bfba8bac9532ecae948f11639085e8cbf277`（`refs/heads/main`） |
| commit 获取方式 | `git ls-remote https://github.com/snape/RVO2-3D refs/heads/main`（2026-10-30；GitHub REST API 当时返回 403 rate-limit，因此按 CONTRACTS.md §8 使用 `git ls-remote` 并通过 raw.githubusercontent.com 逐文件读取） |
| 许可证 | Apache License, Version 2.0（`SPDX-License-Identifier: Apache-2.0`） |
| 版权 | `SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill`；作者 Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha |
| 许可证全文 | <https://www.apache.org/licenses/LICENSE-2.0> |

### 被参考/移植的上游文件

| 上游文件 | 使用内容 |
|---|---|
| `src/Agent.cc` | `Agent::computeNewVelocity` 的 ORCA 平面构造（截球 cut-off circle、锥面 cone、重叠 overlap 三个分支与 `RVO3D_EPSILON = 1e-5F`）；`linearProgram1`、`linearProgram2`、`linearProgram3`、`linearProgram4` 与 `Line` 辅助结构；`planeFail < orcaPlanes_.size()` 的 LP4 回退判据 |
| `src/Agent.h` | `maxSpeed_`、`timeHorizon_`、`radius_`、`prefVelocity_` / `newVelocity_` 语义 |
| `src/Vector3.h` | `abs`、`absSq`、`cross`、`normalize` 的数学语义 |
| `src/Plane.h`、`src/RVOSimulator.h` | `Plane{normal, point}` 与 `timeStep_`（重叠分支 `invTimeStep`）语义 |

### 本项目中的移植位置

| 文件 | 内容 |
|---|---|
| `uavbench/planners/orca3d.py` | RVO2-3D `computeNewVelocity` + `linearProgram1..4` 的 float64 NumPy 移植；生产路径 |
| `tests/test_orca_numeric.py` | 另有一份**逐行字面**的 float64 标量参考移植（`_ref_lp1..4`、`_ref_pair_plane`），仅用于差分核验，不参与生产 |
| `tests/test_orca.py` | 测试内用独立标量公式推导并硬编码期望的平面几何（注释与断言中标注数值来源） |

**本项目未链接、未编译、未分发 RVO2-3D 的 C++ 源码**；没有 C++ 构建依赖，只有上述重新实现的 Python 代码。
按 Apache-2.0 §4，上游版权声明在上文保留，凡实质性取自上游算法的部分均在此披露。

---

## 2. 相对上游的修改与适配（全部在 `uavbench/planners/orca3d.py`）

1. **float32 → float64**：上游以 `float` 编译；本移植全程 `numpy.float64`。
   `RVO3D_EPSILON = 1e-5` 是作用于**平方量**的比较阈值（`denominator²`、`|cross|²`、
   `|planeOptVelocity|²`）。为保持官方分支结构，移植保留字面值 `1e-5`（未改成 1e-8/1e-12）。
   **验证结论：不需要收紧**——随机可行实例上 LP 目标值与独立 SLSQP 参考最大差 `1.7e-12`，
   半空间最大残差 `1.2e-15 m/s`（见 §4），未发现通用可行实例上的次优解。
   如未来收紧该阈值，属算法参数变更，必须写入本文件并重跑全部数值测试。
2. **冻结球体适配（本实验协议，非上游）**：活动—活动 `alpha = 0.5`，相对速度 `v_i − v_j`；
   活动—冻结 `alpha = 1.0`，相对速度 `v_i − 0`，冻结邻机不产生反向约束；冻结—冻结不规划。
   `plane.point = v_i + alpha·u`，半空间 `n·(v − point) >= 0`。
3. **参数映射**：上游 `combinedRadius = radius_ + other->radius_` → 本实验 `combinedRadius = d_plan`（主实验 1.0 m）；
   `timeHorizon = tau = 5.0 s`；`maxSpeed = v_max = 2.0 m/s`；重叠分支 `sim_->timeStep_` → 仿真步长 `dt`（主实验 0.1 s）。
4. **邻机顺序**：上游按 KD-tree 距离顺序；本移植按**稳定升序 agent ID**（同步性契约要求遍历顺序无关）。
   顺序在 `ORCA3DPlanner.plan` 内显式排序，因此输入 `neighbor_ids` 的排列不影响结果。
5. **退化恢复约定（新增）**：上游在完全重合且相对速度为零、以及圆锥分支的径向退化中会出现 `0/0`；
   本移植在此类测度零几何上使用由 `(i, j)` 稳定 ID 对决定的单位方向（交换机对严格反号），
   并对非有限中间量做同样的确定性替换。该行为计入 `diagnostics.extra["degenerate_planes"]`；
   `numerical_error` 只在真正出现非有限中间量时置位。此约定属恢复约定，不恢复已丧失的安全保证。
6. **求解器实现形式**：保留官方 `linearProgram1..4` 的算法结构、分支条件与比较（含 `> 0.0`、
   `<= RVO3D_EPSILON` 等），但将对约束的扫描向量化（NumPy）；LP3 可行路径与 LP4 最小最大违反松弛的
   语义与 `planeFail < m` 回退判据不变。官方 LP 仍是本移植唯一的求解路径（没有改成其它 QP/采样近似）。
7. **按违反量排序的精确预跑（仅性能优化，不改变语义）**：`linearProgram3` 是增量投影；当每个子问题都
   成功时，其最终点就是 `v_pref` 到可行集的精确投影，**与平面枚举顺序无关**（前缀交集投影的归纳性质）。
   因此对初始违反平面数 > 2 的系统，先用按初始违反量降序（稳定排序，并列保持官方顺序）的同一
   `linearProgram3` 试算：成功即返回该精确最优解；失败则回退到**原封不动**的官方 ID 顺序
   `linearProgram3 → linearProgram4` 路径，因此不可行实例仍得到已记录的官方 LP4 松弛结果及其平面顺序。
   语义不变，只减少 LP2/LP1 子问题数；可行实例的差分测试（对逐行字面官方移植）验证两者在 1e-9 内一致。
   预跑成功后会再做一次残差复核（`max(residual) <= 1e-7`）；若复核失败（近并行 EPS 误分类的极端情形）
   则丢弃预跑结果并走官方路径，绝不把不可行点静默当可行解返回。
8. **诊断字段（新增，非上游）**：`used_fallback`（进入官方 LP4 松弛）、`solver_infeasible`（仅由
   廉价**充分**证书证明：某平面 `n·point > v_max + VEL_EPS`，或两反平行约束偏移不相容；
   求解器未收敛绝不置位）、`constraint_violation_max = max(0, n·point − n·v)`、
   `candidate_count_tested = ORCA 平面数`、`initial_overlap_pairs = ||p_ij|| < d_plan 的邻机数`、
   `extra = {n_planes, n_frozen_neighbors, degenerate_planes}`。
9. **速度上限容差修正**：LP 结果若因浮点误差超出 `v_max` 超过 1e-6 m/s 记 `numerical_error`，
   仅允许容差量级的球面投影（handoff §2.2），修正后重新计算约束残差。
10. **numba 后端未交付**：`planners.backend` 接受 `auto|python|numba`；本环境 numba 0.57.0 在
    `import numba` 时拒绝 numpy 1.26.4（`ImportError: Numba needs NumPy 1.24 or less`），且不允许
    `pip install`，无法对 jit 等价实现做 1e-9 奇偶校验。因此：
    - `auto` 确定性解析为 `"python"`（唯一生产后端），`active_backend()` 返回实际后端供 provenance 记录；
    - 显式 `backend: numba` 时抛出 `RuntimeError` 并说明原因，**不会静默替换后端**；
    - 不交付未经验证的 numba 代码。

---

## 3. 本项目的独立依赖（不随仓库分发）

运行时/测试依赖由 `pyproject.toml` 声明，不 vendored：NumPy（BSD-3-Clause）、SciPy（BSD-3-Clause，
仅测试中的 SLSQP 独立参考）、PyYAML（MIT）、pytest（MIT）、numba（BSD-2-Clause，可选，当前环境不可用）。
这些依赖的许可证由各自的发行包提供，不因本仓库的 Apache-2.0 移植代码而改变。

---

## 4. 运行与验证环境记录（2026-10-30）

| 项 | 值 |
|---|---|
| OS | Windows |
| Python | 3.11.4（Anaconda） |
| numpy | 1.26.4 |
| scipy | 1.16.1（仅测试参考） |
| numba | 0.57.0（`import` 失败：需 numpy ≤ 1.24；未使用） |
| pytest | 7.4.0 |
| 网络 | 测试运行时不访问网络 |
| 构建 | 纯 Python；无 C++ 编译步骤 |
| 线程 | 单线程（基准测试期间未启用并行） |

### 复现实验命令

```powershell
cd F:\OHYCode\Project\DATAPROCESSING\OpenTraj\Traj-OUHAOYU\week07\uav_orca_experiment
python -m pytest tests/test_orca.py tests/test_orca_numeric.py -q
# 性能基准（默认跳过，需 --bench）：
python -m pytest tests/test_orca_numeric.py -q --bench -k bench -s
```

### 已获得的验证数字（详见测试与对话报告）

- `tests/test_orca.py`（17 项）与 `tests/test_orca_numeric.py`（25 项 + 1 项基准）全部通过。
- 随机可行实例：半空间最大残差 `1.22e-15 m/s`；`||v|| <= v_max + 1e-8`。
- 与独立 SLSQP 参考的目标值最大差 `1.68e-12`（290 个 SLSQP 收敛实例，`m = 1..5`）。
- 与逐行字面标量移植的差分核验：400 组随机平面系统（LP3/LP4 输出）、300 组平面构造、
  200 组 `solve_orca_velocity` 级可行/不可行混合实例，均 `atol = 1e-9`。
- 性能基准（python 后端，100 机 / 99 邻机 / 1200 步，单线程）：
  **37.52 s（31.27 ms/步）< 180 s**（优化前同基准 68.65 s）。
- 密集压力场景（100 机全部持续涌向同一点、1200 步外推）：约 231 s，仅用于评估
  Scenario B 拥堵期的运行成本，不属于验收基准。

### 上游 commit 复核方式

```powershell
git ls-remote https://github.com/snape/RVO2-3D refs/heads/main
# f940bfba8bac9532ecae948f11639085e8cbf277	refs/heads/main
```
