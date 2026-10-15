# NGSIM I-80 跟驰模型复现与对比实验报告

生成时间（UTC）：2026-10-15T03:58:21+00:00

## 1. 数据集与评价范围

本实验使用 NGSIM I-80 Eastbound（Emeryville, California）2005-04-13 16:00–16:15 PDT 数据，标称采样率 10 Hz。输入为 SI 版本，位置、速度、加速度分别采用 m、m/s、m/s²，未执行二次单位换算。`local_y_m`/标准化 `x` 表示车辆前端中心纵向位置；净间距定义为前车前端位置减后车前端位置再减前车长度。

本报告只评价上游已选定并清洗的单个跟驰片段，不重新筛选车辆对，也不修改原始或清洗轨迹。

- Leader ID：**1251**
- Follower ID：**1259**
- Lane ID：**5**
- 数据时间段：305.5–345.5 s；标准化时长：40.000 s；样本数：401
- 筛选说明：User-configured manual case; accepted only after validating direct adjacency, single-lane continuity, passenger classes, 0.1 s sampling, configured gap and duration bounds, leader speed variation, and leader braking.

## 2. 初始条件

| State | Leader | Follower |
| --- | --- | --- |
| Vehicle ID | 1251 | 1259 |
| Frame ID | 3706 | 3706 |
| Initial time (s) | 305.5 | 305.5 |
| Initial lateral position (m) | 16.39583 | 16.33823 |
| Initial longitudinal front position (m) | 85.85337 | 68.95649 |
| Initial speed (m/s) | 14.31495 | 14.21674 |
| Initial acceleration (m/s²) | -0.4512017 | -2.787438 |
| Observed source speed (m/s) | 14.34087 | 14.32258 |
| Observed source acceleration (m/s²) | -0.890018 | -3.413767 |
| Vehicle length (m) | 4.663449 | 5.577851 |
| Lane ID | 5 | 5 |

- 初始净间距：12.2334 m
- 初始相对速度 \(v_L-v_F\)：0.0982091 m/s
- 初始相对加速度 \(a_L-a_F\)：2.33624 m/s²

表中 `Initial ...` 行为三模型实际使用的初始化状态，与 `outputs/04_prepared/trajectory_clean.csv` 首行一致，并与上方初始净间距、相对速度满足 `s(0)=x_L(0)-x_F(0)-l_L`；`Observed source ...` 行为未改动的原始观测，仅用于追溯平滑前后的差异（原始位置见初始化 JSON）。

真实后车初始加速度仅作为观测值；各模型初始加速度由自身方程计算。

## 3. 数据处理与可追溯性

评价真值来自 `outputs/04_prepared/trajectory_clean.csv`，初始化记录来自 `outputs/04_prepared/initial_conditions.json`，仿真状态来自 `outputs/05_simulation/simulation_status.json`。配置文件为 `config.yaml`。绘图直接使用这些 CSV 中的真实计算数据，没有插值补齐失败后的模型轨迹，也没有跨缺失区间连线。

- `smoothing_enabled`：`True`
- `method`：`Savitzky-Golay position smoothing`
- `savgol_window_length`：`21`
- `savgol_polyorder`：`2`
- `use_smoothed_kinematics`：`True`
- `kinematics_method`：`first and second Savitzky-Golay derivatives of smoothed longitudinal position`
- `sampling_interval_s`：`0.1`
- `raw_values_preserved_in`：`trajectory_raw.csv`
- `no_clipping_or_imputation`：`True`
- `smoothing_context`：`{'source': 'individual raw SI trajectories', 'padding_frames_requested_each_side': 10, 'context_start_frame': 3696, 'context_end_frame': 4106, 'selected_start_frame': 3706, 'selected_end_frame': 4106}`

## 4. 模型与参数

- **OVM**：\(a_F=\kappa[V(s)-v_F]\)，其中 \(V(s)\) 为归一化双曲正切最优速度函数。
- **FVDM**：\(a_F=\kappa[V(s)-v_F]+\lambda(v_L-v_F)\)，与 OVM 共用同一形式的 \(V(s)\)。
- **IDM**：\(a_F=a[1-(v_F/v_0)^\delta-(s^*/s)^2]\)，\(s^*=s_0+\max(0,v_FT+v_F(v_F-v_L)/(2\sqrt{ab}))\)。

| Model | 本次运行参数 |
| --- | --- |
| OVM | `kappa=0.5`, `v_max=33.33`, `s_c=25.0`, `ell=8.0` |
| FVDM | `kappa=0.5`, `lambda=0.5`, `v_max=33.33`, `s_c=25.0`, `ell=8.0` |
| IDM | `v0=33.33`, `T=1.5`, `s0=2.0`, `a=1.2`, `b=2.0`, `delta=4.0`, `minimum_gap_epsilon_m=0.001`, `maximum_braking_m_s2=50.0` |

这些参数是本次运行记录的配置/状态参数；若未经独立训练片段标定，则不得称为真实最优参数。

## 5. 数值积分设置

- 时间步长：0.1 s
- 积分方法：{'speed': 'forward Euler with max(0, v + a*dt)', 'position': 'trapezoidal using old and updated speed'}
- 外部输入：三种模型均使用相同的真实前车位置与速度轨迹。
- 后车初始化：三种模型均使用相同的真实后车初始位置与速度；仿真开始后不使用真实后车状态纠偏。
- 碰撞边界：模拟净间距 ≤ 0 m。

## 6. 制动响应事件规则

Leader 制动事件定义为加速度 ≤ -0.5 m/s² 且连续至少 3 帧，事件时刻取该连续序列的首帧。Follower/模型响应使用相同阈值和连续帧规则，并只在 Leader 事件后 8 s 内搜索；响应时滞为响应首帧减 Leader 事件首帧。无法检测 Leader 事件或窗口内响应时输出 `NaN`，不填入任意数值。

- Leader 制动起点：8.7 s
- 真实 Follower 响应起点：10.5 s
- 真实 Follower 响应时滞：1.8 s
- 图中重新加速事件规则：Leader 加速度 ≥ 0.5 m/s² 且连续至少 3 帧，并在制动事件之后搜索；检测时刻：16.4 s。

## 7. 评价结果

RMSE 均在模型与真值从片段首帧开始的**连续共同有效前缀**上计算。最终位置误差是共同有效终点处的有符号误差 \(x_{sim}-x_{true}\)。最小净间距使用模型文件内全部有限记录（包括已记录的失败/碰撞点）。

| Model | 位置 RMSE (m) | 速度 RMSE (m/s) | 加速度 RMSE (m/s²) | 最小净间距 (m) | 响应时滞 (s) | 最终位置误差 (m) | 共同有效时长 (s) | 完成率 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OVM | 5.169 | 1.726 | 1.514 | 12.23 | 2.8 | -7.13 | 40 | 100.00% | completed |
| FVDM | 4.739 | 1.108 | 0.9502 | 12.23 | 3.5 | -6.651 | 40 | 100.00% | completed |
| IDM | 1.508 | 0.5175 | 0.56 | 7.811 | 4.2 | -0.9849 | 40 | 100.00% | completed |

### 描述性比较

- 在**完整运行模型**中，位置 RMSE最低的是 IDM （1.508）；这是单一片段上的描述性结果，不代表泛化性能。
- 在**完整运行模型**中，速度 RMSE最低的是 IDM （0.5175）；这是单一片段上的描述性结果，不代表泛化性能。
- 在**完整运行模型**中，加速度 RMSE最低的是 IDM （0.56）；这是单一片段上的描述性结果，不代表泛化性能。

### 碰撞与失效分析

- **OVM**：完整运行；未检测到碰撞。
- **FVDM**：完整运行；未检测到碰撞。
- **IDM**：完整运行；未检测到碰撞。

### 误差来源分解

下表把每个模型的位置平方误差拆成“0–5 s 起步瞬态”和“其后对前车扰动的响应”两部分，并按分段 RMSE、是否曾压缩初始净间距、响应时滞误差比较。数值全部由本轮仿真与真值直接计算。

| Model | 模型初始加速度 (m/s²) | 真实后车初始加速度 (m/s²) | 0–5 s 位置 RMSE (m) | 5 s 后位置 RMSE (m) | 0–5 s 误差占比 | 最小净间距未低于初始值 | 响应时滞误差 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OVM | -6.481 | -2.787 | 8.328 | 4.528 | 33.0% | 是 | 1 |
| FVDM | -6.432 | -2.787 | 6.005 | 4.525 | 20.4% | 是 | 1.7 |
| IDM | -3.035 | -2.787 | 0.9298 | 1.574 | 4.8% | 否 | 2.4 |

- **OVM** 以起步瞬态为主：0–5 s 的 RMSE 为 8.328 m，明显高于其后的 4.528 m；该窗口只占时长的 12.7%，却贡献了 33.0% 的位置平方误差。
- **OVM** 的最小净间距等于初始净间距 12.23 m（在六位小数舍入容差内），说明整段仿真从未把间距压缩到初始值以下，没有复现真实后车先接近再拉开的间距过程。
- **OVM** 的制动响应比真实后车晚 1.0 s（阈值与连续帧规则见上一节，改变阈值会改变该数值）。
- **FVDM** 以起步瞬态为主：0–5 s 的 RMSE 为 6.005 m，明显高于其后的 4.525 m；该窗口只占时长的 12.7%，却贡献了 20.4% 的位置平方误差。
- **FVDM** 的最小净间距等于初始净间距 12.23 m（在六位小数舍入容差内），说明整段仿真从未把间距压缩到初始值以下，没有复现真实后车先接近再拉开的间距过程。
- **FVDM** 的制动响应比真实后车晚 1.7 s（阈值与连续帧规则见上一节，改变阈值会改变该数值）。
- **IDM** 的误差更多来自后段对前车扰动的响应：0–5 s RMSE 为 0.9298 m，其后为 1.574 m；前段占时长 12.7%，贡献 4.8% 的位置平方误差。
- **IDM** 的制动响应比真实后车晚 2.4 s（阈值与连续帧规则见上一节，改变阈值会改变该数值）。
- 三模型在 t=0 的加速度由各自方程给出（OVM -6.48 m/s², FVDM -6.43 m/s², IDM -3.04 m/s²），而真实后车观测初始加速度为 -2.79 m/s²。该差异来自各自的平衡态关系，不是把真值当作初始条件。

## 8. 图表说明

模型用颜色和线型双重编码；PNG 按配置导出高分辨率栅格图，PDF 保留线条和文本的矢量表示。误差图中淡化的柱表示未完成/失败模型，柱上同时标出覆盖率与状态。所有坐标轴明确给出物理单位，零净间距/配置碰撞边界在间距图中单独标记。

- `outputs/06_evaluation/acceleration_response.pdf`
- `outputs/06_evaluation/acceleration_response.png`
- `outputs/06_evaluation/error_timeseries.csv`
- `outputs/06_evaluation/evaluation_metadata.json`
- `outputs/06_evaluation/evaluation_metrics.csv`
- `outputs/06_evaluation/experiment_report.md`
- `outputs/06_evaluation/model_comparison.pdf`
- `outputs/06_evaluation/model_comparison.png`
- `outputs/06_evaluation/position_comparison.pdf`
- `outputs/06_evaluation/position_comparison.png`
- `outputs/06_evaluation/position_error.pdf`
- `outputs/06_evaluation/position_error.png`
- `outputs/06_evaluation/spacing_comparison.pdf`
- `outputs/06_evaluation/spacing_comparison.png`
- `outputs/06_evaluation/speed_response.pdf`
- `outputs/06_evaluation/speed_response.png`
- `outputs/06_evaluation/trajectory_selection.pdf`
- `outputs/06_evaluation/trajectory_selection.png`

## 9. 研究局限

1. 结果来自单一车辆对和单一时间窗，不能直接推广到其他驾驶人、车道或交通状态。
2. 未经独立片段标定的文献参考/示例参数只能用于复现流程；不能据此宣称某一模型具有普遍优势。
3. NGSIM 轨迹存在测量噪声，位置平滑及数值微分会影响速度、加速度和制动事件时刻；响应时滞还会随阈值、连续帧数和搜索窗变化。
4. 模型提前失效时，RMSE 仅覆盖失效前的共同有效区间；短区间误差可能看似较小，因此必须结合完成率、最小间距和失效原因解释。
5. 碰撞判据使用一维净间距边界，不包含横向重叠、车辆姿态、反应感知误差或车辆动力学约束。
6. 当前比较不含独立训练/测试划分的参数标定，也未量化参数不确定性或跨事件泛化误差。

## 10. 参考资料

- Bando, M. et al. (1995). Dynamical model of traffic congestion and numerical simulation. *Physical Review E*, 51, 1035. https://doi.org/10.1103/PhysRevE.51.1035
- Jiang, R., Wu, Q. S., & Zhu, Z. J. (2001). Full velocity difference model for a car-following theory. *Physical Review E*, 64, 017101. https://doi.org/10.1103/PhysRevE.64.017101
- Treiber, M., Hennecke, A., & Helbing, D. (2000). Congested traffic states in empirical observations and microscopic simulations. *Physical Review E*, 62, 1805. https://doi.org/10.1103/PhysRevE.62.1805
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
