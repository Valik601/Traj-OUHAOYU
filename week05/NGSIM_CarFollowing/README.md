# NGSIM I-80 跟驰模型复现与对比

本项目从 NGSIM I-80 的 10 Hz SI 轨迹中重建直接前后车关系，自动筛选连续的跟驰事件，并在同一条真实前车轨迹驱动下比较 OVM、FVDM 与 IDM。原始 CSV 始终只读；所有清洗、筛选、仿真和评价结果写入 `outputs/`。

## 环境与安装

已验证环境为 Python 3.11.4。`requirements.txt` 固定的是本机端到端跑通全部结果所用的版本：

```powershell
cd NGSIM_CarFollowing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

默认数据路径在 `config.yaml` 中设置为：

```text
../datasets/NGSIM-I80/NGSIM_I80_20050413_1600-1615_SI.csv
../datasets/NGSIM-I80/source_and_checks.json
```

SI 文件已经使用米和秒，不会再次换算单位。原始英制 CSV 不是运行必需项；本机提供的课堂数据包（`datasets/第4周_NGSIM-I80_课堂数据.zip`）中也只有 SI 文件、使用说明和 `source_and_checks.json`，**不包含** `*_raw.csv`，因此项目不引用任何 raw 输入。

## 一键运行

```powershell
python run_pipeline.py
```

默认运行 STEP 0–6。从某一步重新运行：

```powershell
python run_pipeline.py --start-step 3 --end-step 6
```

仅查看将执行的命令：

```powershell
python run_pipeline.py --dry-run
```

每一步也可独立执行：

```powershell
python src/00_inspect_data.py --config config.yaml
python src/01_find_following_pairs.py --config config.yaml
python src/02_select_following_case.py --config config.yaml
python src/03_prepare_trajectories.py --config config.yaml
python src/04_car_following_models.py --config config.yaml
python src/05_simulate.py --config config.yaml
python src/06_evaluate_and_plot.py --config config.yaml
python src/07_calibrate.py --config config.yaml          # 可选，样本内标定
```

运行测试：

```powershell
python -m pytest -q
```

测试包含单元测试和两类整体校验：`tests/test_artifact_consistency.py` 用独立重写的公式与积分器复算 `outputs/` 中的 CSV/JSON，并复核报告中的 RMSE、最小净间距与终点误差；`tests/test_atomic_write.py` 覆盖所有输出文件的原子落地路径。若上游尚未运行，这些用例会自动跳过而不是伪造通过。

## 数据语义与处理边界

- 原始观测单位是一个车辆在一个 0.1 s 帧中的记录；同一辆车的连续帧属于重复测量，不能当作独立样本。
- `local_y_m` 是车辆前端中心的纵向位置。对同一帧同一车道按它排序，直接相邻车辆组成候选关系。
- 净间距严格定义为 `leader_y - follower_y - leader_length`，不是前端到前端距离。
- 必须先对全部车型建立相邻关系，再筛选前后车均为 `vehicle_class=2` 的事件。
- 原始值不被覆盖。位置平滑方法、窗口和阶数由配置记录；清洗速度与加速度从平滑位置求导，并与原始运动学量并存。
- 自动排名是探索性筛选，不代表模型参数或案例具有总体代表性；示例参数也不声称是标定后的最优参数。

## 更换车辆对或时间窗

默认由 `02_select_following_case.py` 按连续性、速度扰动、减速度、后车响应和间距质量自动排名。排名不只是时长，评分权重见 `config.yaml` 的 `selection.score_weights`。先查看：

```text
outputs/03_cases/candidate_cases.csv      # 候选表（含各项分项得分与是否合格）
outputs/03_cases/candidate_preview.png    # 前几名 + 所选案例的纵向位置/速度/净间距三线图
```

选定案例的纵向位置、速度、净间距曲线也应人工复核（`selected_case_raw.csv` 与 `candidate_preview.png` 已给出同样的三个量），确认不存在换道、缺帧或异常跳变后再进入建模阶段。

若要指定另一个已审查案例，把 `config.yaml` 中的 `selection.manual_case` 从 `null` 改为：

```yaml
manual_case:
  leader_id: 123
  follower_id: 456
  start_frame: 1000
  end_frame: 1300
```

指定案例仍会接受车道、车型、连续性、时长和净间距检查；验证失败会明确报错，而不会静默放宽筛选条件。修改后从 STEP 2 重新运行。

## 修改模型与数值设置

`config.yaml` 的 `models` 部分保存 OVM、FVDM 和 IDM 参数；`simulation.time_step_s` 设置积分步长。当前数值是可复现实验用的文献型示例参数，不是对测试片段标定出的最优值。三个模型共享真实前车轨迹、真实后车初始位置和初始速度，并使用同一积分器。

初始化状态的含义：`outputs/03_cases/selected_case_raw.csv` 与 `local_y_m`/`speed_m_s`/`acceleration_m_s2` 保存未改动的原始观测；`outputs/04_prepared/initial_conditions.json` 中的 `initial_*` 字段是**仿真实际使用的初始化状态**（与 `trajectory_clean.csv` 首行、初始净间距严格一致），原始观测同时保留在 `raw_*`/`source_frame_*` 字段中，便于追溯平滑前后的差别。

## 可选：参数标定（STEP 7）

STEP 7 **不参与默认运行**，因为它的标定与评价使用同一个已选片段，属于样本内演示：

```powershell
python run_pipeline.py --start-step 7 --end-step 7
# 或
python src/07_calibrate.py --config config.yaml --model IDM
```

优化目标（位置/速度归一化误差加权，加失效、碰撞、缺失与非法候选惩罚）、参数边界、优化器选项和随机种子都在 `config.yaml` 的 `calibration` 部分显式配置；未写明的键回退到脚本中的默认值。输出：

```text
outputs/07_calibration/calibrated_parameters.json   # 目标、边界、初始参数、最终参数
outputs/07_calibration/calibration_metrics.csv      # 标定前后指标、函数调用次数、耗时
```

两个产物都带 `in_sample=true / generalization_result=false` 与警告文本。`optimizer_success` 记录的是差分进化自身的收敛判据；本次记录的运行在 `maxiter: 40` 处达到迭代上限（`Maximum number of iterations has been exceeded.`），目标函数虽显著下降，但**这些参数只能作为演示结果，不能称为最优参数，也不能作为泛化性能**。若要做严格比较，请用独立片段标定（`calibration.output_dir` 指向单独目录）、在未参与标定的多车辆对集合上报告误差与失效率。

## 输出目录

```text
outputs/
├── 01_data_quality/      # 数据摘要、异常记录与分布图
├── 02_following_pairs/   # 每帧直接相邻关系
├── 03_cases/             # 候选排名、所选事件与预览
├── 04_prepared/          # 原始/清洗轨迹与初始条件
├── 05_simulation/        # 三模型逐时刻仿真与状态
├── 06_evaluation/        # 指标、六类图、综合图和实验报告
└── 07_calibration/       # 可选样本内标定（默认不生成）
```

`outputs/run_pipeline.log` 记录完整执行顺序与各步骤耗时。CSV/JSON 是图表的可访问数据替代；PNG 用于屏幕和汇报，PDF 保留矢量线条与文字。

## 模型文献

- Bando et al. (1995), Optimal Velocity Model, <https://doi.org/10.1103/PhysRevE.51.1035>
- Jiang, Wu & Zhu (2001), Full Velocity Difference Model, <https://doi.org/10.1103/PhysRevE.64.017101>
- Treiber, Hennecke & Helbing (2000), Intelligent Driver Model, <https://doi.org/10.1103/PhysRevE.62.1805>

本项目的数据审计与科学图表工作流还参考了：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. <https://doi.org/10.48550/arXiv.2609.00065>。

## 局限

单一跟驰事件不能支持模型泛化结论。NGSIM 的位置、速度和加速度含测量与差分噪声；温和平滑会改变高频成分。响应时滞依赖预先声明的加速度阈值和连续帧规则。若需严格比较，应在独立事件上标定参数，再在未参与标定的多车辆事件集合上报告误差与失效率。
