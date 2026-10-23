# 窄路相遇实验（统一人-车 Agent 模型 + 规则化交互）

同一条无路口、无信号灯、无人行道的窄路上：少量行人沿 +x 步行，一辆汽车从右侧沿 −x 迎面驶来；行人可侧移，汽车只能减速/停车。行人与车辆使用**同一个 Agent 框架**（参数不同），并比较两种运动模型（社会力 SFM、Wiedemann 阈值模型）与两套交互规则（R0 基准、R1 修改），寻找"空间不足"与"规则失效"两类失败。

完整实验报告：[report/实验报告.md](report/实验报告.md)

## 环境与安装

本机验证环境：Python 3.12.14、numpy 2.3.5、pandas 3.0.1、matplotlib 3.11.2、PyYAML 6.0.3、pytest 9.1.1。

```powershell
cd narrow_road
python -m pip install -r requirements.txt     # 固定为本机验证过的版本
```

## 一键运行

```powershell
python run_experiments.py            # 全流程：32 次仿真 + 指标 + 参数表 + 8 张图（约 6 s）
python run_experiments.py --no-figures   # 只跑仿真与指标
python run_experiments.py --out outputs_dev   # 换输出目录
python -m pytest -q                  # 38 个测试（约 5 s）
```

运行会在 `outputs/` 下生成全部产物（见下）。代码不写死任何参数：几何、主体参数、规则阈值、模型参数、指标阈值全部在 `config.yaml`。

## 目录结构

```text
narrow_road/
├── config.yaml              # 全部参数（几何/主体/规则/模型/指标）
├── run_experiments.py       # 一键入口
├── requirements.txt
├── narrowroad/
│   ├── common.py            # 配置、初始条件、几何工具、解析判据
│   ├── agent.py             # 统一 Agent 状态 + 感知接口
│   ├── decision.py          # 决策层：行人/车辆交互规则
│   ├── controllers.py       # 控制层：SFM / Wiedemann
│   ├── simulate.py          # 时间步进、约束、僵持检测、指标
│   ├── experiments.py       # 情景矩阵、批量运行、产物落盘
│   ├── tables.py            # 参数表生成
│   └── plotting.py          # 全部图（PNG + PDF）
├── tests/                   # 38 个单元/一致性测试
├── report/实验报告.md
└── outputs/                 # 运行产物（见下）
```

## 关键设计

### 统一的 Agent 表征

行人（全向移动体，α=1）与车辆（纵向移动体，α=0，y 锁死）共享状态、感知、决策与控制接口，只参数不同：

| 统一量 | 行人 | 车辆 |
|---|---|---|
| 形状 | 外接圆 r = 0.25 m | 矩形 4.5 × 1.8 m（SFM 用 3 圆近似） |
| 期望方向 / 速度 | +x / 0.9–1.5 m/s | −x / 9.0 m/s |
| τ / a_max / b_comfort / b_max | 0.5 / 1.5 / 2.0 / 3.0 | 0.4 / 2.0 / 2.5 / 6.0 |
| 横向机动 | 可侧移（≤1.0 m/s） | 0（y ≡ 0） |

### 决策层（显式交互规则）

- **行人**：车辆未驶过、横向余量 c < 0.30 m 且 5 m < g ≤ 25 m → 边侧移边前进；g ≤ 5 m → 等待（R0 僵住 / R1 继续移向路缘）；其余情况正常走。
- **车辆**：对每个前方行人，取"物理安全停车曲线"和"规则允许速度"的最小值。R0 固定要求 c ≥ 0.30 m 才放行；R1 允许速度随余量线性增加，余量 0.10 m 时允许约 1 m/s 蹭行。规则每步重算，条件解除后自动恢复 v0。

### 两种运动模型

- **SFM**：驱动力 + 指数型各向异性斥力（行人←车辆、行人↔行人、车辆←行人只取纵向制动分量）；车辆用三圆近似。
- **Wiedemann（简化）**：自由行驶 / 接近 / 制动 / 跟驰 / 静止保持五个 regime，阈值 s_des = SDX + T·v、s_brake、s_react 与速度差感知阈值 SDV；推广到迎面行人。

两者的定性结论一致（16/16 格子），定量延误差异 ≤ 约 15%。

## 输出

```text
outputs/
├── metrics.csv                  # 32 次仿真的汇总指标
├── ped_metrics.csv              # 每名行人指标
├── trajectories/*.csv           # 逐时刻轨迹（主实验 + 消融）
├── initial_conditions.json      # 冻结的初始条件
├── predictions.json             # 运行前解析可通行判据
├── prediction_consistency.json  # 预测 vs 仿真（16/16 一致）
├── summary.json                 # 报告用关键数字
├── tables/*.md                  # 几何/规则/模型/结果四张表
└── figures/*.png, *.pdf         # 8 张图（含一页总结 summary.png）
```

## 测试

`python -m pytest -q` 覆盖：

- 几何与解析判据（余量公式、两个临界宽度 3.40 / 2.90 m、允许速度映射）；
- 决策层模式（WALK / SIDESTEP / WAIT；R0 冻住侧向 vs R1 移向路缘）与车辆规则（R0 停车 vs R1 低速放行）；
- 控制层（自由行驶/制动/保持、SFM 斥力方向、Wiedemann regime）；
- 安全不变量（任意时刻无身体重叠、行人不越界/不后退、车辆不侧移/不倒车、减速度不超上限）；
- 结果判定（12 个宽度×规则×模型组合的 PASS/DEADLOCK）、确定性、完整矩阵的产物一致性（指标 ↔ 轨迹 ↔ summary ↔ predictions）。
- 可复现性（本机验证）：同一输出目录连续两次运行，CSV / JSON / Markdown / PNG 逐字节一致（PDF 含时间戳）。

## 文献

- Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. *Physical Review E*, 51(5), 4282–4286. <https://doi.org/10.1103/PhysRevE.51.4282>
- Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic. *Nature*, 407, 487–490. <https://doi.org/10.1038/35035023>
- Wiedemann, R. (1974). *Beschreibung des Verkehrsablaufs auf Straßen mittels Simulation*. Universität Karlsruhe.
- Treiber, M., & Kesting, A. (2013). *Traffic Flow Dynamics: Data, Models and Simulation*. Springer. <https://doi.org/10.1007/978-3-642-32460-4>

## 局限

单一初始条件（固定种子）、3 名行人、1 辆车、单车道路段；行人决策阈值与模型参数为设定值，未用真实数据标定；车辆固定在中线且不能侧移；Wiedemann 原为同向跟驰，已做显式简化推广；僵持判定采用 8 s 无净位移、40 m 车前行人窗口的口径。详见报告第 8 节。

