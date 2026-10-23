"""week06 窄路相撞实验：统一人-车 Agent 模型 + 规则化交互。"""

from .common import analytic_predictions, draw_initial_conditions, load_config
from .simulate import RunResult, RunSpec, simulate

__all__ = [
    "analytic_predictions",
    "draw_initial_conditions",
    "load_config",
    "RunResult",
    "RunSpec",
    "simulate",
]
