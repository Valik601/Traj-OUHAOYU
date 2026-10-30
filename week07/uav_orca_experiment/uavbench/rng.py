"""随机流分离（handoff §3.2、§5.5、§5.3）。

所有随机性必须来自显式命名流，禁止依赖函数调用次数隐式共享状态。
流的编号写入 CONTRACTS.md 并保持不变；改动会破坏正式结果的场景/候选/放行可复现性。
"""

from __future__ import annotations

import numpy as np

#: 场景扰动（起点/终点几何）
STREAM_SCENE: int = 0
#: 分时放行随机流 0（主实验）
STREAM_SCHEDULE_0: int = 1
#: 分时放行随机流 1（放行顺序敏感性）
STREAM_SCHEDULE_1: int = 2
#: 候选方向集整体旋转（每个场景一次，M1/M2 共享）
STREAM_CANDIDATE: int = 3
#: 仅扩展使用的 tie-break 流（主实验不得使用）
STREAM_TIE_BREAK: int = 4


def make_rng(seed: int, stream: int) -> np.random.Generator:
    """由场景种子和流编号构造独立随机数生成器。

    负种子（确定性基础场景）映射到固定 32 位熵，保证跨平台确定性。
    """
    entropy = int(seed) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence(entropy=entropy, spawn_key=(int(stream),)))
