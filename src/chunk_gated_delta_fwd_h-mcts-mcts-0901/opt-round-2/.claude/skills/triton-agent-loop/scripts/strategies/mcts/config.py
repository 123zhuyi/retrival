# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MCTS 默认超参数与奖励函数.

这些常量作为 MCTSTree 的缺省值；实例化时可通过 params 字典覆盖。
"""

import math

# MCTS 超参数
C_UCT = math.sqrt(2)          # UCT 探索常数
C_PW = 2.0                    # Progressive widening 系数
ALPHA = 0.5                   # Progressive widening 指数
FAILURE_THRESHOLD = 3         # 节点总失败次数达到该值时剪枝

# 奖励常量
REWARD_OUTPUT_ERROR = -2      # 输出/精度校验失败
REWARD_COMPILE_ERROR = -3     # 编译失败或无有效结果


def reward_success(speedup: float) -> float:
    """成功时奖励为加速比的自然对数."""
    return math.log(speedup)
