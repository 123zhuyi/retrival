import json
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# QuantMatmulReduceSum —— 融合量化分组 matmul + 跨 batch 求和
#
# 借鉴 ops-ref `quant_matmul_reduce_sum` AscendC kernel 的融合思路：
#   out = sum_{i=0}^{batch} (x1_i @ x2_i) * x1Scale * x2Scale
# 把「逐组 bmm -> Python 循环累加」改写为：
#   bmm（单次 cube 矩阵乘，int8 值域在 fp32/int32 累加下精确）
#   -> 逐行乘 x1_scale（per-batch per-row）
#   -> .sum(dim=0) 一次向量化跨 batch 归约（替代 B 次 kernel 发射的顺序累加）
#   -> 乘 x2_scale
#
# scale 应用顺序与 golden 一致（x1_scale 先乘、x2_scale 后乘），
# 归约改为单次向量化 sum（与 kernel 的融合 reduce 对齐），fp32 精度与原逐步累加一致。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Quantized grouped matmul reduce sum（融合版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                x1_scale=None, x2_scale=None):
        """
        Args:
            x1 (Tensor): [batch, m, k]，int8
            x2 (Tensor): [batch, k, n]，int8
            x1_scale (Tensor): [batch, m]，float32
            x2_scale (Tensor): [n]，bfloat16/float32

        Returns:
            Tensor: [m, n]，float32
        """
        # bmm 在 fp32 下对 int8 值域精确（乘积无舍入，累加 <2^24 项可精确表示）
        mid = torch.bmm(x1.float(), x2.float())         # [B, M, N]
        mid = mid * x1_scale.unsqueeze(-1)              # per-batch per-row scale
        acc = mid.sum(dim=0)                            # 一次向量化跨 batch 归约
        return acc * x2_scale.float()


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "35_QuantMatmulReduceSum.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        x1_scale_info = inputs[2]
        x2_scale_info = inputs[3]

        x1 = torch.randint(-5, 5, x1_info["shape"], dtype=torch.int8)
        x2 = torch.randint(-5, 5, x2_info["shape"], dtype=torch.int8)
        x1_scale = random_tensor(x1_scale_info["shape"], torch.float32)

        x2_scale_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        x2_scale_dtype = x2_scale_dtype_map.get(x2_scale_info.get("dtype", "bfloat16"), torch.bfloat16)
        x2_scale = random_tensor(x2_scale_info["shape"], x2_scale_dtype)

        input_groups.append([x1, x2, x1_scale, x2_scale])
    return input_groups


def get_init_inputs():
    return []
