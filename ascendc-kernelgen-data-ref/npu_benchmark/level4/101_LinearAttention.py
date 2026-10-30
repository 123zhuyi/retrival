import json
import os

import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    Linear Attention core.

    算子边界：仅 kernel 化的线性注意力核心机制；
    输入/输出投影在算子定义外，按恒等处理（q = k = v = x 分头）。

    语义与原始实现逐项对齐：
      - 全程在输入 dtype 内计算（原始无 fp32 上转）；
      - 归一化 Z = 1 / (denominator + 1e-6)（原始形式，无 sign 修正）；
      - 无 clamp（原始无此操作）。

    Inputs:
        x: [batch, seq_len, d_model]
        n_heads: number of attention heads
        feature_map: "elu", "relu", or "identity"

    Output:
        [batch, seq_len, d_model]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, n_heads, feature_map):
        batch, sequence, d_model = x.shape
        head_dim = d_model // n_heads
        q = x.view(batch, sequence, n_heads, head_dim).transpose(1, 2)
        k = q
        v = q
        if feature_map == "elu":
            q, k = F.elu(q) + 1.0, F.elu(k) + 1.0
        elif feature_map == "relu":
            q, k = F.relu(q), F.relu(k)
        elif feature_map != "identity":
            raise ValueError("feature_map must be elu, relu, or identity")

        kv = torch.matmul(k.transpose(-2, -1), v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).unsqueeze(-1) + 1e-6)
        output = torch.matmul(q, kv) * z
        output = output.transpose(1, 2).contiguous().view(batch, sequence, d_model)
        return output


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, feature_map):
    generator = torch.Generator()
    generator.manual_seed(seed)
    shape = tuple(spec["shape"])
    dtype = _DTYPE_MAP[spec["dtype"]]
    if feature_map == "identity":
        # identity 路径 q=k=v=x 无 feature map 保护：混合符号输入下分子 (q@kv)
        # 与分母 (q@sum(k)) 都是过零随机游走，输出为二者之比，量级重尾爆炸
        # （实测 max|out| 可达 1e5），低精度中间量化噪声必然超 verify 阈值，
        # 任何正确实现都无法复现。正区间输入使注意力权重非负，输出退化为 v 的
        # 凸组合，|out| <= max|x| 数学上有界。
        # fp16 附加约束：中间量 q@kv ~ N*D_head*s^3 须远低于 65504，取 s=1。
        scale = 1.0 if dtype == torch.float16 else 5.0
        tensor = torch.rand(shape, generator=generator, dtype=torch.float32) * scale
    else:
        # elu/relu 的 feature map 输出非负、分母无符号消去，分布本身安全；
        # 零均值小方差是为了避免原随机 mean in [-5,5] 抽签把中间量 kv /
        # denominator 撑大，导致低精度量化噪声超 verify 阈值。
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.5
    return tensor.to(dtype=dtype)


def _load_cases():
    json_path = os.path.splitext(__file__)[0] + ".json"
    with open(json_path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def _case_specs(case):
    return {item["name"]: item for item in case["inputs"]}


def get_input_groups():
    input_groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = _case_specs(case)
        # 可选 per-case seed 覆盖（历史遗留：case 43/44 的 seed=1000 是旧分布下
        # 修复数值缺陷的手工逃生口；分布约束改造后不再必需，保留仅为输入可复现）
        seed = specs["x"].get("seed", 42 + case_index)
        input_groups.append([
            _random_tensor(specs["x"], seed, specs["feature_map"]["value"]),
            specs["n_heads"]["value"],
            specs["feature_map"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []