import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs GDN (Gated Delta Net) gating computation.
    对标 vLLM fused_gdn_gating_kernel
    (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1680, v0.26.0)。

    数学定义 (对每个 token 位置 s、每个头 h, 内部全部 fp32 计算):
        x          = a[s, h].float() + dt_bias[h].float()
        softplus_x = (1/beta) * log(1 + exp(beta * x))   若 beta*x <= threshold
                   = x                                  否则 (防 exp 溢出)
        g[s, h]    = -exp(A_log[h].float()) * softplus_x        # 输出恒 <= 0
        beta_output[s, h] = sigmoid(b[s, h].float())            # 输出 (0, 1)

    布局约定:
        A_log     [H]        fp32 (对数域衰减系数, exp(A_log) > 0)
        a, b      [batch, H] 任意浮点 dtype (kernel 内部 .to(fp32))
        dt_bias   [H]        fp32
        g         [1, batch, H] fp32 输出
        beta_output [1, batch, H] 与 b 同 dtype 输出
        kernel 的 seq_len 恒为 1 (展平的 batch 维), 与 wrapper fused_gdn_gating 一致。

    与 kernel 的差异: 无 (纯逐元素数学, 无分块/量化路径)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, A_log, a, b, dt_bias, beta: float = 1.0,
                threshold: float = 20.0):
        x = a.float() + dt_bias.float()
        # 严格按 kernel 写法 (1/beta)*log(1+exp(beta*x)), 不用 log1p(exp) 等价式
        softplus_x = torch.where(
            beta * x <= threshold,
            (1.0 / beta) * torch.log(1 + torch.exp(beta * x)),
            x,
        )
        g = -torch.exp(A_log.float()) * softplus_x
        beta_output = torch.sigmoid(b.float()).to(b.dtype)
        # kernel 输出形状为 [1, batch, H]
        return g.unsqueeze(0), beta_output.unsqueeze(0)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "4_FusedGdnGating.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # 标准输入分布: 50% 均匀 [-5, 5] + 50% 正态 (mu ∈ [-5, 5], sigma ∈ [0.1, 2])
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        beta, threshold = 1.0, 20.0
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "beta":
                beta = inp["value"]
            elif name == "threshold":
                threshold = inp["value"]

        batch, H = shapes["a"]
        assert shapes["b"] == [batch, H]
        assert shapes["A_log"] == [H] and shapes["dt_bias"] == [H]

        # 物理约束: A_log 是 log 域衰减 (真实模型经 softplus/初始化后大致在 [-10, 5]),
        # dt_bias 为可学习偏置 (实际模型 |dt_bias| 一般不超过 5); 二者按标准分布生成后
        # 截断到物理合理区间, 保证 g 数值稳定 (exp(A_log) 不溢出)。
        A_log = random_tensor(shapes["A_log"], dtypes["A_log"]).clamp(-10.0, 5.0)
        a = random_tensor(shapes["a"], dtypes["a"])
        b = random_tensor(shapes["b"], dtypes["b"])
        dt_bias = random_tensor(shapes["dt_bias"], dtypes["dt_bias"]).clamp(-5.0, 5.0)

        input_groups.append([A_log, a, b, dt_bias, beta, threshold])
    return input_groups


def get_init_inputs():
    return []