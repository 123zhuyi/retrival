import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs fused MoE expert computation (unquantized).
    对标 vLLM fused_moe_kernel 的两段式调用
    (vllm/model_executor/layers/fused_moe/fused_moe.py:295, v0.26.0;
    编排逻辑见 fused_experts_impl: GEMM1 -> silu_and_mul -> GEMM2(mul_routed_weight) -> sum)。

    数学定义 (对每个 token m 与其第 k 个路由专家 e = topk_ids[m, k], fp32 内部累加):
        y      = hidden[m] @ w1[e].T            # [2I]
        gate   = y[:I], up = y[I:]
        inter  = silu(gate) * up                # silu_and_mul, 即 SwiGLU
        out[m] += topk_weights[m, k] * (inter @ w2[e].T)   # 第二段乘路由权重
        out[m] = sum_k(...)                     # moe_sum: 沿 topk 维求和

    布局约定:
        hidden_states [M, H]        fp16/bf16/fp32
        w1            [E, 2I, H]    同 dtype (gate|up 沿 N 维前后拼接)
        w2            [E, H, I]     同 dtype
        topk_weights  [M, topk]     fp32 (上游 fused_topk softmax/renorm 后的路由权重)
        topk_ids      [M, topk]     int32, 每个 token 内不重复
        out           [M, H]        与 hidden_states 同 dtype

    与 kernel 的差异:
        - kernel 分块/moe_align_block_size 只是调度细节, 数学等价 (fp32 累加器)。
        - 不含 fp8/int8 量化、expert_map (EP)、bias、apply_router_weight_on_input、
          gelu/swigluoai 变体; 标杆覆盖标准 silu (SwiGLU) 路径。
    """

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, w1, w2, topk_weights, topk_ids):
        M, H = hidden_states.shape
        E, N2, _ = w1.shape
        I = N2 // 2
        topk = topk_ids.shape[1]
        out = torch.zeros(M, H, dtype=torch.float32)
        x = hidden_states.float()
        for m in range(M):
            for kk in range(topk):
                e = int(topk_ids[m, kk])
                y = x[m] @ w1[e].float().T           # [2I]
                inter = torch.nn.functional.silu(y[:I]) * y[I:]
                out[m] += float(topk_weights[m, kk]) * (inter @ w2[e].float().T)
        return out.to(hidden_states.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "5_FusedMoeKernel.json")
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
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]

        M, H = shapes["hidden_states"]
        E, N2, _ = shapes["w1"]
        topk = shapes["topk_ids"][1]
        assert shapes["w2"] == [E, H, N2 // 2]
        assert topk <= E

        # 物理约束: 真实 MoE 的 hidden_states 是 post-RMSNorm 激活, 逐元素 RMS ~1;
        # 标准分布 (±5) 经 SwiGLU 的 silu(gate)*up 乘法结构会放大到 ~1e5, fp16 溢出,
        # 故按真实量级缩放 x0.2 (分布形状不变); 真实 MoE 权重是训练收敛的小幅值矩阵,
        # 按 fan-in 缩放 (Xavier 量级)
        hidden_states = (random_tensor(shapes["hidden_states"], dtypes["hidden_states"])
                         .float() * 0.2).to(dtypes["hidden_states"])
        w1 = random_tensor(shapes["w1"], dtypes["w1"]).float()
        w2 = random_tensor(shapes["w2"], dtypes["w2"]).float()
        # 物理约束: 训练收敛的权重零均值 (标准分布正态分支的大均值会让
        # y = x@w.T 出现直流分量, gate 恒为深负值时 silu 饱和为 0 / 恒为深正值时
        # silu(gate)*up 二次爆炸), 先去均值再按 fan-in 缩放
        w1 = ((w1 - w1.mean()) * (H ** -0.5)).to(dtypes["w1"])
        w2 = ((w2 - w2.mean()) * ((N2 // 2) ** -0.5)).to(dtypes["w2"])

        # 物理约束: topk_ids 每个 token 内不重复 (路由语义);
        # topk_weights 为 softmax(router_logits) 的输出: 正值且每个 token 和为 1
        topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).int()
        topk_weights = torch.softmax(torch.randn(M, topk), dim=-1).float()

        input_groups.append([hidden_states, w1, w2, topk_weights, topk_ids])
    return input_groups


def get_init_inputs():
    return []