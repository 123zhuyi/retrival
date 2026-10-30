import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that merges split-K partial attention outputs (log2 域 LSE 加权合并)。
    对标 vLLM _merge_topk_attn_out_kernel
    (vllm/models/minimax_m3/common/ops/sparse_attn.py:419, v0.26.0),
    即 MiniMax-M3 GQA block-sparse decode kernel (split-K over topk 块) 的合并阶段。

    数学定义 (对每个 (token b, head h), C = NUM_TOPK_CHUNKS):
        lse_max = max_c lse_partial[c, b, h]
        w_c     = 2^(lse_partial[c, b, h] - lse_max)   # 空 chunk lse=-inf -> w=0
        w_c    /= sum_c w_c
        out[b, h] = sum_c w_c * o_partial[c, b, h]
    与 softmax 分母合并的标准 flash-decoding 合并完全一致 (log2 域)。

    布局约定:
        o_partial   [C, total_q, H, D]  fp16/bf16/fp32 (与 q/输出同 dtype)
        lse_partial [C, total_q, H]     fp32 (log2 域; 空 chunk 为 -inf)
        out         [total_q, H, D]     与 o_partial 同 dtype

    与 kernel 的差异:
        - 若某 (b, h) 的全部 C 个 chunk 均为空 (lse 全 -inf), kernel 会产出
          nan (exp2(nan)/0), 标杆输出 0 (调用方保证每行至少一个有效 chunk,
          该路径仅防御, 正常输入不会触发)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, o_partial, lse_partial):
        lse = lse_partial.float()                       # [C, total_q, H]
        lse_max = lse.max(dim=0, keepdim=True).values
        weights = torch.exp2(lse - lse_max)             # -inf -> 0
        w_sum = weights.sum(dim=0, keepdim=True)
        # 防御: 全空行 (kernel 为 nan, 标杆输出 0, 见 docstring)
        weights = torch.where(w_sum > 0, weights / w_sum.clamp_min(1e-30),
                              torch.zeros_like(weights))
        out = (o_partial.float() * weights.unsqueeze(-1)).sum(dim=0)
        return out.to(o_partial.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_MergeTopkAttnOut.json")
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
    }

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                dtypes[inp["name"]] = dtype_map[inp["dtype"]]

        C, total_q, H, D = shapes["o_partial"]
        assert shapes["lse_partial"] == [C, total_q, H]

        o_partial = random_tensor(shapes["o_partial"], dtypes["o_partial"])
        # 物理约束: lse_partial 是各 split-K chunk 的 log2(sum exp2(logits)),
        # 量级约 log2(chunk 内有效 token 数) + logit 均值, 典型 [-10, 20];
        # 每行的空 chunk (超出该 (kv_head 组) 有效 topk 的 chunk) 为 -inf。
        # 这里按行随机一个有效 chunk 数 (>=1), 其余置 -inf
        lse = random_tensor(shapes["lse_partial"], torch.float32).clamp(-10.0, 20.0)
        n_valid = torch.randint(1, C + 1, (total_q, H))
        chunk_ids = torch.arange(C).view(C, 1, 1)
        lse = torch.where(chunk_ids < n_valid.view(1, total_q, H),
                          lse, torch.full_like(lse, float("-inf")))

        input_groups.append([o_partial, lse])
    return input_groups


def get_init_inputs():
    return []