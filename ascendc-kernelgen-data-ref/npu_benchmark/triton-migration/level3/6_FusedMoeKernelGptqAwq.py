import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that performs fused MoE expert computation with GPTQ/AWQ int4 weights.
    对标 vLLM fused_moe_kernel_gptq_awq 的两段式调用
    (vllm/model_executor/layers/fused_moe/fused_moe.py:61, v0.26.0;
    编排逻辑同 fused_experts_impl: GEMM1 -> silu_and_mul -> GEMM2(mul_routed_weight) -> sum)。

    数学定义:
        反量化 (int4, 沿 K 两两打包进 int32, 小端: 偶数 k 在低 4 bit):
            q[k, n]   = (w_q[k//2, n] >> (4 * (k % 2))) & 0xF
            zp[k, n]  = (w_zp[k//gs, n//2] >> (4 * (n % 2))) & 0xF   (has_zp=1, AWQ)
                      = 8                                            (has_zp=0, GPTQ)
            w[k, n]   = (q[k, n] - zp[k, n]) * scale[k//gs, n]       (fp32)
        MoE 主体与 fused_moe_kernel 完全一致:
            y = hidden[m] @ w1[e].T; inter = silu(y[:I]) * y[I:]
            out[m] += topk_weights[m, k] * (inter @ w2[e].T); out = sum_topk(...)

    布局约定 (kernel 的 B 为 [E, K//2, N] int32, scale/zp 的 group 维沿 K):
        hidden_states [M, H]            fp16/bf16/fp32
        w1_q          [E, H//2, 2I]     int32 (w1: K=H, N=2I)
        w1_scale      [E, H//gs, 2I]    fp16/bf16/fp32
        w1_zp         [E, H//gs, I]     int32 (N//2 = I; has_zp=0 时首维为 0 的占位)
        w2_q          [E, I//2, H]      int32 (w2: K=I, N=H)
        w2_scale      [E, I//gs, H]     同 scale dtype
        w2_zp         [E, I//gs, H//2]  int32 (同上占位约定)
        topk_weights  [M, topk]         fp32; topk_ids [M, topk] int32
        group_size    int32 attr (32/64/128/或=K 表示 per-channel)
        has_zp        int32 attr (1=AWQ 带零点, 0=GPTQ 隐含零点 8)
        out           [M, H] 与 hidden_states 同 dtype

    与 kernel 的差异:
        - 不含 int8 (w8a16)、act_order (g_idx)、SPLIT_K; 标杆覆盖 int4 w4a16
          的 GPTQ (zp=8) 与 AWQ (显式 zp) 两条主路径, 数学与 kernel 逐 bit 一致。
        - 分块调度 (moe_align/sorted_token_ids) 只是执行顺序, 数学等价。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _dequant(w_q, scale, w_zp, group_size, has_zp):
        # 解包 int4: [E, K//2, N] int32 -> q [E, K, N] (偶数 k 低 4 bit)
        even = (w_q & 0xF)
        odd = ((w_q >> 4) & 0xF)
        q = torch.stack([even, odd], dim=2).reshape(w_q.shape[0], -1, w_q.shape[2])
        q = q.float()
        E, K, N = q.shape
        gs = group_size
        if has_zp:
            # zp 沿 N 两两打包 (偶数 n 低 4 bit), 配对维必须在最后一维再 reshape
            zp_e = (w_zp & 0xF)
            zp_o = ((w_zp >> 4) & 0xF)
            zp = torch.stack([zp_e, zp_o], dim=3).reshape(E, -1, N).float()
        else:
            zp = torch.full((E, K // gs, N), 8.0, dtype=torch.float32)
        s = scale.float().repeat_interleave(gs, dim=1)      # [E, K, N]
        zp = zp.repeat_interleave(gs, dim=1)                # [E, K, N]
        # w[k, n] = (q - zp) * scale, 布局转回 [E, N, K] 便于 GEMM
        return ((q - zp) * s).permute(0, 2, 1).contiguous()

    def forward(self, hidden_states, w1_q, w1_scale, w1_zp, w2_q, w2_scale,
                w2_zp, topk_weights, topk_ids, group_size: int, has_zp: int):
        M, H = hidden_states.shape
        w1 = self._dequant(w1_q, w1_scale, w1_zp, group_size, has_zp)  # [E, 2I, H]
        w2 = self._dequant(w2_q, w2_scale, w2_zp, group_size, has_zp)  # [E, H, I]
        I = w1.shape[1] // 2
        topk = topk_ids.shape[1]
        out = torch.zeros(M, H, dtype=torch.float32)
        x = hidden_states.float()
        for m in range(M):
            for kk in range(topk):
                e = int(topk_ids[m, kk])
                y = x[m] @ w1[e].T
                inter = torch.nn.functional.silu(y[:I]) * y[I:]
                out[m] += float(topk_weights[m, kk]) * (inter @ w2[e].T)
        return out.to(hidden_states.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "6_FusedMoeGptqAwq.json")
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

    def quantize_int4(w, gs, has_zp):
        """w [E, N, K] fp32 -> (w_q [E, K//2, N] int32, scale [E, K//gs, N], zp)"""
        E, N, K = w.shape
        G = K // gs
        wg = w.permute(0, 2, 1).reshape(E, G, gs, N)          # [E, G, gs, N]
        if has_zp:  # AWQ: 非对称, 覆盖 [min, max]
            wmin = wg.amin(dim=2, keepdim=True)
            wmax = wg.amax(dim=2, keepdim=True)
            scale = ((wmax - wmin) / 15.0).clamp_min(1e-4)
            zp = torch.round(-wmin / scale).clamp(0, 15)
            q = torch.round(wg / scale + zp).clamp(0, 15)
        else:       # GPTQ: 对称, 隐含零点 8
            wmax = wg.abs().amax(dim=2, keepdim=True)
            scale = (wmax / 7.0).clamp_min(1e-4)
            zp = torch.full_like(scale, 8.0)
            q = torch.round(wg / scale + zp).clamp(0, 15)
        q = q.reshape(E, G * gs, N).to(torch.int32)           # [E, K, N]
        # 沿 K 两两打包 (偶数 k 低 4 bit)
        q_even, q_odd = q[:, 0::2], q[:, 1::2]
        w_q = (q_even | (q_odd << 4)).contiguous()            # [E, K//2, N]
        scale = scale.reshape(E, G, N).contiguous()
        zp = zp.reshape(E, G, N).to(torch.int32)
        # zp 沿 N 两两打包 (偶数 n 低 4 bit)
        zp_even, zp_odd = zp[:, :, 0::2], zp[:, :, 1::2]
        w_zp = (zp_even | (zp_odd << 4)).contiguous()         # [E, G, N//2]
        return w_q, scale, w_zp

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        group_size, has_zp = 128, 1
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "group_size":
                group_size = inp["value"]
            elif name == "has_zp":
                has_zp = inp["value"]

        M, H = shapes["hidden_states"]
        E, Kh2, N2 = shapes["w1_q"]          # [E, H//2, 2I]
        I = N2 // 2
        gs = group_size
        assert Kh2 * 2 == H and H % gs == 0 and I % gs == 0 and I % 2 == 0 and H % 2 == 0
        assert shapes["w1_scale"] == [E, H // gs, N2]
        assert shapes["w2_q"] == [E, I // 2, H]
        assert shapes["w2_scale"] == [E, I // gs, H]
        topk = shapes["topk_ids"][1]
        assert topk <= E

        # 物理约束: 与 FusedMoe 相同 —— hidden 按真实激活量级 x0.2,
        # 权重零均值 + fan-in 缩放, 再做 int4 量化 (量化在 fp32 权重上进行,
        # 与真实 GPTQ/AWQ 流程一致)
        hidden_states = (random_tensor(shapes["hidden_states"], dtypes["hidden_states"])
                         .float() * 0.2).to(dtypes["hidden_states"])
        w1 = random_tensor([E, N2, H], torch.float32)
        w1 = (w1 - w1.mean()) * (H ** -0.5)
        w2 = random_tensor([E, H, I], torch.float32)
        w2 = (w2 - w2.mean()) * (I ** -0.5)

        w1_q, w1_scale, w1_zp = quantize_int4(w1, gs, has_zp)
        w2_q, w2_scale, w2_zp = quantize_int4(w2, gs, has_zp)
        w1_scale = w1_scale.to(dtypes["w1_scale"])
        w2_scale = w2_scale.to(dtypes["w2_scale"])
        if not has_zp:  # GPTQ: zp 不参与, 传占位空张量
            w1_zp = torch.zeros(0, dtype=torch.int32)
            w2_zp = torch.zeros(0, dtype=torch.int32)

        # 物理约束: topk_ids 每个 token 内不重复; topk_weights 为 softmax 输出
        topk_ids = torch.stack([torch.randperm(E)[:topk] for _ in range(M)]).int()
        topk_weights = torch.softmax(torch.randn(M, topk), dim=-1).float()

        input_groups.append([hidden_states, w1_q, w1_scale, w1_zp,
                             w2_q, w2_scale, w2_zp,
                             topk_weights, topk_ids, gs, has_zp])
    return input_groups


def get_init_inputs():
    return []