import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs chunked GLA output computation (KDA pipeline 的输出阶段).
    chunk_gla_fwd_o_gk(q, v, g, A, h, o, scale, cu_seqlens, chunk_size) -> o
    (vLLM vllm/model_executor/layers/fla/ops/kda.py, 底层 kernel: chunk_gla_fwd_kernel_o)

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致, 已闭环验证):
    每条序列内按 chunk_size 分块, 块内局部位置 t, s:
        o[t] = sum_{s<=t} A[t,s] * v[s]  +  scale * (q[t] * e^g[t]) @ h[chunk]
    即 输出 = 块内因果注意力(A@v) + 块初始状态贡献(q 经逐通道门控衰减后查询 h)。
    注:
      1. 该 kernel 用 exp (自然对数域), 与 fla 仓 KDA 系列的 exp2 (log2 域) 不同,
         g 是自然对数域的 chunk 内 inclusive cumsum;
      2. A 是前向 intra 阶段产出的 Aqk, scale 已含在 A 内, kernel 不再对 A 乘 scale,
         scale 只作用于状态贡献项的 q; A 的严格上三角 kernel 侧强制置 0;
      3. h 布局: 定长 [B, NT, H, K, V], varlen [NT_total, H, K, V] (按全局 chunk 序);
      4. o 是 kernel 侧的预分配输出参数(原地写回), golden 忽略其内容, 内部新建返回;
      5. q 输入为 L2 归一化向量, 与 KDA 系列算子及真实场景(qk-norm)一致。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, v, g, A, h, o=None, scale=None, cu_seqlens=None, chunk_size=64):
        torch.manual_seed(42)
        B, T, H, K = q.shape
        V = v.shape[-1]
        BT = chunk_size

        q_flat = q.float().reshape(B * T, H, K)
        v_flat = v.float().reshape(B * T, H, V)
        g_flat = g.float().reshape(B * T, H, K)
        A_flat = A.float().reshape(B * T, H, BT)
        o_flat = torch.zeros(B * T, H, V, dtype=torch.float32, device=q.device)

        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        gc = 0  # varlen 全局 chunk 计数
        for bi, (t0, t1) in enumerate(bounds):
            c = t0
            while c < t1:
                e = min(c + BT, t1)  # 不足 BT 的尾块按实际长度
                L = e - c
                q_c, g_c = q_flat[c:e], g_flat[c:e]      # [L, H, K]
                v_c = v_flat[c:e]                        # [L, H, V]
                A_c = A_flat[c:e, :, :L]                 # [L, H, L]
                if cu_seqlens is None:
                    h_c = h[bi, (c - t0) // BT].float()  # [H, K, V]
                else:
                    h_c = h[gc].float()

                # 块内因果注意力: kernel 用 mask 强制下三角, 越界列按 0 处理
                lower = torch.tril(torch.ones(L, L, dtype=torch.bool, device=q.device))
                A_m = torch.where(lower[:, None, :], A_c, torch.zeros(1, device=q.device))
                o_intra = torch.einsum('ths,shv->thv', A_m, v_c)
                # 块初始状态贡献: scale * (q * e^g) @ h
                qg = q_c * scale * torch.exp(g_c)
                o_state = torch.einsum('thk,hkv->thv', qg, h_c)
                o_flat[c:e] = o_intra + o_state
                c = e
                gc += 1

        return o_flat.reshape(B, T, H, V).to(q.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_ChunkGLAFwdKernelO.json")
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
    for case in cases:
        # 先解析 attr (g 的 cumsum 依赖 cu_seqlens 和 chunk_size)
        shapes, dtypes = {}, {}
        scale = None
        cu_seqlens = None
        chunk_size = 64
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "scale":
                scale = inp["value"]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "chunk_size":
                chunk_size = inp["value"]

        # q: L2 归一化, 与 KDA 系列算子及真实场景(qk-norm)一致
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        # v: 实际为前向产出的 v_new (delta rule 校正后的值), 量级 O(1)
        v = random_tensor(shapes["v"], dtypes["v"])
        # A: 前向 intra 产出的 Aqk, scale 已含在内, 量级 ~±scale;
        # 严格上三角 kernel 侧强制置 0, 全随机生成即可
        A = random_tensor(shapes["A"], dtypes["A"]) * scale
        # h: 各 chunk 的初始状态, 定长 [B, NT, H, K, V], varlen [NT_total, H, K, V]
        h = random_tensor(shapes["h"], dtypes["h"])
        # o: kernel 侧预分配输出参数(原地写回), 内容无关, 给零初始化占位
        o = torch.zeros(shapes["o"], dtype=dtypes["o"])

        # g: kernel 用 exp, 约定输入为自然对数域的 chunk 内 inclusive cumsum。
        # 先生成逐 token 负增量 (自然对数域, 保证块内衰减 <= 1),
        # 再按序列边界和 chunk_size 分段做 cumsum, 与上游 chunk_local_cumsum 对齐
        B, T, H, K = shapes["g"]
        delta = -torch.empty(B, T, H, K).uniform_(0.01, 0.15)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        g = delta.clone()
        delta_flat = delta.reshape(B * T, H, K)
        g_flat = g.reshape(B * T, H, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                g_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        g = g.to(dtypes["g"])

        input_groups.append([q, v, g, A, h, o, scale, cu_seqlens, chunk_size])
    return input_groups


def get_init_inputs():
    return []