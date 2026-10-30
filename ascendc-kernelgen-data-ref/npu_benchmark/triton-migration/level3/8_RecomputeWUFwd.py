import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Model that performs KDA chunked forward WY-representation recomputation.
    fla recompute_w_u_fwd_kernel (KDA 变体, fla/ops/kda/wy_fast.py:
    recompute_w_u_fwd_kda_kernel; Ascend NPU 变体:
    fla/ops/kda/backends/triton_ascend/wy_fast.py:
    recompute_w_u_fwd_kda_kernel_npu, 两者数学语义完全一致):
        recompute_w_u_fwd(k, v, beta, A, gk, q, cu_seqlens) -> (w, u, qg, kg)

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致):
    每条序列内按 BT (= A.shape[-1], 即 chunk_size) 分块, 支持 GQA
    (i_h = i_hv // (HV // H), k/q 按 H 头取, 其余按 HV 头), 逐通道门控 gk
    为 log2 域 chunk 内 inclusive cumsum (上游 chunk_local_cumsum 产物)。
    A [B, T, HV, BT] 为块内单位下三角逆矩阵 (上游 chunk_kda_fwd_intra 的
    Akk = (I + tril(Akk_raw, -1))^{-1} 输出), 每个块 (长度 L <= BT) 内:
        w  = A @ (k * beta * 2^gk)            # WY 表示 w
        u  = A @ (v * beta)                   # WY 表示 u
        kg = k * 2^(gk[last] - gk)            # 衰减到块末, 供块间状态更新
        qg = q * 2^gk                         # 仅 q 非 None 时输出 (STORE_QG)
    注:
      1. 尾块 L < BT 时仅前 L 列 A 有效 (kernel boundary_check 零填充等价);
         kg 的 last 为块内最后一个有效 token (min(c+BT, eos) - 1);
      2. 内部 fp32 计算, w/kg/qg 回铸 k.dtype, u 回铸 v.dtype;
      3. q/k 输入必须是 L2 归一化后的向量(真实 KDA 场景层内先做 qk-norm,
         与 A 的生成自洽);
      4. 支持 cu_seqlens varlen ([1, total_T, ...] 布局)。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, k, v, beta, A, gk, q=None, cu_seqlens=None):
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[-1]
        BT = A.shape[-1]
        G = HV // H  # qk 头与 v 头的分组映射: i_h = i_hv // G

        k_f, v_f, g_f = k.float(), v.float(), gk.float()
        b_f, A_f = beta.float(), A.float()
        q_f = q.float() if q is not None else None

        k_flat = k_f.reshape(B * T, H, K)
        v_flat = v_f.reshape(B * T, HV, V)
        g_flat = g_f.reshape(B * T, HV, K)
        b_flat = b_f.reshape(B * T, HV)
        A_flat = A_f.reshape(B * T, HV, BT)
        q_flat = q_f.reshape(B * T, H, K) if q_f is not None else None

        w = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device)
        u = torch.zeros(B * T, HV, V, dtype=torch.float32, device=k.device)
        kg = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device)
        qg = torch.zeros(B * T, HV, K, dtype=torch.float32, device=k.device) \
            if q is not None else None

        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

        head_map = torch.arange(HV, device=k.device) // G

        for (t0, t1) in bounds:
            c = t0
            while c < t1:
                e = min(c + BT, t1)  # chunk 结束(不含), 不足 BT 的尾块按实际长度
                L = e - c
                k_c = k_flat[c:e][:, head_map]           # [L, HV, K]
                v_c = v_flat[c:e]                        # [L, HV, V]
                g_c = g_flat[c:e]                        # [L, HV, K]
                b_c = b_flat[c:e]                        # [L, HV]
                A_c = A_flat[c:e, :, :L]                 # [L, HV, L] 仅前 L 列有效

                kb = k_c * b_c[..., None] * torch.exp2(g_c)   # [L, HV, K]
                vb = v_c * b_c[..., None]                     # [L, HV, V]
                w[c:e] = torch.einsum('ths,shd->thd', A_c, kb)
                u[c:e] = torch.einsum('ths,shv->thv', A_c, vb)
                kg[c:e] = k_c * torch.exp2(g_c[L - 1:L] - g_c)
                if qg is not None:
                    q_c = q_flat[c:e][:, head_map]
                    qg[c:e] = q_c * torch.exp2(g_c)
                c = e

        dt = k.dtype
        w = w.reshape(B, T, HV, K).to(dt)
        u = u.reshape(B, T, HV, V).to(v.dtype)
        kg = kg.reshape(B, T, HV, K).to(dt)
        if qg is not None:
            qg = qg.reshape(B, T, HV, K).to(dt)
        return w, u, qg, kg


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "8_RecomputeWUFwd.json")
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
        cu_seqlens = None
        has_q = True
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = None if inp["value"] is None else \
                    torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "has_q":
                has_q = inp["value"]

        # q/k: 必须 L2 归一化 (真实 KDA 场景层内先做 qk-norm; 且 ||k||~sqrt(K) 时
        # (I+Akk)^{-1} 会组合爆炸, kernel 与 golden 同时溢出, 已实测)
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"]) \
            if has_q else None
        v = random_tensor(shapes["v"], dtypes["v"])
        # beta 物理含义是 sigmoid 门控系数, 取值 (0, 1)
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)

        # gk: kernel 约定输入为 log2 域的 chunk 内 inclusive cumsum。
        # 这里先生成逐 token 负增量 (log2 域, 保证块内衰减 <= 1),
        # 再按序列边界和 BT (= A.shape[-1]) 分段做 cumsum, 与上游 chunk_local_cumsum 对齐
        B, T, HV, K = shapes["gk"]
        H = shapes["k"][2]
        G = HV // H
        BT = shapes["A"][-1]
        assert shapes["A"] == [B, T, HV, BT]
        delta = -torch.empty(B, T, HV, K).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        gk = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        gk_flat = gk.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, BT):
                c1 = min(c0 + BT, t1)
                gk_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        gk = gk.to(dtypes["gk"])

        # A: 与上游 chunk_kda_fwd_intra 输出自洽的块内单位下三角逆矩阵
        # Akk_raw[t,s] = beta_t * sum_d k[t,d]*k[s,d]*2^(gk[t,d]-gk[s,d]) (s < t)
        # A = (I + tril(Akk_raw, -1))^{-1} (fp32 解下三角系统, 回铸 A.dtype)
        k_flat32 = k.float().reshape(B * T, H, K)
        g_flat32 = gk.float().reshape(B * T, HV, K)
        b_flat32 = beta.float().reshape(B * T, HV)
        A_flat = torch.zeros(B * T, HV, BT, dtype=torch.float32)
        head_map = torch.arange(HV) // G
        for (t0, t1) in bounds:
            c = t0
            while c < t1:
                e = min(c + BT, t1)
                L = e - c
                k_c = k_flat32[c:e][:, head_map]          # [L, HV, K]
                g_c = g_flat32[c:e]                       # [L, HV, K]
                b_c = b_flat32[c:e]                       # [L, HV]
                D = g_c.unsqueeze(1) - g_c.unsqueeze(0)   # [L(t), L(s), HV, K]
                strict = torch.tril(torch.ones(L, L, dtype=torch.bool), -1)
                DecS = torch.where(strict[:, :, None, None], torch.exp2(D),
                                   torch.zeros(1))
                Akk_raw = torch.einsum('thd,shd,tshd->hts', k_c, k_c, DecS) \
                    * b_c.t()[:, :, None]
                eye = torch.eye(L).expand(HV, L, L)
                Akk_inv = torch.linalg.solve_triangular(eye + Akk_raw, eye,
                                                        upper=False)
                A_flat[c:e, :, :L] = Akk_inv.permute(1, 0, 2)
                c = e
        A = A_flat.reshape(B, T, HV, BT).to(dtypes["A"])

        input_groups.append([k, v, beta, A, gk, q, cu_seqlens])
    return input_groups


def get_init_inputs():
    return []