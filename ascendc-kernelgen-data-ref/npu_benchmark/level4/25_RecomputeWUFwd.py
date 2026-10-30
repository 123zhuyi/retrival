import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RecomputeWUFwd —— KDA chunked forward 的 WY 表示重算（w/u/qg/kg）
#
# 借鉴 ops-ref `chunk_kda_fwd` AscendC kernel 的「重算段」算法（fla 的
# recompute_w_u_fwd_kda_kernel）。每条序列按 BT (= A.shape[-1]) 分块，A 为块内
# 单位下三角逆 (I+tril(Akk_raw,-1))^{−1}（前级 intra 输出），块内：
#   w   = A @ (k·β·2^gk)            # WY 表示 w
#   u   = A @ (v·β)                 # WY 表示 u
#   kg  = k·2^(gk_last − gk)        # 衰减到块末, 供块间状态更新
#   qg  = q·2^gk                     # q 非 None 时输出 (STORE_QG)
#
# 原版逐 chunk Python 循环；本版本把整条序列补零到 BT 的整数倍、reshape 为
# [B, M, BT, ...] 后一次 batched einsum 完成全部 chunk（对应 kernel 按 chunk
# 分核、块内一次 matmul 得到 w/u）。A 的尾块越界列用列有效性 mask 归零（对应
# kernel boundary_check 零填充）；gk 末块后向填充最后一个有效值保证 kg 正确。
# GQA：k/q 按 H 头取，经 head_map (i_h = i_hv // G) 展开到 HV 头。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """KDA WY-representation recomputation（融合分块向量化版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, k, v, beta, A, gk, q=None, cu_seqlens=None):
        """
        k: [B, T, H, K]；v: [B, T, HV, V]；beta: [B, T, HV]；A: [B, T, HV, BT]
        （块内单位下三角逆）；gk: [B, T, HV, K]（log2 域 chunk 内 cumsum）；
        q: [B, T, H, K] 可选。
        Returns (w, u, qg, kg)；qg 仅 q 非 None 时非 None。
        """
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        BT = A.shape[-1]
        G = HV // H

        if cu_seqlens is None:
            w, u, qg, kg = self._wu_segment(k, v, beta, A, gk, q, BT, G)
        else:
            cu = cu_seqlens.tolist()
            w = torch.zeros(B, T, HV, K, device=k.device, dtype=torch.float32)
            u = torch.zeros(B, T, HV, V, device=k.device, dtype=torch.float32)
            kg = torch.zeros(B, T, HV, K, device=k.device, dtype=torch.float32)
            qg = torch.zeros(B, T, HV, K, device=k.device, dtype=torch.float32) \
                if q is not None else None
            for n in range(len(cu) - 1):
                s, e = cu[n], cu[n + 1]
                wi, ui, qgi, kgi = self._wu_segment(
                    k[:, s:e], v[:, s:e], beta[:, s:e], A[:, s:e], gk[:, s:e],
                    q[:, s:e] if q is not None else None, BT, G)
                w[:, s:e], u[:, s:e], kg[:, s:e] = wi, ui, kgi
                if qg is not None:
                    qg[:, s:e] = qgi

        dt = k.dtype
        w = w.to(dt)
        u = u.to(v.dtype)
        kg = kg.to(dt)
        if qg is not None:
            qg = qg.to(dt)
        return w, u, qg, kg

    def _wu_segment(self, k, v, beta, A, gk, q, BT, G):
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        dev = k.device
        M = (T + BT - 1) // BT
        Tp = M * BT
        pad = Tp - T

        kp = F.pad(k.float(), (0, 0, 0, 0, 0, pad))       # [B, Tp, H, K]
        vp = F.pad(v.float(), (0, 0, 0, 0, 0, pad))       # [B, Tp, HV, V]
        gp = F.pad(gk.float(), (0, 0, 0, 0, 0, pad))      # [B, Tp, HV, K]
        bp = F.pad(beta.float(), (0, 0, 0, pad))          # [B, Tp, HV]
        Ap = F.pad(A.float(), (0, 0, 0, 0, 0, pad))       # [B, Tp, HV, BT]
        qp = F.pad(q.float(), (0, 0, 0, 0, 0, pad)) if q is not None else None

        L_last = T - (M - 1) * BT
        if L_last < BT:
            gr_view = gp.view(B, M, BT, HV, K)
            gr_view[:, M - 1, L_last:] = gr_view[:, M - 1, L_last - 1:L_last]
            gp = gr_view.reshape(B, Tp, HV, K)

        head_map = torch.arange(HV, device=dev) // G
        ke = kp[..., head_map, :].contiguous()            # [B, Tp, HV, K]
        qe = qp[..., head_map, :].contiguous() if qp is not None else None

        kr = ke.view(B, M, BT, HV, K)
        vr = vp.view(B, M, BT, HV, V)
        gr = gp.view(B, M, BT, HV, K)
        br = bp.view(B, M, BT, HV)
        Ar = Ap.view(B, M, BT, HV, BT)

        # ---- A 尾块列有效性（kernel boundary_check 零填充） ----
        col_valid = torch.ones(M, BT, dtype=torch.bool, device=dev)
        if L_last < BT:
            col_valid[M - 1, L_last:] = False
        A_eff = Ar * col_valid[None, :, None, None, :]

        # ---- w/u：A @ (kβ2^gk / vβ) ----
        kb = kr * br.unsqueeze(-1) * torch.exp2(gr)
        vb = vr * br.unsqueeze(-1)
        w = torch.einsum('bmths,bmshk->bmthk', A_eff, kb)
        u = torch.einsum('bmths,bmshv->bmthv', A_eff, vb)

        # ---- kg = k·2^(gk_last−gk)；qg = q·2^gk ----
        gk_last = gr[:, :, -1:, :]
        kg = kr * torch.exp2(gk_last - gr)
        qg = qe.view(B, M, BT, HV, K) * torch.exp2(gr) if qe is not None else None

        def trim(x4):
            return x4.reshape(B, Tp, *x4.shape[3:])[:, :T]

        return trim(w), trim(u), (trim(qg) if qg is not None else None), trim(kg)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "25_RecomputeWUFwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
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

        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"]) \
            if has_q else None
        v = random_tensor(shapes["v"], dtypes["v"])
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)

        B, T, HV, K = shapes["gk"]
        H = shapes["k"][2]
        G = HV // H
        BT = shapes["A"][-1]
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
                k_c = k_flat32[c:e][:, head_map]
                g_c = g_flat32[c:e]
                b_c = b_flat32[c:e]
                D = g_c.unsqueeze(1) - g_c.unsqueeze(0)
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
