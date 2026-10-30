import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ChunkKdaFwdIntra —— chunk 内 KDA 前向中间量（Aqk/Akk/w/u/qg/kg）融合分块向量化
#
# 借鉴 ops-ref `chunk_kda_fwd` AscendC kernel 的「块内」算法。对每条序列按
# C=chunk_size 分块，块内（下标 i=t, j=s 为 chunk 内位置，d 为 K 通道）：
#   Aqk[t,s] = scale · Σ_d q[t,d]·k[s,d]·2^(gk[t,d]−gk[s,d])        (s ≤ t, 含对角线)
#   Akk_raw[t,s] = β_t · Σ_d k[t,d]·k[s,d]·2^(gk[t,d]−gk[s,d])      (s < t, 严格下三角)
#   Akk     = (I + tril(Akk_raw, −1))^{−1}                          (单位下三角逆)
#   w       = Akk @ (k·β·2^gk)                                       (WY 表示 w)
#   u       = Akk @ (v·β)                                            (WY 表示 u)
#   kg      = k·2^(gk_last − gk)                                     (衰减到块末)
#   qg      = q·2^gk                                                  (disable_recompute 时)
#
# 原版逐 chunk Python 循环 + 逐 chunk einsum；本版本把整条序列补零到 C 的整数倍、
# reshape 为 [B, M, C, HV, K] 后一次 batched einsum / batched solve_triangular
# 完成全部 chunk（M = T/C 个块），仅保留 chunk 维作为 batch 维 —— 对应 kernel 按
# chunk 分核、块内整块向量化计算 Aqk/Akk/w/u 的流水。gk 为输入（log2 域 chunk 内
# inclusive cumsum）；数值稳健用 exp2(clamp(gk_t−gk_s, max=0)) 防止上三角上溢
# （对应 PROGRESS_CHUNK 记录的坑）。
#
# 与原版逐块计算数学等价（fp32 累乘），输出 dtype 边界与原版一致
# （Aqk/Akk/w/kg/qg → k.dtype，u → v.dtype；disable_recompute=False 时 qg=None）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Chunked KDA intra-chunk forward（融合分块向量化版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, gk, beta, scale, cu_seqlens=None, chunk_size=64,
                chunk_indices=None, safe_gate=False, disable_recompute=False):
        """
        q,k: [B, T, H, K]；v: [B, T, HV, V]；gk: [B, T, HV, K]（log2 域 chunk 内
        inclusive cumsum）；beta: [B, T, HV]。
        Returns (w, u, qg, kg, Aqk, Akk)；qg 仅 disable_recompute=True 时非 None。
        """
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        G = HV // H

        if cu_seqlens is None:
            w, u, qg, kg, Aqk, Akk = self._intra_segment(
                q, k, v, gk, beta, scale, chunk_size, G, disable_recompute)
        else:
            cu = cu_seqlens.tolist()
            # varlen：每条序列独立分块（B 恒为 1），输出按全局 token 序拼回
            Bsz = q.shape[0]
            Tp_global = T
            out_shape = dict(w=(Bsz, Tp_global, HV, K), u=(Bsz, Tp_global, HV, V),
                             kg=(Bsz, Tp_global, HV, K), Aqk=(Bsz, Tp_global, HV, chunk_size),
                             Akk=(Bsz, Tp_global, HV, chunk_size))
            w = torch.zeros(out_shape["w"], dtype=torch.float32, device=q.device)
            u = torch.zeros(out_shape["u"], dtype=torch.float32, device=q.device)
            kg = torch.zeros(out_shape["kg"], dtype=torch.float32, device=q.device)
            Aqk = torch.zeros(out_shape["Aqk"], dtype=torch.float32, device=q.device)
            Akk = torch.zeros(out_shape["Akk"], dtype=torch.float32, device=q.device)
            qg = torch.zeros(out_shape["w"], dtype=torch.float32, device=q.device) \
                if disable_recompute else None
            for n in range(len(cu) - 1):
                s, e = cu[n], cu[n + 1]
                wi, ui, qgi, kgi, Aqi, Aki = self._intra_segment(
                    q[:, s:e], k[:, s:e], v[:, s:e], gk[:, s:e], beta[:, s:e],
                    scale, chunk_size, G, disable_recompute)
                w[:, s:e], u[:, s:e], kg[:, s:e] = wi, ui, kgi
                Aqk[:, s:e], Akk[:, s:e] = Aqi, Aki
                if qg is not None:
                    qg[:, s:e] = qgi

        dt = k.dtype
        Aqk = Aqk.to(dt)
        Akk = Akk.to(dt)
        w = w.to(dt)
        u = u.to(v.dtype)
        kg = kg.to(dt)
        if qg is not None:
            qg = qg.to(dt)
        return w, u, qg, kg, Aqk, Akk

    def _intra_segment(self, q, k, v, gk, beta, scale, BT, G, disable_recompute):
        """对整段 [B, T, ...] 序列分块向量化，返回各输出 [B, T, HV, ...]（fp32）。"""
        B, T, H, K = k.shape
        HV, V = v.shape[2], v.shape[3]
        dev = q.device
        M = (T + BT - 1) // BT
        Tp = M * BT
        pad = Tp - T

        qf, kf, vf = q.float(), k.float(), v.float()
        gf, bf = gk.float(), beta.float()

        # ---- 补零：q/k/v/beta 补 0；gk 末块后向填充最后一个有效值 ----
        qp = F.pad(qf, (0, 0, 0, 0, 0, pad))          # [B, Tp, H, K]
        kp = F.pad(kf, (0, 0, 0, 0, 0, pad))
        vp = F.pad(vf, (0, 0, 0, 0, 0, pad))          # [B, Tp, HV, V]
        gp = F.pad(gf, (0, 0, 0, 0, 0, pad))          # [B, Tp, HV, K]
        bp = F.pad(bf, (0, 0, 0, pad))                # [B, Tp, HV]

        L_last = T - (M - 1) * BT
        if L_last < BT:
            gr_view = gp.view(B, M, BT, HV, K)
            gr_view[:, M - 1, L_last:] = gr_view[:, M - 1, L_last - 1:L_last]
            gp = gr_view.reshape(B, Tp, HV, K)

        # ---- GQA：q/k 头展开到 HV（i_h = i_hv // G） ----
        head_map = torch.arange(HV, device=dev) // G
        qe = qp[..., head_map, :].contiguous()        # [B, Tp, HV, K]
        ke = kp[..., head_map, :].contiguous()

        qr = qe.view(B, M, BT, HV, K)
        kr = ke.view(B, M, BT, HV, K)
        vr = vp.view(B, M, BT, HV, V)
        gr = gp.view(B, M, BT, HV, K)
        br = bp.view(B, M, BT, HV)

        # ---- 块内衰减 D[t,s,hv,d] = gk[t]−gk[s]，clamp(max=0) 防上三角上溢 ----
        D = gr.unsqueeze(3) - gr.unsqueeze(2)         # [B, M, BT, BT, HV, K]
        decay = torch.exp2(torch.clamp(D, max=0.0))

        tril = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=dev), 0)
        strict = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=dev), -1)

        # ---- Aqk[t,s] = scale·Σ_d q[t]k[s]decay (s≤t) ----
        dec_lower = decay * tril.view(1, 1, BT, BT, 1, 1)
        Aqk = scale * torch.einsum('bmthd,bmshd,bmtshd->bmhts', qr, kr, dec_lower)
        Aqk = Aqk.permute(0, 1, 3, 2, 4)              # [B, M, BT(t), HV, BT(s)]

        # ---- Akk_raw[t,s] = β_t·Σ_d k[t]k[s]decay (s<t)；Akk=(I+Akk_raw)^{−1} ----
        dec_strict = decay * strict.view(1, 1, BT, BT, 1, 1)
        Akk_raw = torch.einsum('bmthd,bmshd,bmtshd->bmhts', kr, kr, dec_strict)
        Br = br.permute(0, 1, 3, 2).unsqueeze(-1)     # [B, M, HV, BT(t), 1]
        Akk_raw = Akk_raw * Br                        # 行缩放 β_t
        eye = torch.eye(BT, device=dev, dtype=torch.float32) \
            .expand(B, M, HV, BT, BT).clone()
        Akk_inv = torch.linalg.solve_triangular(eye + Akk_raw, eye, upper=False)
        Akk = Akk_inv.permute(0, 1, 3, 2, 4)          # [B, M, BT(t), HV, BT(s)]

        # ---- w/u：Akk @ (kβ2^gk / vβ) ----
        g2 = torch.exp2(gr)                           # [B, M, BT, HV, K]
        kb = kr * br.unsqueeze(-1) * g2
        vb = vr * br.unsqueeze(-1)
        w = torch.einsum('bmhts,bmshk->bmthk', Akk_inv, kb)   # [B, M, BT, HV, K]
        u = torch.einsum('bmhts,bmshv->bmthv', Akk_inv, vb)   # [B, M, BT, HV, V]

        # ---- kg = k·2^(gk_last−gk)；qg = q·2^gk ----
        gk_last = gr[:, :, -1:, :]                    # [B, M, 1, HV, K]
        kg = kr * torch.exp2(gk_last - gr)            # [B, M, BT, HV, K]
        qg = qr * g2 if disable_recompute else None

        # ---- 拼回 [B, Tp, ...] 并裁掉补零 ----
        def trim(x4):
            return x4.reshape(B, Tp, *x4.shape[3:])[:, :T]

        Aqk_o = trim(Aqk)                             # [B, T, HV, BT]
        Akk_o = trim(Akk)
        w_o = trim(w)
        u_o = trim(u)
        kg_o = trim(kg)
        qg_o = trim(qg) if qg is not None else None
        return w_o, u_o, qg_o, kg_o, Aqk_o, Akk_o


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "19_ChunkKdaFwdIntra.json")
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
        shapes, dtypes = {}, {}
        scale = None
        cu_seqlens = None
        chunk_size = 64
        chunk_indices = None
        safe_gate = False
        disable_recompute = False
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
            elif name == "chunk_indices":
                chunk_indices = None if inp["value"] is None else \
                    torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "safe_gate":
                safe_gate = inp["value"]
            elif name == "disable_recompute":
                disable_recompute = inp["value"]

        # 交替打开 disable_recompute，覆盖 qg 输出路径
        if "disable_recompute" not in [x.get("name") for x in case["inputs"]]:
            disable_recompute = (case_idx % 4 == 0)

        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)

        B, T, HV, K = shapes["gk"]
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
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                gk_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        gk = gk.to(dtypes["gk"])

        input_groups.append([q, k, v, gk, beta, scale, cu_seqlens, chunk_size,
                             chunk_indices, safe_gate, disable_recompute])
    return input_groups


def get_init_inputs():
    return []
