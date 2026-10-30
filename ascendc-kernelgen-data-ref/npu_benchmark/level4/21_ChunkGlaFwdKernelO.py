import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ChunkGlaFwdKernelO —— chunk GLA 输出（inter 状态贡献 + intra 块内因果注意力）
#
# 借鉴 ops-ref `chunk_kda_fwd` AscendC kernel 的「输出段」算法。每条序列按
# C=chunk_size 分块，块内局部位置 t、s（A 为前级 intra 产出的 Aqk，scale 已含在
# A 内，kernel 不再对 A 乘 scale）：
#   o[t] = Σ_{s≤t} A[t,s]·v[s]  +  scale·(q[t]·e^{g[t]}) @ h[chunk]
# 即 输出 = 块内因果注意力(A@v) + 块初始状态贡献(q 经自然对数域门控 e^g 衰减后
# 查询状态 h)。该 kernel 用 exp（自然对数域，g 为 chunk 内 inclusive cumsum）。
#
# 原版逐 chunk Python 循环；本版本把整条序列补零到 C 的整数倍、reshape 为
# [B, M, C, ...] 后一次 batched einsum 完成全部 chunk（对应 kernel 按 chunk
# 分核、块内向量化计算 o_intra 与 o_state 的两段流水）。A 的严格上三角与尾块
# 越界列用一次 tril + 列有效性 mask 归零（对应 kernel 的 mask 强制下三角与
# boundary_check 零填充）。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Chunked GLA output（融合分块向量化版）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, v, g, A, h, o=None, scale=None, cu_seqlens=None, chunk_size=64):
        """
        q: [B, T, H, K]；v: [B, T, H, V]；g: [B, T, H, K]（自然对数域 chunk 内
        inclusive cumsum）；A: [B, T, H, C]（前级 Aqk，严格上三角无效）；h: [B, M, H, K, V]。
        Returns o: [B, T, H, V]（dtype 与 q 一致）。
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        BT = chunk_size
        scale = scale if scale is not None else K ** -0.5

        if cu_seqlens is None:
            return self._o_segment(q, v, g, A, h, scale, BT)
        else:
            cu = cu_seqlens.tolist()
            o = torch.zeros(B, T, H, V, device=q.device, dtype=q.dtype)
            gc = 0  # 全局 chunk 计数（varlen h 按全局 chunk 序）
            for n in range(len(cu) - 1):
                s, e = cu[n], cu[n + 1]
                seg_len = e - s
                seg_M = (seg_len + BT - 1) // BT
                seg_o = self._o_segment(
                    q[:, s:e], v[:, s:e], g[:, s:e], A[:, s:e],
                    h[gc:gc + seg_M], scale, BT)
                o[:, s:e] = seg_o
                gc += seg_M
            return o

    def _o_segment(self, q, v, g, A, h, scale, BT):
        B, T, H, K = q.shape
        V = v.shape[-1]
        dev = q.device
        M = (T + BT - 1) // BT
        Tp = M * BT
        pad = Tp - T

        qp = F.pad(q.float(), (0, 0, 0, 0, 0, pad))      # [B, Tp, H, K]
        vp = F.pad(v.float(), (0, 0, 0, 0, 0, pad))      # [B, Tp, H, V]
        gp = F.pad(g.float(), (0, 0, 0, 0, 0, pad))      # [B, Tp, H, K]
        Ap = F.pad(A.float(), (0, 0, 0, 0, 0, pad))      # [B, Tp, H, C]

        qr = qp.view(B, M, BT, H, K)
        vr = vp.view(B, M, BT, H, V)
        gr = gp.view(B, M, BT, H, K)
        Ar = Ap.view(B, M, BT, H, BT)

        # ---- 块内因果 mask + 尾块列有效性（kernel: mask 强制下三角, 越界列按 0） ----
        tril = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=dev), 0)
        col_valid = torch.ones(M, BT, dtype=torch.bool, device=dev)
        L_last = T - (M - 1) * BT
        if L_last < BT:
            col_valid[M - 1, L_last:] = False
        A_eff = Ar * tril[None, None, :, None, :] * col_valid[None, :, None, None, :]

        # ---- 块内因果注意力 o_intra = tril(A) @ v ----
        o_intra = torch.einsum('bmths,bmshv->bmthv', A_eff, vr)

        # ---- 块初始状态贡献 o_state = scale·(q·e^g) @ h ----
        qg = qr * scale * torch.exp(gr)
        o_state = torch.einsum('bmchk,bmhkv->bmchv', qg, h.float())

        o = (o_intra + o_state).reshape(B, Tp, H, V)[:, :T]
        return o.to(q.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "21_ChunkGlaFwdKernelO.json")
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
    for case in cases:
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

        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        v = random_tensor(shapes["v"], dtypes["v"])
        A = random_tensor(shapes["A"], dtypes["A"]) * scale
        h = random_tensor(shapes["h"], dtypes["h"])
        o = torch.zeros(shapes["o"], dtype=dtypes["o"])

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
