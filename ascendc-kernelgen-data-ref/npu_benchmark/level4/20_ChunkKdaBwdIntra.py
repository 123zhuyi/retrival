import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


# ---------------------------------------------------------------------------
# ChunkKdaBwdIntra —— 融合 chunk 内 KDA 反向（chunk_kda_fwd_update / kda_bwd）
#
# 借鉴 ops-ref `chunk_kda_fwd_update` AscendC kernel 的块内算法：前向把每条序列
# 切成 C=chunk_size 的 chunk，块内矩阵（Aqk/Akk/w/u/v_new）一次向量化计算，并把
# 这些中间量（qg/kg/v_new/h）导出供反向使用。本反向即消费这些块内中间量，把
# 逐 token 的 delta-rule 梯度改写为「整块一次向量化」：
#
#   记块内衰减矩阵 E[t,s,d] = 2^(g[t,d]-g[s,d])（s <= t，块外/上三角为 0），
#   局部位置 t、s，q/k 经 head_map = arange(HV)//G 由 H 头扩展到 HV 头：
#     dq2[t,d] = Σ_{s<=t} dAqk[t,s] · k[s,d] · E[t,s,d]
#     dkA[t,d] = Σ_{s<=t} dAkk[t,s] · k[s,d] · E[t,s,d]      （未乘 beta）
#     dkt[t,d] = Σ_{s>=t} (dAqk[s,t]·q[s,d] + dAkk[s,t]·β_s·k[s,d]) · E[s,t,d]
#     dq_out = dq + dq2
#     dk_out = dk + β_t·dkA + dkt
#     db_out = db + Σ_d dkA·k                                  （fp32 累加）
#     dg_out = dg + q·dq2 + (β_t·dkA - dkt)·k                  （fp32 累加）
#
# 实现要点（对齐 kernel 的「整块向量化」）：
#   1. 每条序列 pad 到 C 的倍数，reshape 成 [B, M, C, HV, ...]（M = 块数），
#      E/dAqk 的上三角与块外列天然为 0，pad 行的 k/q/E 为 0，故一次 batched
#      einsum 等价于原逐 chunk 循环，且消除了 Python 块循环。
#   2. q/k 的 H→HV 头扩展用一次 gather（head_map = arange(HV)//G）完成。
#   3. dg 为「对自然对数门控 cumsum 的梯度」（kernel 全程 exp2，链式 ln2 与
#      下游 cumsum 反向的 RCP_LN2 相消，故不含 ln2）。
#   4. db/dg 以 fp32 累加返回，dq/dk 回铸输入 dtype（kernel 语义）。
# ---------------------------------------------------------------------------


def _pad_T(x, pad):
    """把 x 的 T 维（dim1）末尾补 pad 个 0。x 任意 n 维。"""
    if pad == 0:
        return x
    n = x.dim()
    pads = [0] * (2 * n)
    pads[2 * (n - 1 - 1) + 1] = pad          # dim1 的 right pad
    return F.pad(x, tuple(pads))


class Model(nn.Module):
    """
    chunk_kda_bwd_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                        cu_seqlens, chunk_indices, chunk_size, safe_gate)
        -> (dq_out, dk_out, db_out, dg_out)

    delta 规则块内反向的整块向量化实现（与 Triton/kernel 数学语义一致）。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, E,
                cu_seqlens=None, chunk_indices=None, chunk_size=64, safe_gate=False):
        B, T, H, K = k.shape
        HV = g.shape[2]
        C = chunk_size
        G = HV // H

        if cu_seqlens is not None:
            # varlen：逐逻辑序列独立分块（kernel 变长路径）
            return self._forward_varlen(
                q, k, beta, dAqk, dAkk, dq, dk, db, dg, E,
                cu_seqlens, C, G, H, HV, K, dq.dtype, dk.dtype)

        # ---- 头扩展：q/k 从 H 头 gather 到 HV 头（kernel GQA 头映射 head_map）----
        # 注：dq/dk/db/dg 输入本身已是 HV 头，不参与 head_map 扩展。
        head_map = torch.arange(HV, device=k.device) // G     # [HV]
        q_hv = q.float()[:, :, head_map, :].contiguous()       # [B, T, HV, K]
        k_hv = k.float()[:, :, head_map, :].contiguous()
        b_ = beta.float()                                      # [B, T, HV]
        dAq = dAqk.float()                                     # [B, T, HV, C]
        dAk = dAkk.float()
        dq_in = dq.float().contiguous()                        # [B, T, HV, K]（已是 HV）
        dk_in = dk.float().contiguous()
        db_in = db.float()                                     # [B, T, HV]
        dg_in = dg.float().contiguous()
        E_ = E.float()                                         # [B, T, C, HV, K]

        # ---- pad 到 C 的倍数并 reshape 为块（一次向量化整块反向）----
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T
        q_p = _pad_T(q_hv, pad).view(B, Tp // C, C, HV, K)
        k_p = _pad_T(k_hv, pad).view(B, Tp // C, C, HV, K)
        b_p = _pad_T(b_, pad).view(B, Tp // C, C, HV)
        dAq_p = _pad_T(dAq, pad).view(B, Tp // C, C, HV, C)
        dAk_p = _pad_T(dAk, pad).view(B, Tp // C, C, HV, C)
        dq_p = _pad_T(dq_in, pad).view(B, Tp // C, C, HV, K)
        dk_p = _pad_T(dk_in, pad).view(B, Tp // C, C, HV, K)
        db_p = _pad_T(db_in, pad).view(B, Tp // C, C, HV)
        dg_p = _pad_T(dg_in, pad).view(B, Tp // C, C, HV, K)
        E_p = _pad_T(E_, pad).view(B, Tp // C, C, C, HV, K)    # [B, M, C(t), C(s), HV, K]

        # 块内反向（对照上方公式；dAq/E 的 dim2 = 行 token i，dim3/末维 = 列 token j）
        # dq2 / dkA： Σ_j dA[·,i,j] · k[j] · E[i,j]（E 下三角，自动截断 j<=i 与块外）
        dq2 = torch.einsum('bmihj,bmjhd,bmijhd->bmihd', dAq_p, k_p, E_p)
        dkA = torch.einsum('bmihj,bmjhd,bmijhd->bmihd', dAk_p, k_p, E_p)
        db2 = torch.einsum('bmihd,bmihd->bmih', dkA, k_p)
        kb = k_p * b_p.unsqueeze(-1)                           # [B, M, C, HV, K]
        # dkt[·, j] = Σ_i dA[·,i,j] · (q|βk)[i] · E[i,j]：输出 token = j（E 的列），
        # 求和 over 行 i（源 token），E 下三角保证只取 i>=j 的项。
        dkt = (torch.einsum('bmihj,bmihd,bmijhd->bmjhd', dAq_p, q_p, E_p) +
               torch.einsum('bmihj,bmihd,bmijhd->bmjhd', dAk_p, kb, E_p))

        dq_out = dq_p + dq2
        dk_out = dk_p + b_p.unsqueeze(-1) * dkA + dkt
        db_out = db_p + db2
        dg_out = dg_p + q_p * dq2 + (b_p.unsqueeze(-1) * dkA - dkt) * k_p

        # 去 pad、回铸（dq/dk -> 输入 dtype；db/dg -> fp32，kernel 语义）
        dq_out = dq_out.view(B, Tp, HV, K)[:, :T].reshape(B, T, HV, K).to(dq.dtype)
        dk_out = dk_out.view(B, Tp, HV, K)[:, :T].reshape(B, T, HV, K).to(dk.dtype)
        db_out = db_out.view(B, Tp, HV)[:, :T].reshape(B, T, HV).float()
        dg_out = dg_out.view(B, Tp, HV, K)[:, :T].reshape(B, T, HV, K).float()
        return dq_out, dk_out, db_out, dg_out

    def _forward_varlen(self, q, k, beta, dAqk, dAkk, dq, dk, db, dg, E,
                        cu_seqlens, C, G, H, HV, K, dq_dt, dk_dt):
        """varlen 兜底：逐逻辑序列走同一整块向量化逻辑。"""
        B, T, H, K = k.shape
        head_map = torch.arange(HV, device=k.device) // G
        q_hv = q.float()[:, :, head_map, :]
        k_hv = k.float()[:, :, head_map, :]
        b_ = beta.float()
        dAq = dAqk.float()
        dAk = dAkk.float()
        dq_in = dq.float()          # [B, T, HV, K]（已是 HV，不 gather）
        dk_in = dk.float()
        db_in = db.float()
        dg_in = dg.float()
        E_ = E.float()
        cu = cu_seqlens.tolist()

        dq_out = torch.zeros_like(dq_in)
        dk_out = torch.zeros_like(dk_in)
        db_out = torch.zeros_like(db_in)
        dg_out = torch.zeros_like(dg_in)
        for n in range(len(cu) - 1):
            t0, t1 = cu[n], cu[n + 1]
            L = t1 - t0
            Tp = ((L + C - 1) // C) * C
            pad = Tp - L
            sl = slice(t0, t1)
            q_s = _pad_T(q_hv[:, sl], pad).view(1, Tp // C, C, HV, K)
            k_s = _pad_T(k_hv[:, sl], pad).view(1, Tp // C, C, HV, K)
            b_s = _pad_T(b_[:, sl], pad).view(1, Tp // C, C, HV)
            dAq_s = _pad_T(dAq[:, sl], pad).view(1, Tp // C, C, HV, C)
            dAk_s = _pad_T(dAk[:, sl], pad).view(1, Tp // C, C, HV, C)
            dq_s = _pad_T(dq_in[:, sl], pad).view(1, Tp // C, C, HV, K)
            dk_s = _pad_T(dk_in[:, sl], pad).view(1, Tp // C, C, HV, K)
            db_s = _pad_T(db_in[:, sl], pad).view(1, Tp // C, C, HV)
            dg_s = _pad_T(dg_in[:, sl], pad).view(1, Tp // C, C, HV, K)
            E_s = _pad_T(E_[:, sl], pad).view(1, Tp // C, C, C, HV, K)

            dq2 = torch.einsum('bmihj,bmjhd,bmijhd->bmihd', dAq_s, k_s, E_s)
            dkA = torch.einsum('bmihj,bmjhd,bmijhd->bmihd', dAk_s, k_s, E_s)
            db2 = torch.einsum('bmihd,bmihd->bmih', dkA, k_s)
            kb = k_s * b_s.unsqueeze(-1)
            dkt = (torch.einsum('bmihj,bmihd,bmijhd->bmjhd', dAq_s, q_s, E_s) +
                   torch.einsum('bmihj,bmihd,bmijhd->bmjhd', dAk_s, kb, E_s))

            dq_out[:, sl] = (dq_s + dq2).view(1, Tp, HV, K)[:, :L]
            dk_out[:, sl] = (dk_s + b_s.unsqueeze(-1) * dkA + dkt).view(1, Tp, HV, K)[:, :L]
            db_out[:, sl] = (db_s + db2).view(1, Tp, HV)[:, :L]
            dg_out[:, sl] = (dg_s + q_s * dq2 + (b_s.unsqueeze(-1) * dkA - dkt)
                             * k_s).view(1, Tp, HV, K)[:, :L]

        return (dq_out.reshape(B, T, HV, K).to(dq_dt),
                dk_out.reshape(B, T, HV, K).to(dk_dt),
                db_out.reshape(B, T, HV).float(),
                dg_out.reshape(B, T, HV, K).float())


def _decay_matrix(g, cu_seqlens, chunk_size):
    """前向衰减矩阵（原 forward 内的 E 构造，挪到输入生成阶段）。

    返回 fp32 稠密填充张量 [B, T, BT, HV, K]：
        E[b, t, j, hv, d] = 2^(g[b,t,hv,d] - g[b,c+j,hv,d])
        c 为 t 所在 chunk 的起点，仅 j <= t-c 有效，块外及 s > t 处为 0。
    """
    B, T, HV, K = g.shape
    BT = chunk_size
    g_flat = g.float().reshape(B * T, HV, K)
    E = torch.zeros(B * T, BT, HV, K)

    if cu_seqlens is None:
        bounds = [(b * T, (b + 1) * T) for b in range(B)]
    else:
        cu = cu_seqlens.tolist()
        bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]

    for (t0, t1) in bounds:
        c = t0
        while c < t1:
            e = min(c + BT, t1)
            L = e - c
            g_c = g_flat[c:e]                            # [L, HV, K]
            D = g_c.unsqueeze(1) - g_c.unsqueeze(0)      # [L(t), L(s), HV, K]
            lower = torch.tril(torch.ones(L, L, dtype=torch.bool))
            E[c:e, :L] = torch.where(lower[:, :, None, None],
                                     torch.exp2(D), torch.zeros(1))
            c = e
    return E.reshape(B, T, BT, HV, K)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "20_ChunkKdaBwdIntra.json")
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
        cu_seqlens = None
        chunk_size = 64
        chunk_indices = None
        safe_gate = False
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "chunk_size":
                chunk_size = inp["value"]
            elif name == "chunk_indices":
                chunk_indices = None if inp["value"] is None else \
                    torch.tensor(inp["value"], dtype=torch.int64)
            elif name == "safe_gate":
                safe_gate = inp["value"]

        # q/k: L2 归一化，与前向算子及真实 KDA 场景(qk-norm)保持一致
        q = F.normalize(random_tensor(shapes["q"], torch.float32), p=2, dim=-1).to(dtypes["q"])
        k = F.normalize(random_tensor(shapes["k"], torch.float32), p=2, dim=-1).to(dtypes["k"])
        beta = torch.empty(shapes["beta"], dtype=dtypes["beta"]).uniform_(0.0, 1.0)
        dAqk = random_tensor(shapes["dAqk"], dtypes["dAqk"])
        dAkk = random_tensor(shapes["dAkk"], dtypes["dAkk"])
        dq = random_tensor(shapes["dq"], dtypes["dq"])
        dk = random_tensor(shapes["dk"], dtypes["dk"])
        db = random_tensor(shapes["db"], dtypes["db"])
        dg = random_tensor(shapes["dg"], dtypes["dg"])

        # g: kernel 约定输入为 log2 域的 chunk 内 inclusive cumsum（逐 token 负增量）
        B, T, HV, K = shapes["g"]
        delta = -torch.empty(B, T, HV, K).uniform_(0.01, 0.2)
        if cu_seqlens is None:
            bounds = [(b * T, (b + 1) * T) for b in range(B)]
        else:
            cu = cu_seqlens.tolist()
            bounds = [(cu[n], cu[n + 1]) for n in range(len(cu) - 1)]
        g = delta.clone()
        delta_flat = delta.reshape(B * T, HV, K)
        g_flat = g.reshape(B * T, HV, K)
        for (t0, t1) in bounds:
            for c0 in range(t0, t1, chunk_size):
                c1 = min(c0 + chunk_size, t1)
                g_flat[c0:c1] = torch.cumsum(delta_flat[c0:c1], dim=0)
        g = g.to(dtypes["g"])

        E = _decay_matrix(g, cu_seqlens, chunk_size)     # 前向中间量，预计算

        input_groups.append([q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, E,
                             cu_seqlens, chunk_indices, chunk_size, safe_gate])
    return input_groups


def get_init_inputs():
    return []
