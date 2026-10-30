import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mamba2Bwd —— Mamba-2 SSD chunk-scan 反向（逐 token 精确伴随）
#
# 对标 ops-ref `chunk_gated_delta_rule` / mamba_ssm `_mamba_chunk_scan_combined_bwd`
# 的反向递推（数据依赖使反向逐 token 串行）。前向（记 a_t = exp(A_h·dt'_t)，
# W_t = dt'_t·x_t ⊗ B_t）：
#   S_t = a_t ⊙ S_{t-1} + dt'_t·x_t ⊗ B_t
#   y_t = C_t^T·S_t (+ D·x_t)；z 门控 silu(z)·y_t
# 反向（dS 从 dfinal_states 初始化，逆序传播）：
#   dy    = dout_t·silu(z_t)
#   dz_t  = dout_t·y_t·silu'(z_t)
#   dC_t  = S_t^T·dy ; dS += dy ⊗ C_t          (读出路径)
#   dD   += dy ⊙ x_t ; dx_t = D ⊙ dy            (D 跳连路径)
#   dx_t += dt'_t·(dS·B_t)                       (写入路径)
#   dB_t  = dt'_t·(dS^T·x_t)
#   ddt'_t += Σ_{p,n} dS·(x_t⊗B_t)
#   ddA_t = a_t·Σ(dS ⊙ S_{t-1})                 (∂S_t/∂dA_t = a_t⊙S_{t-1})
#   dA_h += ddA_t·dt'_t ; ddt'_t += ddA_t·A_h   (衰减路径)
#   dS   <- a_t ⊙ dS                            (传给上一步)
#   dinitial_states = 递推结束时的 dS
# dt 预处理链反传（与 _chunk_cumsum_bwd 一致）：
#   d(raw) = ddt'·[lo≤sp≤hi]·(σ(raw) 若 raw≤20 否则 1)
#   ddt = d(raw) ; ddt_bias[h] = Σ_{b,t} d(raw)
#
# 前向中间量（S_all/y_all/a/dtr/raw/sp/sil/dsil）由 get_input_groups 预计算传入，
# forward 内零前向计算、仅做反向递推。GQA：dB/dC 组内各头求和恢复 [B,T,G,N]。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Mamba-2 SSD chunk-scan 反向（融合显式递推版）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, D, z, dt_bias, initial_states,
                dout, dfinal_states,
                S_all, y_all, a, dtr, raw, sp, sil, dsil,
                chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi):
        device = x.device
        Bsz, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        R = H // G

        xf = x.float()
        Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
        Ch = C.float().repeat_interleave(R, dim=2)
        has_z = z.shape[0] > 0
        has_init = initial_states.shape[0] > 0
        has_dD = D.numel() > 0
        Af = A.float()
        Df = D.float() if has_dD else None
        Dv = (Df.view(1, H, 1) if Df.dim() == 1
              else Df.view(1, H, P)) if has_dD else None

        S0 = (initial_states.float() if has_init
              else torch.zeros(Bsz, H, P, N, device=device))

        dof = dout.float()
        dxf = torch.zeros_like(xf)
        dBh = torch.zeros_like(Bh)
        dCh = torch.zeros_like(Ch)
        ddtp = torch.zeros_like(dtr)
        dAf = torch.zeros_like(Af)
        dDf = torch.zeros_like(Df) if has_dD else None
        dzf = (torch.zeros(Bsz, T, H, P, device=device)
               if has_z else None)
        dS = (dfinal_states.float().clone() if dfinal_states.shape[0] > 0
              else torch.zeros(Bsz, H, P, N, device=device))

        for t in reversed(range(T)):
            dy = dof[:, t] * sil[:, t] if has_z else dof[:, t]
            if has_z:
                dzf[:, t] = dof[:, t] * y_all[:, t] * dsil[:, t]
            S_t = S_all[t]
            dCh[:, t] = (S_t * dy.unsqueeze(-1)).sum(-2)          # [B,H,N]
            dS = dS + dy.unsqueeze(-1) * Ch[:, t].unsqueeze(-2)   # dy ⊗ C_t
            if has_dD:
                acc = dy * xf[:, t]                               # [B,H,P]
                dDf += acc.sum((0, 2)) if Df.dim() == 1 else acc.sum(0)
                dxf[:, t] = Dv * dy
            # S_t = a⊙S_{t-1} + dt'·x⊗B
            dxf[:, t] += dtr[:, t].unsqueeze(-1) * (dS * Bh[:, t].unsqueeze(-2)).sum(-1)
            dBh[:, t] = dtr[:, t].unsqueeze(-1) * (dS * xf[:, t].unsqueeze(-1)).sum(-2)
            ddtp[:, t] += (dS * (xf[:, t].unsqueeze(-1)
                                 * Bh[:, t].unsqueeze(-2))).sum((-1, -2))
            S_prev = S_all[t - 1] if t > 0 else S0
            ddA = a[:, t] * (dS * S_prev).sum((-1, -2))           # [B,H]
            dAf += (ddA * dtr[:, t]).sum(0)
            ddtp[:, t] += ddA * Af
            dS = a[:, t].unsqueeze(-1).unsqueeze(-1) * dS
        dinitial = dS

        # dt 预处理链反传
        unclamped = (sp >= dt_limit_lo) & (sp <= dt_limit_hi)
        draw = ddtp * unclamped
        if dt_softplus:
            draw = draw * torch.where(raw <= 20.0, torch.sigmoid(raw),
                                      torch.ones_like(raw))
        ddt_bias = draw.sum((0, 1))                               # [H]

        # GQA：组内各头梯度求和，恢复 [B,T,G,N]
        dB = dBh.view(Bsz, T, G, R, N).sum(3).to(B.dtype)
        dC = dCh.view(Bsz, T, G, R, N).sum(3).to(C.dtype)

        dz_out = dzf.to(z.dtype) if has_z else torch.empty(0, device=device)
        dinit_out = dinitial if has_init else torch.empty(0, H, P, N, device=device)
        return (dxf.to(x.dtype), draw.to(dt.dtype), dAf, dB, dC,
                dDf if has_dD else torch.empty(0, device=device),
                dz_out, ddt_bias, dinit_out)


def _forward_intermediates(x, dt, A, B, C, D, z, dt_bias, initial_states,
                           dt_softplus, dt_limit_lo, dt_limit_hi):
    """前向计算（原 forward 内的重算部分），全部在输入生成阶段完成。

    返回（均 fp32）：S_all [T,B,H,P,N]、y_all [B,T,H,P]、a [B,T,H]、
    dtr [B,T,H]（clamp 后 dt'）、raw [B,T,H]、sp [B,T,H]、sil/dsil。
    """
    Bsz, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    R = H // G

    xf = x.float()
    Bh = B.float().repeat_interleave(R, dim=2)
    Ch = C.float().repeat_interleave(R, dim=2)
    has_z = z.shape[0] > 0
    has_init = initial_states.shape[0] > 0
    has_dD = D.numel() > 0

    raw = dt.float() + dt_bias.float()
    sp = F.softplus(raw) if dt_softplus else raw
    dtr = sp.clamp(dt_limit_lo, dt_limit_hi)
    a = (dtr * A.float()).exp()
    Dv = None
    if has_dD:
        Df = D.float()
        Dv = Df.view(1, H, 1) if Df.dim() == 1 else Df.view(1, H, P)

    S = (initial_states.float().clone() if has_init
         else torch.zeros(Bsz, H, P, N))
    S_all = torch.empty(T, Bsz, H, P, N)
    y_all = torch.empty(Bsz, T, H, P)
    for t in range(T):
        S = (a[:, t].unsqueeze(-1).unsqueeze(-1) * S
             + (dtr[:, t].unsqueeze(-1).unsqueeze(-1)
                * xf[:, t].unsqueeze(-1)) * Bh[:, t].unsqueeze(-2))
        S_all[t] = S
        y_all[:, t] = (S * Ch[:, t].unsqueeze(-2)).sum(-1)
    if has_dD:
        y_all = y_all + xf * Dv.unsqueeze(0)

    if has_z:
        zf = z.float()
        sig = torch.sigmoid(zf)
        sil = zf * sig
        dsil = sig * (1.0 + zf * (1.0 - sig))
    else:
        sil = torch.empty(0)
        dsil = torch.empty(0)

    return S_all, y_all, a, dtr, raw, sp, sil, dsil


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "34_Mamba2Bwd.json")
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
        shapes, dtypes, attrs = {}, {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                dtypes[inp["name"]] = dtype_map[inp["dtype"]]
            else:
                attrs[inp["name"]] = inp["value"]

        Bsz, T, H, P = shapes["x"]
        G, N = shapes["B"][2], shapes["B"][3]
        chunk_size = int(attrs["chunk_size"])
        dt_softplus = int(attrs["dt_softplus"])
        dt_limit_lo = float(attrs["dt_limit_lo"])
        dt_limit_hi = float(attrs["dt_limit_hi"])

        x = random_tensor(shapes["x"], dtypes["x"])
        dt = random_tensor(shapes["dt"], dtypes["dt"])
        A = (-torch.exp(torch.rand(shapes["A"]) * math.log(16.0))).to(dtypes["A"])
        B = random_tensor(shapes["B"], dtypes["B"])
        C = random_tensor(shapes["C"], dtypes["C"])
        D = random_tensor(shapes["D"], dtypes["D"])
        if shapes["z"][0] == 0:
            z = torch.empty(shapes["z"], dtype=dtypes["z"])
        else:
            z = random_tensor(shapes["z"], dtypes["z"])
        dt_bias = (torch.rand(shapes["dt_bias"]) * 4.0 - 5.0).to(dtypes["dt_bias"])
        if shapes["initial_states"][0] == 0:
            initial_states = torch.empty(shapes["initial_states"],
                                         dtype=dtypes["initial_states"])
        else:
            initial_states = random_tensor(shapes["initial_states"],
                                           dtypes["initial_states"])
        dout = random_tensor(shapes["dout"], dtypes["dout"])
        if shapes["dfinal_states"][0] == 0:
            dfinal_states = torch.empty(shapes["dfinal_states"],
                                        dtype=dtypes["dfinal_states"])
        else:
            dfinal_states = random_tensor(shapes["dfinal_states"],
                                          dtypes["dfinal_states"])

        S_all, y_all, a, dtr, raw, sp, sil, dsil = _forward_intermediates(
            x, dt, A, B, C, D, z, dt_bias, initial_states,
            dt_softplus, dt_limit_lo, dt_limit_hi)

        input_groups.append([x, dt, A, B, C, D, z, dt_bias, initial_states,
                             dout, dfinal_states,
                             S_all, y_all, a, dtr, raw, sp, sil, dsil,
                             chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi])
    return input_groups


def get_init_inputs():
    return []
