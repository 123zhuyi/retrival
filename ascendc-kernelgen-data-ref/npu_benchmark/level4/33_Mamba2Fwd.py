import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


# ---------------------------------------------------------------------------
# Mamba2Fwd —— Mamba-2 SSD chunk-scan 前向，融合分块版
#
# 借鉴 ops-ref `chunk_gated_delta_rule` / `chunk_kda_fwd` 的分块递推算法。
# 逐 token 递推（状态 S ∈ R^{P×N}，P=headdim, N=dstate）：
#     dt' = clamp(softplus(dt + dt_bias))           # dt 预处理
#     a_t = exp(dt'_t · A_h)                         # 标量衰减（A 恒负）
#     S_t = a_t·S_{t-1} + dt'_t·(x_t ⊗ B_t)          # 外积写入
#     y_t = S_t·C_t + D ⊙ x_t                        # 更新后读出（+D 跳连）
# 被改写为「块内矩阵 + 块间状态传播」的分块递推（C = chunk_size）：
#     gk    = cumsum(dt'·A / ln2)                    # log2 累计衰减 [B,T,H]
#     v     = dt'·x                                  # 写入值 [B,T,H,P]
#     decay = 2^{clamp(gk_i - gk_j, 0)}              # 块内衰减矩阵 [C,C]
#     KKT   = decay ⊙ (C_i · B_j)                    # 块内 C·B 系数（GQA 头组共享）
#     o_c   = (C·2^{gk})@S + KKT@v                   # 块输出 = 状态传播 + 块内贡献
#     S_{c+1} = 2^{gk_last}·S + (v·2^{gk_last-gk})^T@B
#
# 与逐 token 递推数学等价；内部 fp32，输出 y 转回 x.dtype，final_state 转 C.dtype。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)


class Model(nn.Module):
    """
    Mamba-2 SSD chunk-scan forward（分块融合版，语义对齐 mamba_chunk_scan_combined）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, dt, A, B, C, D, z, dt_bias, initial_states,
                chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi):
        device = x.device
        Bsz, T, H, P = x.shape
        G, N = B.shape[2], B.shape[3]
        R = H // G

        xf = x.float()
        Bh = B.float().repeat_interleave(R, dim=2)   # [B,T,H,N]
        Ch = C.float().repeat_interleave(R, dim=2)   # [B,T,H,N]

        # ---- dt 预处理（fp32，与 kernel _chunk_cumsum_fwd 一致）----
        dtr = dt.float() + dt_bias.float()           # [B,T,H]
        if dt_softplus:
            dtr = F.softplus(dtr)   # threshold=20 ↔ kernel where(dt<=20, sp, dt)
        dtr = dtr.clamp(dt_limit_lo, dt_limit_hi)

        # ---- 分块递推 ----
        if initial_states.shape[0] > 0:
            S = initial_states.float().clone()       # [B,H,P,N]
        else:
            S = torch.zeros(Bsz, H, P, N, device=device)

        # 转 [B,H,T,...] 布局
        xf_h = xf.transpose(1, 2)                    # [B,H,T,P]
        dtr_h = dtr.transpose(1, 2)                  # [B,H,T]
        Bh_h = Bh.transpose(1, 2)                    # [B,H,T,N]
        Ch_h = Ch.transpose(1, 2)
        gk_full = (dtr_h * A.float().view(1, H, 1) / _LN2).cumsum(2)  # [B,H,T]

        CS = int(chunk_size)
        Tp = ((T + CS - 1) // CS) * CS
        xp = F.pad(xf_h, (0, 0, 0, Tp - T))
        dtrp = F.pad(dtr_h, (0, Tp - T))
        v = dtrp.unsqueeze(-1) * xp                  # [B,H,Tp,P]
        gkp = F.pad(gk_full, (0, Tp - T))            # 填充区 gk 保持末值
        if Tp > T:
            gkp[:, :, T:] = gkp[:, :, T - 1:T]
        # gk_shift[i] = gk_abs[i-1]（i=0 时为 0）→ 块内用 chunk-local gk
        gk_shift = torch.cat(
            [torch.zeros_like(gkp[:, :, :1]), gkp[:, :, :-1]], dim=2)
        Bp = F.pad(Bh_h, (0, 0, 0, Tp - T))          # [B,H,Tp,N]
        Cp = F.pad(Ch_h, (0, 0, 0, Tp - T))

        ys = []
        for c in range(Tp // CS):
            vc = v[:, :, c * CS:(c + 1) * CS]          # [B,H,C,P]
            Bc = Bp[:, :, c * CS:(c + 1) * CS]         # [B,H,C,N]
            Cc = Cp[:, :, c * CS:(c + 1) * CS]
            # chunk-local gk：减去块首前一个位置的累计值（首块减 0）
            gk = gkp[:, :, c * CS:(c + 1) * CS] \
                - gk_shift[:, :, c * CS:c * CS + 1]     # [B,H,C]

            decay = torch.exp2(torch.clamp(
                gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))   # [B,H,C,C]
            # 因果：只有 j<=i 的写入对 o_i 有贡献（含对角线，update-then-read）
            KKT = torch.tril(decay * torch.einsum('bhcn,bhjn->bhcj', Cc, Bc),
                             diagonal=0)                        # [B,H,C,C]

            qg = Cc * torch.exp2(gk).unsqueeze(-1)   # [B,H,C,N]
            o_c = torch.einsum('bhcn,bhpn->bhcp', qg, S) \
                + torch.einsum('bhcj,bhjp->bhcp', KKT, vc)     # [B,H,C,P]
            ys.append(o_c)

            gk_last = gk[:, :, -1:]                   # [B,H,1]
            S = torch.exp2(gk_last).unsqueeze(-1) * S \
                + torch.einsum('bhjp,bhjn->bhpn',
                               vc * torch.exp2(gk_last - gk).unsqueeze(-1), Bc)

        y = torch.cat(ys, dim=2)[:, :, :T].transpose(1, 2).contiguous()  # [B,T,H,P]

        # ---- D 跳连（在 z 门控之前）；D [H] 按头广播，[H,P] 逐通道 ----
        if D.numel() > 0:
            Df = D.float()
            Dv = Df.view(1, 1, H, 1) if Df.dim() == 1 else Df.view(1, 1, H, P)
            y = y + xf * Dv
        # ---- z 门控（silu，D 残差之后）----
        if z.shape[0] > 0:
            y = y * F.silu(z.float())

        return y.to(x.dtype), S.to(C.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "33_Mamba2Fwd.json")
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
        shapes, dtypes, attrs = {}, {}, {}
        for inp in case["inputs"]:
            if inp.get("type") == "tensor":
                shapes[inp["name"]] = inp["shape"]
                dtypes[inp["name"]] = dtype_map[inp["dtype"]]
            else:
                attrs[inp["name"]] = inp["value"]

        Bsz, T, H, P = shapes["x"]
        G, N = shapes["B"][2], shapes["B"][3]
        assert shapes["dt"] == [Bsz, T, H] and shapes["A"] == [H]
        assert shapes["B"] == shapes["C"] == [Bsz, T, G, N]
        assert H % G == 0
        assert shapes["D"] in ([H], [H, P])
        assert shapes["dt_bias"] == [H]
        assert shapes["z"][0] in (0, Bsz)
        assert shapes["initial_states"][0] in (0, Bsz)
        chunk_size = int(attrs["chunk_size"])
        dt_softplus = int(attrs["dt_softplus"])
        dt_limit_lo = float(attrs["dt_limit_lo"])
        dt_limit_hi = float(attrs["dt_limit_hi"])

        x = random_tensor(shapes["x"], dtypes["x"])
        dt = random_tensor(shapes["dt"], dtypes["dt"])
        # 物理约束: A = -exp(A_log) 恒负, HF/fla 初始化 A_log ∈ (0, ln16)
        A = (-torch.exp(torch.rand(shapes["A"]) * math.log(16.0))).to(dtypes["A"])
        B = random_tensor(shapes["B"], dtypes["B"])
        C = random_tensor(shapes["C"], dtypes["C"])
        D = random_tensor(shapes["D"], dtypes["D"])
        if shapes["z"][0] == 0:      # rmsnorm 路径: kernel 内无 z 门控
            z = torch.empty(shapes["z"], dtype=dtypes["z"])
        else:
            z = random_tensor(shapes["z"], dtypes["z"])
        # 物理约束: dt_bias 初始化为 inv_softplus(U(0.001,0.1)) 量级, 恒负
        dt_bias = (torch.rand(shapes["dt_bias"]) * 4.0 - 5.0).to(dtypes["dt_bias"])
        if shapes["initial_states"][0] == 0:
            initial_states = torch.empty(shapes["initial_states"],
                                         dtype=dtypes["initial_states"])
        else:
            initial_states = random_tensor(shapes["initial_states"],
                                           dtypes["initial_states"])

        input_groups.append([x, dt, A, B, C, D, z, dt_bias, initial_states,
                             chunk_size, dt_softplus, dt_limit_lo, dt_limit_hi])
    return input_groups


def get_init_inputs():
    return []
