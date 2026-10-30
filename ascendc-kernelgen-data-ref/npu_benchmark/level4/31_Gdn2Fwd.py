import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import os


# ---------------------------------------------------------------------------
# Gdn2Fwd —— GDN-2（Gated DeltaNet 2）chunkwise 前向，融合分块版
#
# 借鉴 ops-ref `chunk_gated_delta_rule` / `chunk_kda_fwd` 的分块流水线算法。
# 逐 token 递推（S ∈ R^{K×V}，* 为逐通道 Hadamard 积）：
#     S <- S * exp(g_t)                       (K 轴逐通道衰减)
#     erase = (b_t * k_t)^T S                  (擦除门读衰减后状态)
#     v_new = w_t * v_t - erase                (写入门 delta 规则)
#     S <- S + k_t ⊗ v_new                     (秩一更新)
#     o_t = (scale * q_t)^T S                  (更新后读出)
# 被改写为「块内矩阵 + 块间状态传播」的分块递推（C = 64）：
#     gk    = cumsum(g/ln2)                    # chunk-local log2 累计门控 [K]
#     kr    = b ⊙ k                            # 擦除读键
#     decay = 2^{clamp(gk_i - gk_j, 0)}        # 逐 K 通道衰减矩阵
#     Aqk   = tril(Σ_K scale·q_i·k_j·decay)    # 块内 QK 系数（含对角线）
#     L     = tril(Σ_K kr_i·k_j·decay)         # 块内擦除干扰矩阵（严格下三角）
#     w = (I+L)^{-1}(kr·2^gk),  u = (I+L)^{-1}(w⊙v)
#     v_new = u - w·h_c                        # 相对入块状态的残差
#     o_c   = qg·h_c + Aqk·v_new,  qg = scale·q·2^gk
#     h_{c+1} = 2^{gk_last}⊙h_c + kg^T·v_new,  kg = k·2^{gk_last-gk}
#
# 与逐步递推数学等价，把 O(T) 次顺序小步压缩为 O(T/C) 次向量化矩阵块。
# 全程 fp32，输出 o 转回 v.dtype，final_state 保持 fp32。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)
_CHUNK = 64


class Model(nn.Module):
    """
    GDN-2 chunkwise forward（分块融合版，语义对齐 chunk_gated_delta_rule）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, g, b, w, initial_state, cu_seqlens, scale,
                use_qk_l2norm_in_kernel):
        device = q.device
        B, T, H, K = q.shape
        V = v.shape[-1]
        qf, kf = q.float(), k.float()
        if use_qk_l2norm_in_kernel:
            # fla l2norm: y = x * rsqrt(sum(x^2) + 1e-6)
            qf = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
        vf, gf, bf, wf = v.float(), g.float(), b.float(), w.float()

        has_init = initial_state is not None and initial_state.shape[0] > 0
        if cu_seqlens is not None and len(cu_seqlens) > 0:
            assert B == 1, "varlen 要求 B=1"
            N = len(cu_seqlens) - 1
            o = torch.empty(1, T, H, V, dtype=torch.float32, device=device)
            final = torch.empty(N, H, K, V, dtype=torch.float32, device=device)
            for n in range(N):
                s, e = int(cu_seqlens[n]), int(cu_seqlens[n + 1])
                S0 = (initial_state[n:n + 1].float() if has_init
                      else torch.zeros(1, H, K, V, device=device))
                o_n, S_n = self._chunk_segment(
                    qf[:, s:e], kf[:, s:e], vf[:, s:e],
                    gf[:, s:e], bf[:, s:e], wf[:, s:e], S0, scale)
                o[:, s:e] = o_n
                final[n] = S_n[0]
        else:
            N = B
            S0 = initial_state.float() if has_init else torch.zeros(B, H, K, V, device=device)
            o, S_n = self._chunk_segment(qf, kf, vf, gf, bf, wf, S0, scale)
            final = S_n
        return o.to(v.dtype), final

    def _chunk_segment(self, qf, kf, vf, gf, bf, wf, S0, scale):
        """整段序列分块递推，返回 (o([B,T,H,V] fp32), S([B,H,K,V] fp32))。
        qf/kf/gf/bf [B,T,H,K]; vf/wf [B,T,H,V]。gf 为 log 域衰减（逐 K 通道），
        bf 为擦除门（逐 K 通道），wf 为写入门（逐 V 通道）。"""
        B, T, H, K = qf.shape
        V = vf.shape[-1]
        dev = qf.device
        C = _CHUNK
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T
        # 转 [B,H,T,...] 布局后 pad T 维；pad 空 token：k/b/g=0、v/w=0
        qp = F.pad(qf.transpose(1, 2), (0, 0, 0, pad))     # [B,H,Tp,K]
        kp = F.pad(kf.transpose(1, 2), (0, 0, 0, pad))
        gp = F.pad(gf.transpose(1, 2), (0, 0, 0, pad))
        bp = F.pad(bf.transpose(1, 2), (0, 0, 0, pad))
        vp = F.pad(vf.transpose(1, 2), (0, 0, 0, pad))     # [B,H,Tp,V]
        wp = F.pad(wf.transpose(1, 2), (0, 0, 0, pad))

        S = S0.contiguous()                       # [B,H,K,V] fp32
        o = torch.zeros(B, H, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32)
        M = Tp // C
        for c in range(M):
            qc = qp[:, :, c * C:(c + 1) * C]      # [B,H,C,K]
            kc = kp[:, :, c * C:(c + 1) * C]
            gc = gp[:, :, c * C:(c + 1) * C]
            bc = bp[:, :, c * C:(c + 1) * C]
            vc = vp[:, :, c * C:(c + 1) * C]      # [B,H,C,V]
            wc = wp[:, :, c * C:(c + 1) * C]

            gk = gc.cumsum(2) / _LN2                      # [B,H,C,K] log2 累计
            kr = bc * kc                                  # 擦除读键 = b⊙k
            qs = qc * scale                               # scale 折进 q

            decay = torch.exp2(torch.clamp(
                gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))  # [B,H,C,C,K]

            Aqk = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', qs, kc, decay),
                             diagonal=0)
            L = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', kr, kc, decay),
                           diagonal=-1)
            A_plus_L = eye + L

            w_sol = torch.linalg.solve_triangular(
                A_plus_L, kr * torch.exp2(gk), upper=False, unitriangular=True)
            u_sol = torch.linalg.solve_triangular(
                A_plus_L, wc * vc, upper=False, unitriangular=True)

            v_new = u_sol - torch.einsum('bhik,bhkv->bhiv', w_sol, S)  # [B,H,C,V]

            qg = qs * torch.exp2(gk)
            o[:, :, c * C:(c + 1) * C] = torch.einsum('bhik,bhkv->bhiv', qg, S) \
                + torch.einsum('bhij,bhjv->bhiv', Aqk, v_new)
            kg = kc * torch.exp2(gk[:, :, -1:, :] - gk)
            S = torch.exp2(gk[:, :, -1:, :]).transpose(-1, -2) * S \
                + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

        return o[:, :, :T].transpose(1, 2).contiguous(), S


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "31_Gdn2Fwd.json")
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

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens, scale, use_l2norm = None, None, 1
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens":
                cu_seqlens = inp["value"]
            elif name == "scale":
                scale = inp["value"]
            elif name == "use_qk_l2norm_in_kernel":
                use_l2norm = inp["value"]

        B, T, H, K = shapes["q"]
        V = shapes["v"][-1]
        N = len(cu_seqlens) - 1 if cu_seqlens else B
        if cu_seqlens:
            assert B == 1 and cu_seqlens[-1] == T
        assert shapes["initial_state"][0] in (0, N)

        q = random_tensor(shapes["q"], dtypes["q"])
        k = random_tensor(shapes["k"], dtypes["k"])
        v = random_tensor(shapes["v"], dtypes["v"])
        # 物理约束: use_qk_l2norm_in_kernel=0 时上游必须先对 q/k 做 L2 归一化
        # (否则 delta rule 的 (I - k*(b*k)^T) 在 ||k|| 大时特征值越界发散,
        # kernel 本身也会 nan), 这里按真实模型行为预归一化
        if not use_l2norm:
            q = F.normalize(q.float(), p=2, dim=-1).to(dtypes["q"])
            k = F.normalize(k.float(), p=2, dim=-1).to(dtypes["k"])
        # 物理约束: g 是 log 域逐通道衰减, 恒 <= 0 (真实模型来自
        # -exp(A_log)*softplus(...))。沿用官方测试生成器 _rand_inputs 的
        # uniform(-5, -0.1), dtype 跟随输入 (tests/ops/test_gdn2.py)
        g = torch.empty(shapes["g"], dtype=torch.float32).uniform_(-5.0, -0.1).to(dtypes["g"])
        # 物理约束: b (擦除门) 与 w (写入门) 是 sigmoid 类门控, 恒在 (0, 1)
        b = torch.rand(shapes["b"], dtype=dtypes["b"])
        w = torch.rand(shapes["w"], dtype=dtypes["w"])
        if shapes["initial_state"][0] == 0:
            initial_state = torch.zeros(0, H, K, V, dtype=torch.float32)
        else:
            # 初始状态是历史递推结果, 幅值与 v 同量级; 标准分布生成即可
            initial_state = random_tensor(shapes["initial_state"],
                                          torch.float32)

        cu_t = torch.tensor(cu_seqlens, dtype=torch.int32) if cu_seqlens else None
        input_groups.append([q, k, v, g, b, w, initial_state, cu_t,
                             scale, use_l2norm])
    return input_groups


def get_init_inputs():
    return []
