import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ChunkKdaWithFusedGateFwd —— 融合门控的 chunk KDA 前向
#
# 借鉴 ops-ref `chunk_kda_fwd`（fused-gate 变体）的分块流水线算法：
#   1. 融合门控（kda_gate_fwd_kernel，全程 fp32）：
#          gamma = -exp(A_log) * softplus(raw_g + g_bias)     # <= 0，逐 K 通道
#   2. 逐 token 递推  S_t = alpha_t·S_{t-1} + k_t ⊗ beta_t(v_t - S_{t-1}^T k_t)
#      被改写为「块内矩阵运算 + 块间状态传播」的分块递推（C = 块长）：
#        gk    = cumsum(gamma / ln2)                 # chunk-local log2 累计 gate
#        qg    = q·2^gk,  kgq = k·2^{-gk},  kg = k·2^{gk_last - gk}
#        Aqk   = tril(qg @ kgq^T) · scale            # 块内 QK 系数矩阵（含衰减）
#        L     = tril((k·2^{gk}) @ kgq^T · beta, 下三角)
#        Akk   = (I + L)^{-1}
#        w = Akk·(beta·k·2^{gk}),  u = Akk·(beta·v)
#        v_new = u - w·h_c
#        o_c   = scale·qg·h_c + Aqk·v_new
#        h_{c+1} = 2^{gk_last} ⊙ h_c + kg^T·v_new
#
# 与逐步递推数学等价，把 O(T) 次顺序小步压缩为 O(T/C) 次向量化矩阵块。
# 全程 fp32，输出 o 转回 v.dtype，final_state 保持 fp32。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)
_CHUNK = 64


class Model(nn.Module):
    """Chunk KDA forward（融合门控 + 分块版，语义对齐 chunk_kda_with_fused_gate_fwd）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, v, raw_g, beta, A_log, g_bias=None, scale=None,
                initial_state=None, output_final_state=False, cu_seqlens=None):
        """
        Args:
            q (Tensor): [B, T, H, K]
            k (Tensor): [B, T, H, K]
            v (Tensor): [B, T, H, V]
            raw_g (Tensor): [B, T, H*K] 原始门控投影
            beta (Tensor): [B, T, H]
            A_log (Tensor): [H] log 衰减
            g_bias (Tensor, optional): [H*K] 门控偏置
            scale (float, optional): 默认 K ** -0.5
            initial_state (Tensor, optional): [B, H, K, V]（varlen 时 [N, H, K, V]）
            output_final_state (bool)
            cu_seqlens (Tensor, optional): varlen 累计长度（约定 B == 1）

        Returns:
            o (Tensor): [B, T, H, V]，dtype 与 v 一致
            final_state (Tensor or None): [B, H, K, V]（varlen 为 [N, H, K, V]），fp32
        """
        B, T, H, K = q.shape
        V = v.shape[-1]
        if scale is None:
            scale = K ** -0.5

        # ---- 1. 融合门控（fp32）：gamma = -exp(A_log) * softplus(raw_g + g_bias) ----
        rg = raw_g.float()
        if rg.dim() == 3:          # [B, T, H*K] -> [B, T, H, K]
            rg = rg.reshape(B, T, H, K)
        if g_bias is not None:
            rg = rg + g_bias.float().reshape(H, K)
        a = -torch.exp(A_log.float().reshape(H))                    # [H]
        g = a.view(1, 1, H, 1) * F.softplus(rg)                     # [B, T, H, K]，<= 0

        # ---- 2. 转置为 [B, H, T, ...] 并升 fp32 ----
        qf = q.transpose(1, 2).float()                              # [B, H, T, K]
        kf = k.transpose(1, 2).float()
        vf = v.transpose(1, 2).float()
        gf = g.transpose(1, 2)
        bf = beta.transpose(1, 2).float()                           # [B, H, T]

        if cu_seqlens is None:
            o, S = self._chunk_all(qf, kf, vf, gf, bf, scale, T, K, V, initial_state)
            final_state = S if output_final_state else None
        else:
            cu = cu_seqlens.cpu().tolist()
            o = torch.zeros(B, H, T, V, device=q.device, dtype=torch.float32)
            final_states = []
            for i in range(len(cu) - 1):
                bos, eos = cu[i], cu[i + 1]
                st = initial_state[i].float() if initial_state is not None else None
                seg_o, S = self._chunk_all(
                    qf[:, :, bos:eos], kf[:, :, bos:eos], vf[:, :, bos:eos],
                    gf[:, :, bos:eos], bf[:, :, bos:eos], scale,
                    eos - bos, K, V, st)
                o[:, :, bos:eos] = seg_o
                final_states.append(S)
            final_state = torch.stack(final_states, dim=0) if output_final_state else None

        o = o.transpose(1, 2).contiguous().to(v.dtype)              # [B, T, H, V]
        return o, final_state

    def _chunk_all(self, qf, kf, vf, gf, bf, scale, T, K, V, initial_state):
        """整段序列分块递推，返回 o([B,H,T,V] fp32) 与末状态 S([B,H,K,V] fp32)。"""
        B, H = qf.shape[0], qf.shape[1]
        dev = qf.device
        C = _CHUNK
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T
        qp = F.pad(qf, (0, 0, 0, pad))
        kp = F.pad(kf, (0, 0, 0, pad))
        vp = F.pad(vf, (0, 0, 0, pad))
        gp = F.pad(gf, (0, 0, 0, pad))
        bp = F.pad(bf, (0, pad))

        S = torch.zeros(B, H, K, V, device=dev, dtype=torch.float32)
        if initial_state is not None:
            S = initial_state.float()

        o = torch.zeros(B, H, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32).expand(B, H, C, C).clone()
        M = Tp // C
        for c in range(M):
            qc = qp[:, :, c * C:(c + 1) * C]
            kc = kp[:, :, c * C:(c + 1) * C]
            vc = vp[:, :, c * C:(c + 1) * C]
            gc = gp[:, :, c * C:(c + 1) * C]
            bc = bp[:, :, c * C:(c + 1) * C]                       # [B,H,C]

            gk = gc.cumsum(2) / _LN2
            gk_last = gk[:, :, -1:, :]                              # 2^gk_last [B,H,1,K]

            # per-K 衰减矩阵 2^{clamp(gk_i - gk_j, 0)}（clamp 防极端负 gate 下
            # 2^{-gk} 上溢，kernel 的 GammaCompute 亦先 mask 后 exp）
            decay = torch.exp2(torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))

            # Aqk = tril(sum_k q_i k_j 2^{gk_i - gk_j}) · scale
            Aqk = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', qc, kc, decay),
                             diagonal=0) * scale
            # L = tril(sum_k k_i k_j 2^{gk_i - gk_j} · beta_i, 严格下三角)
            L = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', kc, kc, decay)
                           * bc.unsqueeze(-1), diagonal=-1)
            A_plus_L = eye + L

            w = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * kc * torch.exp2(gk),
                upper=False, unitriangular=True)
            u = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * vc, upper=False, unitriangular=True)

            v_new = u - w @ S
            qg = qc * torch.exp2(gk)
            o[:, :, c * C:(c + 1) * C] = torch.einsum('bhik,bhkv->bhiv', qg, S) * scale \
                + torch.einsum('bhij,bhjv->bhiv', Aqk, v_new)
            kg = kc * torch.exp2(gk_last - gk)
            S = torch.exp2(gk_last).transpose(-1, -2) * S \
                + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

        return o[:, :, :T], S


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "13_ChunkKdaWithFusedGateFwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-0.1, 0.1).item())
            sigma = float(torch.empty(1).uniform_(0.01, 0.05).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-0.1, 0.1)

    def beta_tensor(shape, dtype):
        t = torch.empty(shape, dtype=torch.float32).uniform_(0.01, 0.1)
        return t.to(dtype)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    tensor_names = ("q", "k", "v", "raw_g", "beta", "A_log", "g_bias", "initial_state")

    input_groups = []
    for case in cases:
        tensors = {}
        scale = None
        output_final_state = False
        cu_seqlens = None

        for inp in case["inputs"]:
            name = inp.get("name", "")
            if name in tensor_names:
                dtype = dtype_map[inp.get("dtype", "bfloat16")]
                if name == "beta":
                    tensors[name] = beta_tensor(inp["shape"], dtype)
                else:
                    tensors[name] = random_tensor(inp["shape"], dtype)
            elif name == "scale":
                scale = inp["value"]
            elif name == "output_final_state":
                output_final_state = bool(inp["value"])
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int64)

        input_groups.append([
            tensors["q"], tensors["k"], tensors["v"], tensors["raw_g"],
            tensors["beta"], tensors["A_log"], tensors.get("g_bias"),
            scale, tensors.get("initial_state"), output_final_state, cu_seqlens,
        ])
    return input_groups


def get_init_inputs():
    return []
