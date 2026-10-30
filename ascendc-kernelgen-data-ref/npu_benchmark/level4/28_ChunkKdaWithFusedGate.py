import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ChunkKdaWithFusedGate —— 融合门控 + 分块 KDA（含 q/k l2norm、safe_gate 等选项）
#
# 借鉴 ops-ref `chunk_kda_fwd`（fused-gate 变体）的分块流水线算法：
#   1. 融合门控（kda_gate_fwd_kernel，fp32）：
#        gamma = lower_bound * sigmoid(exp(A_log) * (raw_g + g_bias))   # safe_gate
#        gamma = -exp(A_log) * softplus(raw_g + g_bias)                 # 默认
#   2. 可选 q/k l2norm（kernel 的 use_qk_l2norm_in_kernel）。
#   3. chunk 分块递推（块内矩阵 + 块间状态传播）：
#        gk = cumsum(gamma / ln2)
#        qg = q·2^gk,  kgq = k·2^{-gk},  kg = k·2^{gk_last - gk}
#        Aqk = tril(qg @ kgq^T) · scale
#        L   = tril((k·2^{gk}) @ kgq^T · beta, 下三角)
#        Akk = (I + L)^{-1}
#        w = Akk·(beta·k·2^{gk}),  u = Akk·(beta·v)
#        v_new = u - w·h_c
#        o_c   = scale·qg·h_c + Aqk·v_new
#        h_{c+1} = 2^{gk_last} ⊙ h_c + kg^T·v_new
#
# 与逐 token 递推数学等价，把 O(T) 次顺序小步压缩为 O(T/C) 次向量化矩阵块。
# 全程 fp32，输出 o 转回 v.dtype，final_state 保持 fp32。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)
_CHUNK = 64


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        raw_g,
        beta,
        A_log,
        g_bias,
        initial_state,
        scale,
        lower_bound,
        use_bias,
        use_lower_bound,
        use_initial_state,
        output_final_state,
        use_qk_l2norm,
    ):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by query heads")
        query = q.float()
        key = k.float()
        if use_qk_l2norm:
            query = F.normalize(query, p=2.0, dim=-1, eps=1e-6)
            key = F.normalize(key, p=2.0, dim=-1, eps=1e-6)
        query = query.repeat_interleave(value_heads // heads, dim=2)
        key = key.repeat_interleave(value_heads // heads, dim=2)

        # ---- 1. 融合门控（fp32） ----
        gate_input = raw_g.float()
        if use_bias:
            gate_input = gate_input + g_bias.float().reshape(value_heads, key_dim)
        decay_rate = torch.exp(A_log.float())[:, None]              # [HV, 1]
        if use_lower_bound:
            gate = lower_bound * torch.sigmoid(decay_rate * gate_input)
        else:
            gate = -decay_rate * F.softplus(gate_input)             # [B,T,HV,K]，<= 0

        if use_initial_state:
            state = initial_state.float().clone()
        else:
            state = torch.zeros(
                batch, value_heads, key_dim, value_dim,
                device=q.device, dtype=torch.float32,
            )

        # ---- 2. 分块 KDA ----
        o, final_state = self._chunk_kda(
            query, key, v, gate, beta, scale, state,
        )

        output = o.to(v.dtype)
        final_state = final_state if output_final_state else None
        return output, final_state

    def _chunk_kda(self, query, key, v, gate, beta, scale, state):
        """分块递推，返回 o([B,T,HV,V] fp32) 与末状态 S([B,HV,K,V] fp32)。"""
        B, T, HV, K = query.shape
        V = v.shape[-1]
        dev = query.device
        C = _CHUNK
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T

        qf = query.transpose(1, 2).float()                          # [B,HV,T,K]
        kf = key.transpose(1, 2).float()
        vf = v.transpose(1, 2).float()
        gf = gate.transpose(1, 2).float()
        bf = beta.transpose(1, 2).float()                           # [B,HV,T]

        qp = F.pad(qf, (0, 0, 0, pad))
        kp = F.pad(kf, (0, 0, 0, pad))
        vp = F.pad(vf, (0, 0, 0, pad))
        gp = F.pad(gf, (0, 0, 0, pad))
        bp = F.pad(bf, (0, pad))

        S = state.float().contiguous()                              # [B,HV,K,V]
        o = torch.zeros(B, HV, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32).expand(B, HV, C, C).clone()
        M = Tp // C
        for c in range(M):
            qc = qp[:, :, c * C:(c + 1) * C]
            kc = kp[:, :, c * C:(c + 1) * C]
            vc = vp[:, :, c * C:(c + 1) * C]
            gc = gp[:, :, c * C:(c + 1) * C]
            bc = bp[:, :, c * C:(c + 1) * C]                        # [B,HV,C]

            gk = gc.cumsum(2) / _LN2
            gk_last = gk[:, :, -1:, :]

            # per-K 衰减矩阵（clamp 防极端负 gate 下 2^{-gk} 上溢）
            decay = torch.exp2(torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))
            Aqk = torch.tril(torch.einsum('bhik,bhjk,bhijk->bhij', qc, kc, decay),
                             diagonal=0) * scale
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

        return o.transpose(1, 2).contiguous()[:, :T], S


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.15):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * scale
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _normalized_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    tensor = F.normalize(tensor, p=2.0, dim=-1, eps=1e-6)
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _beta_tensor(spec, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    logits = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    return torch.sigmoid(logits).to(
        dtype=_DTYPE_MAP[spec["dtype"]]
    ).npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        groups.append([
            _normalized_tensor(specs["q"], 42 + case_index * 8),
            _normalized_tensor(specs["k"], 43 + case_index * 8),
            _random_tensor(specs["v"], 44 + case_index * 8, 0.2),
            _random_tensor(specs["raw_g"], 45 + case_index * 8, 0.8),
            _beta_tensor(specs["beta"], 46 + case_index * 8),
            _random_tensor(specs["A_log"], 47 + case_index * 8, 0.4),
            _random_tensor(specs["g_bias"], 48 + case_index * 8, 0.3),
            _random_tensor(
                specs["initial_state"], 49 + case_index * 8, 0.04
            ),
            specs["scale"]["value"],
            specs["lower_bound"]["value"],
            specs["use_bias"]["value"],
            specs["use_lower_bound"]["value"],
            specs["use_initial_state"]["value"],
            specs["output_final_state"]["value"],
            specs["use_qk_l2norm"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
