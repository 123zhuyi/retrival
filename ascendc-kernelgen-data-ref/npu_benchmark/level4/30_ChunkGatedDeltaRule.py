import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ChunkGatedDeltaRule —— chunk 版 GDR 前向（分块递推，C = chunk_size）
#
# 借鉴 ops-ref `chunk_gated_delta_rule` 的分块流水线算法：
#   - kernel 内门控（use_gate_in_kernel）：
#         gamma = -exp(A_log) * softplus(g + dt_bias)
#   - beta 可选 sigmoid（allow_neg_eigval 时 ×2）；q/k 可选 l2norm。
#   - 每个 chunk 内一次向量化矩阵运算，chunk 间做状态传播：
#        gk = cumsum(gamma/ln2)
#        Aqk = tril(sum_k q_i k_j 2^{gk_i-gk_j}) · scale
#        L   = tril(sum_k k_i k_j 2^{gk_i-gk_j} · beta_i, 下三角)
#        w = (I+L)^{-1}(beta·k·2^{gk})，u = (I+L)^{-1}(beta·v)
#        v_new = u - w·h
#        o_c   = scale·(q·2^{gk})·h + Aqk·v_new
#        h_{c+1} = 2^{gk_last}⊙h + (k·2^{gk_last-gk})^T·v_new
#
# 与逐 token 递推数学等价，把 O(T) 次顺序小步压缩为 O(T/C) 次向量化矩阵块。
# 全程 fp32，输出 o 转回 v.dtype；state_v_first 时状态以 [HV, V, K] 交换布局。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        scale,
        chunk_size,
        use_gate_in_kernel,
        use_bias,
        use_initial_state,
        output_final_state,
        use_qk_l2norm,
        use_beta_sigmoid,
        allow_neg_eigval,
        state_v_first,
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

        # ---- 门控（kernel 内 or 直通） ----
        if use_gate_in_kernel:
            gate_input = g.float()
            if use_bias:
                gate_input = gate_input + dt_bias.float()
            gate = -torch.exp(A_log.float()) * F.softplus(gate_input)
        else:
            gate = g.float()

        # ---- beta 激活 ----
        beta_value = beta.float()
        if use_beta_sigmoid:
            beta_value = torch.sigmoid(beta_value)
            if allow_neg_eigval:
                beta_value = beta_value * 2.0

        if use_initial_state:
            state = initial_state.float().clone()
            if state_v_first:
                state = state.transpose(-2, -1).contiguous()
        else:
            state = torch.zeros(
                batch, value_heads, key_dim, value_dim,
                device=q.device, dtype=torch.float32,
            )

        output = self._chunk_kda(
            query, key, v, gate, beta_value, scale, state, chunk_size)

        if state_v_first:
            state = state.transpose(-2, -1).contiguous()
        final_state = state if output_final_state else None
        return output.to(v.dtype), final_state

    def _chunk_kda(self, query, key, v, gate, beta, scale, state, C):
        """分块递推，返回 o([B,T,HV,V] fp32)；原地更新 state([B,HV,K,V] fp32)。"""
        B, T, HV, K = query.shape
        V = v.shape[-1]
        dev = query.device
        C = int(C)
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T

        qf = query.transpose(1, 2).float()
        kf = key.transpose(1, 2).float()
        vf = v.transpose(1, 2).float()
        gf = gate.transpose(1, 2).float()                           # [B,HV,T] 或 [B,HV,T,K]
        per_k = gf.dim() == 4
        bf = beta.transpose(1, 2).float()                           # [B,HV,T]

        qp = F.pad(qf, (0, 0, 0, pad))
        kp = F.pad(kf, (0, 0, 0, pad))
        vp = F.pad(vf, (0, 0, 0, pad))
        gp = F.pad(gf, (0, pad)) if gf.dim() == 3 else F.pad(gf, (0, 0, 0, pad))
        bp = F.pad(bf, (0, pad))

        S = state.float().contiguous()
        o = torch.zeros(B, HV, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32).expand(B, HV, C, C).clone()
        M = Tp // C
        for c in range(M):
            qc = qp[:, :, c * C:(c + 1) * C]
            kc = kp[:, :, c * C:(c + 1) * C]
            vc = vp[:, :, c * C:(c + 1) * C]
            gc = gp[:, :, c * C:(c + 1) * C]
            bc = bp[:, :, c * C:(c + 1) * C]

            gk = gc.cumsum(2) / _LN2
            if per_k:
                gk_last = gk[:, :, -1:, :]
                decay = torch.exp2(
                    torch.clamp(gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))   # [B,HV,C,C,K]
                Aqk = torch.tril(
                    torch.einsum('bhik,bhjk,bhijk->bhij', qc, kc, decay), diagonal=0) * scale
                L = torch.tril(
                    torch.einsum('bhik,bhjk,bhijk->bhij', kc, kc, decay)
                    * bc.unsqueeze(-1), diagonal=-1)
                gl = torch.exp2(gk_last).transpose(-1, -2)                      # [B,HV,K,1]
            else:
                gk_last = gk[:, :, -1:]                                         # [B,HV,1]
                decay = torch.exp2(
                    torch.clamp(gk.unsqueeze(-1) - gk.unsqueeze(2), max=0.0))   # [B,HV,C,C]
                Aqk = torch.tril(
                    torch.einsum('bhik,bhjk,bhij->bhij', qc, kc, decay), diagonal=0) * scale
                L = torch.tril(
                    torch.einsum('bhik,bhjk,bhij->bhij', kc, kc, decay)
                    * bc.unsqueeze(-1), diagonal=-1)
                gl = torch.exp2(gk_last).unsqueeze(-1)                          # [B,HV,1,1]

            A_plus_L = eye + L
            if per_k:
                gk4 = torch.exp2(gk)
            else:
                gk4 = torch.exp2(gk).unsqueeze(-1)                  # [B,HV,C,1]
            w = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * kc * gk4,
                upper=False, unitriangular=True)
            u = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * vc, upper=False, unitriangular=True)

            v_new = u - w @ S
            qg = qc * gk4
            o[:, :, c * C:(c + 1) * C] = torch.einsum('bhik,bhkv->bhiv', qg, S) * scale \
                + torch.einsum('bhij,bhjv->bhiv', Aqk, v_new)
            kg = kc * torch.exp2(gk_last - gk) if per_k else \
                kc * torch.exp2(gk_last - gk).unsqueeze(-1)
            S = gl * S + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

        state.copy_(S)
        return o.transpose(1, 2).contiguous()[:, :T]


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


def _gate_tensor(spec, seed, raw_gate):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    if raw_gate:
        tensor = tensor * 0.8
    else:
        tensor = -F.softplus(tensor) * 0.08
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _beta_tensor(spec, seed, raw_beta):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    )
    if not raw_beta:
        tensor = torch.sigmoid(tensor)
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        raw_gate = specs["use_gate_in_kernel"]["value"]
        raw_beta = specs["use_beta_sigmoid"]["value"]
        groups.append([
            _normalized_tensor(specs["q"], 42 + case_index * 8),
            _normalized_tensor(specs["k"], 43 + case_index * 8),
            _random_tensor(specs["v"], 44 + case_index * 8, 0.2),
            _gate_tensor(specs["g"], 45 + case_index * 8, raw_gate),
            _beta_tensor(specs["beta"], 46 + case_index * 8, raw_beta),
            _random_tensor(specs["A_log"], 47 + case_index * 8, 0.4),
            _random_tensor(specs["dt_bias"], 48 + case_index * 8, 0.3),
            _random_tensor(
                specs["initial_state"], 49 + case_index * 8, 0.04
            ),
            specs["scale"]["value"],
            specs["chunk_size"]["value"],
            raw_gate,
            specs["use_bias"]["value"],
            specs["use_initial_state"]["value"],
            specs["output_final_state"]["value"],
            specs["use_qk_l2norm"]["value"],
            raw_beta,
            specs["allow_neg_eigval"]["value"],
            specs["state_v_first"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
