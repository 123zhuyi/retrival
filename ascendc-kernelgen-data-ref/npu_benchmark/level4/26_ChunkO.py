import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ChunkO —— chunk 级注意力输出（inter + intra）。

    借鉴 ops-ref `chunk_kda_fwd` 的「输出段」公式：
        o_c = scale * (qg_c @ h_c + Aqk_c @ v_c)
    其中 qg = q·2^gate（gate 缩放 query），
        Aqk = tril(qg @ kgq^T)（块内 QK 系数矩阵，含对角线），
        h_c 为块输入状态（由上一层给出），v 为当前块 value。

    原版对每个 chunk 用 Python 循环做 matmul + tril；本版本把整条序列
    补零到 chunk_size 整数倍后 reshape 为 [B, M, C, ...]，所有 chunk 的
    qg / Aqk / inter / intra 一次向量化完成：

      - gate_mode=1：gate = g（chunk 内累计 gate），qg = q·2^gate，
        Aqk = tril(qg @ kgq^T)，kgq = k·2^{-gate}。
      - gate_mode=2：gate = g_gamma · position（线性衰减），结构同上。
      - 输出 = scale * (qg @ h + Aqk @ v)，与 kernel 输出公式一致。

    补零 token 的 q/k/v 为 0，对 Aqk 与输出贡献为 0，结果与原版逐块循环等价。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q,
        k,
        v,
        h,
        g,
        g_gamma,
        scale,
        chunk_size,
        gate_mode,
        state_v_first,
    ):
        torch.manual_seed(42)
        batch, tokens, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by query heads")
        group_size = value_heads // heads

        query = q.repeat_interleave(group_size, dim=2).float()   # [B,T,HV,K]
        key = k.repeat_interleave(group_size, dim=2).float()
        value = v.float()                                        # [B,T,HV,V]

        C = chunk_size
        M = (tokens + C - 1) // C
        Tp = M * C
        pad = Tp - tokens

        # 补零并分块
        def _pad4(x):
            return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))   # pad T dim
        qp = _pad4(query).view(batch, M, C, value_heads, key_dim)
        kp = _pad4(key).view(batch, M, C, value_heads, key_dim)
        vp = _pad4(value).view(batch, M, C, value_heads, value_dim)

        state = h.float()                                          # [B,M,HV,K,V] 或 [B,M,HV,V,K]
        if state_v_first:
            state = state.transpose(-1, -2)                        # -> [B,M,HV,K,V]

        # ---- gate ----
        if gate_mode == 1:
            gpad = torch.nn.functional.pad(g.float(), (0, 0, 0, pad))
            gate = gpad.view(batch, M, C, value_heads)             # [B,M,C,HV]
        elif gate_mode == 2:
            positions = torch.arange(1, C + 1, device=q.device, dtype=torch.float32)
            gate = (g_gamma.float()[None, :, None]
                    * positions[None, None, :]).expand(batch, M, value_heads, C)
            gate = gate.permute(0, 1, 3, 2)                        # [B,M,C,HV]
        elif gate_mode != 0:
            raise ValueError("gate_mode must be 0, 1, or 2")
        else:
            gate = None

        # ---- 块内矩阵：qg / Aqk ----
        if gate is not None:
            g2 = torch.exp2(gate[..., None])                       # [B,M,C,HV,1]
            qg = qp * g2                                           # q·2^gate
            kgq = kp * torch.exp2(-gate[..., None])                # k·2^{-gate}
        else:
            qg = qp
            kgq = kp

        Aqk = torch.tril(torch.einsum('bmihk,bmjhk->bmhij', qg, kgq), diagonal=0)
        # [B,M,HV,C,C]

        # ---- inter + intra ----
        inter = torch.einsum('bmchk,bmhkv->bmchv', qg, state)      # qg @ h
        intra = torch.einsum('bmhij,bmjhv->bmihv', Aqk, vp)        # Aqk @ v

        result = (inter + intra) * scale                            # [B,M,C,HV,V]
        out = result.reshape(batch, Tp, value_heads, value_dim)[:, :tokens]
        return out.to(v.dtype)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(spec, seed, scale=0.12):
    generator = torch.Generator()
    generator.manual_seed(seed)
    tensor = torch.randn(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * scale
    return tensor.to(dtype=_DTYPE_MAP[spec["dtype"]]).npu()


def _cumulative_gate(spec, seed, chunk_size):
    generator = torch.Generator()
    generator.manual_seed(seed)
    increments = -torch.rand(
        tuple(spec["shape"]), generator=generator, dtype=torch.float32
    ) * 0.06
    output = torch.empty_like(increments)
    tokens = increments.shape[1]
    for start in range(0, tokens, chunk_size):
        end = min(tokens, start + chunk_size)
        output[:, start:end] = increments[:, start:end].cumsum(dim=1)
    return output.npu()


def _load_cases():
    path = os.path.splitext(__file__)[0] + ".json"
    with open(path, "r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_input_groups():
    torch.manual_seed(42)
    groups = []
    for case_index, case in enumerate(_load_cases()):
        specs = {item["name"]: item for item in case["inputs"]}
        chunk_size = specs["chunk_size"]["value"]
        groups.append([
            _random_tensor(specs["q"], 42 + case_index * 6, 0.1),
            _random_tensor(specs["k"], 43 + case_index * 6, 0.1),
            _random_tensor(specs["v"], 44 + case_index * 6, 0.15),
            _random_tensor(specs["h"], 45 + case_index * 6, 0.04),
            _cumulative_gate(specs["g"], 46 + case_index * 6, chunk_size),
            _random_tensor(specs["g_gamma"], 47 + case_index * 6, 0.03),
            specs["scale"]["value"],
            chunk_size,
            specs["gate_mode"]["value"],
            specs["state_v_first"]["value"],
        ])
    return groups


def get_init_inputs():
    return []
