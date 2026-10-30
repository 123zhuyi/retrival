import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    GDN fused recurrent forward 的性能基线（golden for
    fused_recurrent_gated_delta_rule，fla/ops/gated_delta_rule/fused_recurrent.py）。

    语义对齐 fla-org `fused_recurrent_gated_delta_rule_fwd_kernel`（GDN 逐 token
    递推；该文件无分平台 backend dispatch，此 Triton kernel 即全平台唯一参考；
    别名 fused_recurrent_gdn）。逐 token（按 t 递增，S 为 [B,HV,K,V] fp32）：

        mode 1（标量门控）: S *= exp(g_t)                    # g [B,T,HV] 每 hv 标量
            若 use_gate_in_kernel（等价 USE_GATE_IN_KERNEL，要求 A_log 非 None）:
                g = -exp(A_log) * softplus(g + dt_bias)      # A_log/dt_bias [HV] 标量
                （use_bias <-> HAS_DT_BIAS；wrapper 在非 fuse 时强制 A_log=None）
        mode 2: S *= exp(gk_t) 沿 K 通道                     # gk [B,T,HV,K]
        mode 4: S *= exp(gv_t) 沿 V 通道                     # gv [B,T,HV,V]
        门控按 g→gk→gv 连乘，均在 residual 之前（先衰减后读出）
        β: use_beta_sigmoid 时 sigmoid(β)（allow_neg_eigval 时再 ×2）
        residual = (v − kᵀS)·β ；S += k⊗residual
        o_t = (q·scale)ᵀS  （读出用更新后状态；scale 预乘在 q 上）
    自然 e 底门控（与 chunk 路径的 log2 域不同）；o 回铸 v.dtype，
    final_state fp32；state_v_first 时状态出入 [.., V, K]（内部 [K,V] 转置）；
    GQA：i_h = i_hv // (HV//H)。

    use_qk_l2norm 时按 kernel 约定归一：x / sqrt(Σx² + 1e-6)（eps 在根号内；
    F.normalize(eps=1e-6) 是 x/max(‖x‖,eps)，极小范数输入才分叉——用例输入
    已预归一化 ‖x‖≈1，差异低于 bf16/fp16 ULP）。

    输入约定（与 kernel 指针布局一致，均为等长批量模式；kernel 的 cu_seqlens
    varlen / beta=None（默认全 1）/ headwise beta [B,T,HV,V] 不在用例范围）：
      q/k [B,T,H,K]（HV % H == 0）；v [B,T,HV,V]
      g [B,T,HV]（fuse 时原始 pre-activation，否则 log 域衰减）
      gk [B,T,HV,K]；gv [B,T,HV,V]（log 域衰减，exp(gate) ≤ 1）
      beta [B,T,HV]（use_beta_sigmoid 时 raw logits，否则已 sigmoid）
      A_log/dt_bias [HV] fp32；initial_state [B,HV,K,V]（state_v_first 时
      [B,HV,V,K]）fp32；scale 标量
    输出：o [B,T,HV,V]（v.dtype）；final_state [B,HV,K,V]（或 V-first）fp32。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _l2norm(xf):
        # kernel: x / sqrt(sum(x^2) + 1e-6)，eps 在根号内
        return xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)

    def forward(
        self,
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        A_log,
        dt_bias,
        initial_state,
        scale,
        gate_mode,
        use_gate_in_kernel,
        use_bias,
        use_initial_state,
        output_final_state,
        use_qk_l2norm,
        use_beta_sigmoid,
        allow_neg_eigval,
        state_v_first,
    ):
        batch, tokens, heads, key_dim = q.shape
        value_heads, value_dim = v.shape[2], v.shape[3]
        if value_heads % heads != 0:
            raise ValueError("value heads must be divisible by query heads")
        query = q.float()
        key = k.float()
        if use_qk_l2norm:
            query = self._l2norm(query)
            key = self._l2norm(key)
        query = query.repeat_interleave(value_heads // heads, dim=2)
        key = key.repeat_interleave(value_heads // heads, dim=2)

        scalar_gate = g.float()
        if gate_mode & 1 and use_gate_in_kernel:
            if use_bias:
                scalar_gate = scalar_gate + dt_bias.float()
            scalar_gate = -torch.exp(A_log.float()) * F.softplus(
                scalar_gate
            )
        key_gate = gk.float()
        value_gate = gv.float()
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
                batch,
                value_heads,
                key_dim,
                value_dim,
                device=q.device,
                dtype=torch.float32,
            )
        output = torch.empty_like(v)
        for token in range(tokens):
            if gate_mode & 1:
                state = state * torch.exp(
                    scalar_gate[:, token, :, None, None]
                )
            if gate_mode & 2:
                state = state * torch.exp(
                    key_gate[:, token, :, :, None]
                )
            if gate_mode & 4:
                state = state * torch.exp(
                    value_gate[:, token, :, None, :]
                )
            key_token = key[:, token]
            residual = v[:, token].float() - torch.einsum(
                "bhk,bhkv->bhv", key_token, state
            )
            residual = residual * beta_value[:, token, :, None]
            state = state + torch.einsum(
                "bhk,bhv->bhkv", key_token, residual
            )
            output[:, token] = (
                torch.einsum(
                    "bhk,bhkv->bhv", query[:, token] * scale, state
                ).to(v.dtype)
            )
        if state_v_first:
            state = state.transpose(-2, -1).contiguous()
        final_state = state if output_final_state else None
        return output, final_state


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_input_groups():
    def json_path():
        here = os.path.dirname(os.path.abspath(__file__))
        stem = os.path.splitext(os.path.basename(__file__))[0]
        names = [stem]
        if stem.endswith("_torch"):
            names.append(stem[: -len("_torch")])
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    with open(json_path(), "r", encoding="utf-8-sig") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def _random_tensor(shape, dtype, gen, scale=0.15):
        return (torch.randn(tuple(shape), generator=gen,
                            dtype=torch.float32) * scale).to(dtype)

    def _normalized_tensor(shape, dtype, gen):
        tensor = torch.randn(tuple(shape), generator=gen,
                             dtype=torch.float32)
        tensor = F.normalize(tensor, p=2.0, dim=-1, eps=1e-6)
        return tensor.to(dtype)

    def _decay_tensor(shape, dtype, gen, raw_gate):
        tensor = torch.randn(tuple(shape), generator=gen,
                             dtype=torch.float32)
        if raw_gate:
            tensor = tensor * 0.8
        else:
            tensor = -F.softplus(tensor) * 0.05
        return tensor.to(dtype)

    def _beta_tensor(shape, dtype, gen, raw_beta):
        tensor = torch.randn(tuple(shape), generator=gen,
                             dtype=torch.float32)
        if not raw_beta:
            tensor = torch.sigmoid(tensor)
        return tensor.to(dtype)

    groups = []
    for case_index, case in enumerate(cases):
        gen = torch.Generator().manual_seed(42 + case_index)
        specs = {item["name"]: item for item in case["inputs"]}
        gate_mode = specs["gate_mode"]["value"]
        fused_scalar = bool(gate_mode & 1) and \
            specs["use_gate_in_kernel"]["value"]
        raw_beta = specs["use_beta_sigmoid"]["value"]
        dtype = _DTYPE_MAP[specs["q"]["dtype"]]
        hvdtype = _DTYPE_MAP[specs["v"]["dtype"]]
        groups.append([
            _normalized_tensor(specs["q"]["shape"], dtype, gen),
            _normalized_tensor(specs["k"]["shape"], dtype, gen),
            _random_tensor(specs["v"]["shape"], hvdtype, gen, 0.2),
            _decay_tensor(specs["g"]["shape"], dtype, gen, fused_scalar),
            _decay_tensor(specs["gk"]["shape"], dtype, gen, False),
            _decay_tensor(specs["gv"]["shape"], dtype, gen, False),
            _beta_tensor(specs["beta"]["shape"], dtype, gen, raw_beta),
            _random_tensor(specs["A_log"]["shape"], torch.float32,
                           gen, 0.4),
            _random_tensor(specs["dt_bias"]["shape"], torch.float32,
                           gen, 0.3),
            _random_tensor(specs["initial_state"]["shape"],
                           torch.float32, gen, 0.04),
            specs["scale"]["value"],
            gate_mode,
            specs["use_gate_in_kernel"]["value"],
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
