import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os


# ---------------------------------------------------------------------------
# OlmoHybridGdnFullForward —— OLMo 混合 Gated Delta Net（GDN）全前向，融合版
#
# 借鉴 ops-ref `fused_recurrent_kda` / `chunk_gated_delta_rule` / `causal_conv1d`
# 的融合算法，把原版逐 token 循环改写为「分块矩阵 + 块间状态传播」：
#   (1) 因果 depthwise conv1d（F.conv1d, groups=conv_dim, 左 pad）+ SiLU；
#   (2) GQA 扩展（num_heads_k -> num_heads_v）；
#   (3) 融合门控（fp32）：g = -exp(A_log)·softplus(a + dt_bias)（恒 <= 0），
#       beta = sigmoid(b)·2；
#   (4) 门控 delta rule 递推（fp32），分块版（C = 64）：
#         gk = cumsum(g/ln2)（每头标量）;  qg = qn·2^gk
#         Aqk = tril(Σ_K qn_i·kn_j·2^{gk_i-gk_j})
#         L   = tril(beta_i·Σ_K kn_i·kn_j·2^{gk_i-gk_j})
#         w = (I+L)^{-1}(beta·kn·2^gk), u = (I+L)^{-1}(beta·v)
#         v_new = u - w·S;  o_c = qg·S + Aqk·v_new
#         S' = 2^{gk_last}·S + kg^T·v_new（kg = kn·2^{gk_last-gk}）
#   (5) per-head 门控 RMSNorm（eps=1e-5, gate clamp [-10,10]）+ SiLU；
#   (6) 输出投影 F.linear（fp32 计算，写回 output_buffer）。
#
# 与原版逐 token 递推数学等价；内部 fp32，输出回 input dtype。
#
# 最小修复版改动（不触碰任何精度路径）：
#   1. kg 指数修复为 log2 域相减（已在本版中，见 _chunk_all 内注释）；
#   2. 删除死代码 kgq（从未使用，且强衰减下 exp2(-gk) 自身上溢产生 inf）。
# ---------------------------------------------------------------------------

_LN2 = math.log(2.0)
_CHUNK = 64


def rms_norm(x, normalized_shape, weight=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    variance = x.pow(2).mean(dims, keepdim=True)
    out = x * torch.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight
    return out


def silu(x):
    return x * torch.sigmoid(x)


class Model(nn.Module):
    """
    OLMo Hybrid Gated Delta Net (GDN) Full Forward（分块融合版）。
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, hidden_states, output_buffer, projected_qkvg, proj_b, proj_a, w_conv1d,
                dt_bias, A_log, w_o_norm, w_o_proj, num_heads_k, num_heads_v,
                head_k_dim, head_v_dim, conv_kernel_size):
        num_tokens = hidden_states.shape[0]
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Clamp decay parameters to avoid fp16 overflow/NaN in exp/softplus
        A_log = A_log.clamp(-3.0, 3.0)
        dt_bias = dt_bias.clamp(-3.0, 3.0)

        key_dim = num_heads_k * head_k_dim
        value_dim = num_heads_v * head_v_dim

        # (1)(2) projections precomputed by get_input_groups
        mixed_qkv = projected_qkvg[:, :key_dim * 2 + value_dim]
        gate = projected_qkvg[:, key_dim * 2 + value_dim:]
        b = proj_b
        a = proj_a

        # (3) Causal depthwise conv1d + SiLU
        conv_dim = key_dim * 2 + value_dim
        mixed_qkv_t = mixed_qkv.unsqueeze(0).transpose(1, 2)
        conv_w = w_conv1d.view(conv_dim, 1, conv_kernel_size)
        pad_size = conv_kernel_size - 1
        mixed_qkv_padded = F.pad(mixed_qkv_t, (pad_size, 0))
        mixed_qkv_conv = F.conv1d(mixed_qkv_padded, conv_w, groups=conv_dim)
        mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1)
        mixed_qkv_conv = silu(mixed_qkv_conv)

        query = mixed_qkv_conv[:, :key_dim].view(num_tokens, num_heads_k, head_k_dim)
        key = mixed_qkv_conv[:, key_dim:2 * key_dim].view(num_tokens, num_heads_k, head_k_dim)
        value = mixed_qkv_conv[:, 2 * key_dim:].view(num_tokens, num_heads_v, head_v_dim)

        # (4) GQA expansion of Q and K
        if num_heads_v > num_heads_k:
            expand_ratio = num_heads_v // num_heads_k
            query = query.unsqueeze(2).expand(-1, -1, expand_ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)
            key = key.unsqueeze(2).expand(-1, -1, expand_ratio, -1).reshape(num_tokens, num_heads_v, head_k_dim)

        # (5) GDN gate（fp32）
        g = -A_log.float().exp().unsqueeze(0) * F.softplus(a.float() + dt_bias.unsqueeze(0))
        beta = torch.sigmoid(b.float()) * 2.0

        q_f = query.float().unsqueeze(0)            # [1, T, Hv, K]
        k_f = key.float().unsqueeze(0)
        v_f = value.float().unsqueeze(0)            # [1, T, Hv, V]
        g_f = g.float().unsqueeze(0)                # [1, T, Hv]
        b_f = beta.float().unsqueeze(0)             # [1, T, Hv]
        scale = 1.0 / math.sqrt(head_k_dim)

        # L2 归一化（与原版 +1e-6 一致），scale 折进 q
        k_f = k_f / (torch.norm(k_f, dim=-1, keepdim=True) + 1e-6)
        q_f = q_f / (torch.norm(q_f, dim=-1, keepdim=True) + 1e-6) * scale

        # (6) Gated delta rule recurrence（分块融合）
        core_attn_out, _ = self._chunk_all(q_f, k_f, v_f, g_f, b_f)
        core_attn_out = core_attn_out[0].to(dtype)  # [T, Hv, V]

        # (7) Per-head gated RMSNorm with SiLU(gate)
        gate = gate.view(num_tokens, num_heads_v, head_v_dim).clamp(-10.0, 10.0)
        core_attn_out_flat = core_attn_out.reshape(-1, head_v_dim)
        gate_flat = gate.reshape(-1, head_v_dim)
        normed = rms_norm(core_attn_out_flat, (head_v_dim,), weight=w_o_norm)
        core_attn_out_normed = normed * gate_flat * torch.sigmoid(gate_flat)

        # (8) Output projection in fp32（docstring: "Output projection in fp32,
        # cast back to input dtype"；w_o_proj 提升为 fp32 避免与 bf16 输入混乘）
        core_attn_out_reshaped = core_attn_out_normed.view(num_tokens, value_dim)
        output = F.linear(core_attn_out_reshaped.float(), w_o_proj.float())
        output_buffer[:num_tokens] = output
        return output_buffer

    def _chunk_all(self, qf, kf, vf, gf, bf):
        """整段序列分块递推（每头标量门控）。
        qf/kf [B,T,H,K]; vf [B,T,H,V]; gf/bf [B,T,H]（每头标量 log 衰减 / beta）。
        返回 (o([B,T,H,V] fp32), S([B,H,K,V] fp32))。"""
        B, T, H, K = qf.shape
        V = vf.shape[-1]
        dev = qf.device
        C = _CHUNK
        Tp = ((T + C - 1) // C) * C
        pad = Tp - T
        # [B,T,H,...] -> [B,H,T,...] 布局；pad 空 token：k/v/beta/g=0
        qp = F.pad(qf.transpose(1, 2), (0, 0, 0, pad))
        kp = F.pad(kf.transpose(1, 2), (0, 0, 0, pad))
        vp = F.pad(vf.transpose(1, 2), (0, 0, 0, pad))
        gp = F.pad(gf.transpose(1, 2), (0, pad))     # [B,H,Tp]
        bp = F.pad(bf.transpose(1, 2), (0, pad))

        S = torch.zeros(B, H, K, V, device=dev, dtype=torch.float32)
        o = torch.zeros(B, H, Tp, V, device=dev, dtype=torch.float32)
        eye = torch.eye(C, device=dev, dtype=torch.float32).expand(B, H, C, C).clone()
        M = Tp // C
        for c in range(M):
            qc = qp[:, :, c * C:(c + 1) * C]         # [B,H,C,K]
            kc = kp[:, :, c * C:(c + 1) * C]
            vc = vp[:, :, c * C:(c + 1) * C]         # [B,H,C,V]
            gc = gp[:, :, c * C:(c + 1) * C]         # [B,H,C]
            bc = bp[:, :, c * C:(c + 1) * C]         # [B,H,C]

            gk = gc.cumsum(2) / _LN2                 # [B,H,C] 标量 log2 累计
            gl = torch.exp2(gk[:, :, -1:])           # [B,H,1]
            decay = torch.exp2(torch.clamp(
                gk.unsqueeze(3) - gk.unsqueeze(2), max=0.0))  # [B,H,C,C]

            qg = qc * torch.exp2(gk).unsqueeze(-1)   # [B,H,C,K]
            # 【修改】删除死代码 kgq = kc * torch.exp2(-gk)（从未使用，且强衰减下
            #   exp2(-gk) 自身上溢产生 inf；删除不影响任何输出）。
            kg = kc * torch.exp2(gk[:, :, -1:] - gk).unsqueeze(-1)  # log2 域相减（修复：原 gl-gk 为线性域/对数域混用）

            Aqk = torch.tril(torch.einsum(
                'bhik,bhjk,bhijk->bhij', qc, kc, decay.unsqueeze(-1)), diagonal=0)
            L = torch.tril(torch.einsum(
                'bhik,bhjk,bhijk->bhij', kc, kc, decay.unsqueeze(-1))
                * bc.unsqueeze(-1), diagonal=-1)
            A_plus_L = eye + L

            w = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * kc * torch.exp2(gk).unsqueeze(-1),
                upper=False, unitriangular=True)
            u = torch.linalg.solve_triangular(
                A_plus_L, bc.unsqueeze(-1) * vc, upper=False, unitriangular=True)

            v_new = u - torch.einsum('bhik,bhkv->bhiv', w, S)
            o[:, :, c * C:(c + 1) * C] = torch.einsum('bhik,bhkv->bhiv', qg, S) \
                + torch.einsum('bhij,bhjv->bhiv', Aqk, v_new)
            S = gl.unsqueeze(-1) * S \
                + torch.einsum('bhik,bhiv->bhkv', kg, v_new)

        return o[:, :, :T].transpose(1, 2).contiguous(), S


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "17_OlmoHybridGdnFullForward.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype_):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype_ is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype_)
            return torch.normal(mu, sigma, shape, dtype=dtype_)
        else:
            return torch.empty(shape, dtype=dtype_).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

        def tensor_input(name):
            info = next(inp for inp in inputs if inp["name"] == name)
            return random_tensor(info["shape"], dtype_map[info["dtype"]])

        def attr_input(name):
            return next(inp for inp in inputs if inp["name"] == name)["value"]

        hs_info = next(inp for inp in inputs if inp["name"] == "hidden_states")
        ob_info = next(inp for inp in inputs if inp["name"] == "output_buffer")
        dtype = dtype_map[hs_info["dtype"]]
        hidden_states = tensor_input("hidden_states")
        output_buffer = torch.empty(ob_info["shape"], dtype=dtype)

        w_qkvg = tensor_input("w_qkvg")
        w_b = tensor_input("w_b")
        w_a = tensor_input("w_a")
        with torch.no_grad():
            projected_qkvg = F.linear(hidden_states, w_qkvg)
            proj_b = F.linear(hidden_states, w_b)
            proj_a = F.linear(hidden_states, w_a)

        input_groups.append([
            hidden_states,
            output_buffer,
            projected_qkvg,
            proj_b,
            proj_a,
            tensor_input("w_conv1d"),
            tensor_input("dt_bias"),
            tensor_input("A_log"),
            tensor_input("w_o_norm"),
            tensor_input("w_o_proj"),
            attr_input("num_heads_k"),
            attr_input("num_heads_v"),
            attr_input("head_k_dim"),
            attr_input("head_v_dim"),
            attr_input("conv_kernel_size"),
        ])
    return input_groups


def get_init_inputs():
    return []
