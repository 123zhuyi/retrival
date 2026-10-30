import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CrossformerAttention —— 分组窗口 MHA + pos_bias（融合注意力）
#
# 借鉴 ops-ref `flash_attention_score` AscendC kernel 的融合手法：
#   attention_out = Softmax( Mask( scale·(Q·d_scale_q)(K·d_scale_k)^T + pse, atten_mask ) ) @ V
# 在 CrossFormer 分组窗口注意力（N 个 token 按 group_size 划组，组内全连接 MHA）中落地为：
#   1. scale 融合进 Q：kernel 的 d_scale_q 在 QK^T 的 cube 阶段乘到 query 上，
#      而非对整张 N_group×N_group 分数矩阵单独做一次 Mul —— 省一次中间 kernel。
#   2. pos_bias 加 scores：kernel 的 pse 在 QK^T 之后、softmax 之前加到分数矩阵
#      （pseType≠1 时先 mul 后 add，即 scale 已在 QK^T 中，pse 直接 add）。
#   3. mask 一次生成：kernel 的 atten_mask（1=不参与）在 softmax 前一次 masked_fill
#      -inf，等价原版「先加 mask 再 softmax」，但把 mask 生成/广播从逐头逐组循环
#      收敛为整批一次。
#   4. 整批分组：所有 (B × 组数) 的组窗口拼成一个 batch 维，一次 matmul+softmax+
#      matmul 完成全部组（kernel 按 batch/head 分核，单核内即三段式）。
#
# 与原版逐组循环数学等价（fp32 内部累乘）；仅把 scale 应用位置收敛到 Q 上、
# mask 收敛为一次生成。接口（shape/dtype/参数顺序）与原版完全一致。
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """CrossFormer Group Attention —— 融合版（纯函数，无训练参数）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor,
                group_size,
                q: torch.Tensor = None,
                k: torch.Tensor = None,
                v: torch.Tensor = None,
                num_heads: int = 1,
                mask: torch.Tensor = None,
                scale: float = None,
                pos_bias: torch.Tensor = None,
                feature_shape: tuple = None) -> torch.Tensor:
        """
        Args:
            x: [batch, N, channels]（feature_shape=None 时）或 [batch, H*W, channels]
            group_size: int 或 (gh, gw)
            q,k,v: 可选自定义 Q/K/V（形状同 x）
            num_heads: 头数
            mask: 可选 mask，广播到 (B, num_heads, N_group, N_group)（0=参与, -inf=不参与）
            scale: 缩放（默认 1/sqrt(head_dim)）
            pos_bias: 可选位置偏置，shape (N_group, N_group) 或 (num_heads, N_group, N_group)
            feature_shape: (H, W) 若 x 是展平特征图，否则 None
        Returns:
            out: [batch, N_original, channels]
        """
        B, N, C = x.shape

        if isinstance(group_size, int):
            gh = gw = group_size
        else:
            gh, gw = group_size

        if feature_shape is not None:
            H, W = feature_shape
            x_2d = x.view(B, H, W, C)
            num_h = H // gh
            num_w = W // gw
            x_groups = x_2d.view(B, num_h, gh, num_w, gw, C)
            x_groups = x_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
            x_groups = x_groups.view(B, num_h * num_w, gh * gw, C)
            B_flat = B * num_h * num_w
            x_flat = x_groups.view(B_flat, gh * gw, C)

            if q is not None:
                q_2d = q.view(B, H, W, C)
                q_groups = q_2d.view(B, num_h, gh, num_w, gw, C).permute(0, 1, 3, 2, 4, 5).contiguous()
                q_flat = q_groups.view(B_flat, gh * gw, C)
            else:
                q_flat = x_flat
            if k is not None:
                k_2d = k.view(B, H, W, C)
                k_groups = k_2d.view(B, num_h, gh, num_w, gw, C).permute(0, 1, 3, 2, 4, 5).contiguous()
                k_flat = k_groups.view(B_flat, gh * gw, C)
            else:
                k_flat = x_flat
            if v is not None:
                v_2d = v.view(B, H, W, C)
                v_groups = v_2d.view(B, num_h, gh, num_w, gw, C).permute(0, 1, 3, 2, 4, 5).contiguous()
                v_flat = v_groups.view(B_flat, gh * gw, C)
            else:
                v_flat = x_flat

            out_flat = self._attention(x_flat, q_flat, k_flat, v_flat,
                                       num_heads, mask, scale, pos_bias)
            out_groups = out_flat.view(B, num_h * num_w, gh * gw, C)
            out_groups = out_groups.view(B, num_h, num_w, gh, gw, C)
            out_groups = out_groups.permute(0, 1, 3, 2, 4, 5).contiguous()
            out = out_groups.view(B, H, W, C).view(B, H * W, C)
            return out

        # 单组模式
        q_in = x if q is None else q
        k_in = x if k is None else k
        v_in = x if v is None else v
        out = self._attention(x, q_in, k_in, v_in, num_heads, mask, scale, pos_bias)
        return out

    def _attention(self, x, q, k, v, num_heads, mask, scale, pos_bias):
        """融合注意力核心（一组内的 MHA）：scale 进 Q、pos_bias 加 scores、mask 一次生成。"""
        orig_dtype = x.dtype
        B, N, C = x.shape
        head_dim = C // num_heads
        scale = scale if scale is not None else (head_dim ** -0.5)

        # ---- 分头 + 升 fp32（kernel 内部 upcast 累乘，与原版一致） ----
        def to_heads(t):
            return t.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).float()

        qh = to_heads(q) * scale                 # scale 融合进 Q（d_scale_q 等价）
        kh = to_heads(k)
        vh = to_heads(v)

        scores = torch.matmul(qh, kh.transpose(-2, -1))   # [B, H, N, N]

        # ---- pos_bias 加 scores（kernel 的 pse：先 mul 后 add） ----
        if pos_bias is not None:
            if pos_bias.dim() == 2:
                scores = scores + pos_bias.float().unsqueeze(0).unsqueeze(0)
            else:
                scores = scores + pos_bias.float().unsqueeze(0)

        # ---- mask 施加（kernel 的 atten_mask；输入为 0=参与 / -inf=不参与 的加法掩码，
        #      一次广播到 [B,H,N,N] 加到 scores，softmax 后即置 0） ----
        if mask is not None:
            scores = scores + mask.float()

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, vh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return out.to(orig_dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_CrossformerAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            if dtype is torch.bfloat16:
                return torch.normal(mu, sigma, shape, dtype=torch.float32).to(dtype)
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]

        args = {}
        for inp in inputs:
            n = inp["name"]
            if inp.get("type") == "tensor":
                args[n] = random_tensor(inp["shape"], dtype_map[inp["dtype"]])
            else:
                args[n] = inp["value"]

        x = args.pop("x")
        group_size = args.pop("group_size")
        q = args.pop("q", None)
        k = args.pop("k", None)
        v = args.pop("v", None)
        num_heads = args.pop("num_heads", 1)
        mask = args.pop("mask", None)
        scale = args.pop("scale", None)
        pos_bias = args.pop("pos_bias", None)
        feature_shape = args.pop("feature_shape", None)

        # mask 生成：加法掩码，0=参与 / -inf=不参与（与原版 attn + mask 语义兼容）。
        # 保证对角线始终 0（每个 query 至少能看到自己），避免整行 -inf 导致 softmax NaN。
        if mask is not None:
            mf = mask.float()
            N_g = mf.shape[-1]
            blk = (torch.rand(mf.shape) < 0.15)
            if mf.dim() == 2:
                blk = blk & ~torch.eye(N_g, dtype=torch.bool)
            else:
                eye = torch.eye(N_g, dtype=torch.bool).unsqueeze(0)
                blk = blk & ~eye.expand(mf.shape[0], N_g, N_g)
            mask = torch.where(blk, torch.full_like(mf, float('-inf')),
                               torch.zeros_like(mf)).to(mask.dtype)

        input_groups.append([x, group_size, q, k, v, num_heads, mask, scale,
                             pos_bias, feature_shape])
    return input_groups


def get_init_inputs():
    return []
