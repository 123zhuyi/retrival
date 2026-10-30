import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Chunk-scaled dot-product KKT forward：计算 beta * K * K^T（含门控衰减）。

    借鉴 ops-ref `chunk_kda_fwd` 的「chunk 内 A 矩阵」计算算法：
    kernel 对每条序列按 C=chunk_size 分块，块内一次性向量化计算
        A_chunk = tril( beta_i * sum_d k_i k_j * exp2(gk_i - gk_j) )   （严格下三角）
    原版实现用 (batch, head, chunk) 三重 Python 循环逐块计算；本版本把
    「按 chunk 分块 + 按 head 组展开」向量化：

      - 序列补零到 C 的整数倍，reshape 为 [B, M, C, Hg, K]；
      - 一次 einsum 得到所有 (b, chunk, k-head) 的 Kc @ Kc^T；
      - g 的衰减 exp(clamp(gi - gj, 0)) 与 beta 逐行缩放按 [B, M, C, H] 广播，
        H 个 q/beta 头通过 hg = h // group 索引共享的 k-head 结果；
      - 严格下三角 mask（kernel: where(o_t[:,None] > o_t[None,:], A, 0)）。

    与原版逐块计算数学等价（fp32 累乘），仅把 O(B*H*T/C) 次小块发射压缩为
    O(T/C) 次整批向量化块，且 k-head 结果在共享它的 group 个 head 间只算一次。
    """

    FLA_CHUNK_SIZE = 64

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, k, g=None, beta=None, cu_seqlens=None, chunk_indices=None,
                chunk_size=None, output_dtype=torch.float32):
        """
        Args:
            k (Tensor): [B, T, Hg, K].
            g (Tensor, optional): 累计 gate [B, T, H].
            beta (Tensor): [B, T, H].
            cu_seqlens (Tensor, optional): 变长累计长度（约定 B == 1）.
            chunk_indices (Tensor, optional): 预计算 chunk 索引（本实现忽略，
                按 canonical 顺序处理，与原版一致）.
            chunk_size (int): chunk 大小，默认 64.
            output_dtype (torch.dtype): 输出 dtype，默认 fp32.

        Returns:
            Tensor: beta * K * K^T，shape [B, T, H, BT].
        """
        if chunk_size is None:
            chunk_size = self.FLA_CHUNK_SIZE
        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        BT = chunk_size
        group = H // Hg  # 每个 k 头组对应的 q/beta 头数
        dev = k.device

        A = torch.zeros(B, T, H, BT, device=dev, dtype=output_dtype)

        # chunk 内严格下三角 mask（kernel: where(o_t[:,None] > o_t[None,:], A, 0)）
        tril_mask = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=dev),
                               diagonal=-1)

        if cu_seqlens is not None:
            cu = cu_seqlens.tolist()
            spans = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
        else:
            spans = [(b, 0, T) for b in range(B)]

        for b, bos, eos in spans:
            T_seq = eos - bos
            M = (T_seq + BT - 1) // BT
            Tp = M * BT
            pad = Tp - T_seq

            # ---- 补零到 chunk 整数倍并分块 ----
            k_seg = k[b, bos:eos].float()                       # [T_seq, Hg, K]
            k_pad = torch.zeros(Tp, Hg, K, device=dev, dtype=torch.float32)
            k_pad[:T_seq] = k_seg
            kc = k_pad.view(M, BT, Hg, K)                       # [M, C, Hg, K]

            g_seg = g[b, bos:eos].float() if g is not None else None   # [T_seq, H]
            beta_seg = beta[b, bos:eos].float()                 # [T_seq, H]

            def _pad_head(x, T_seq_, Tp_):
                p = torch.zeros(Tp_, H, device=dev, dtype=torch.float32)
                p[:T_seq_] = x
                return p

            gc = _pad_head(g_seg, T_seq, Tp).view(M, BT, H) if g_seg is not None else None
            bc = _pad_head(beta_seg, T_seq, Tp).view(M, BT, H)  # [M, C, H]

            # ---- 一次 einsum：所有 (chunk, k-head) 的 Kc @ Kc^T ----
            kkt = torch.einsum('mihk,mjhk->mhij', kc, kc)       # [M, Hg, C, C]

            # ---- H 个 head 共享 group 个 k-head 结果 ----
            hg_idx = torch.arange(H, device=dev) // group        # [H]
            kkt_h = kkt[:, hg_idx]                              # [M, H, C, C]

            if g is not None:
                diff = gc.unsqueeze(2) - gc.unsqueeze(1)        # [M, C, C, H]
                decay = torch.exp(torch.clamp(diff, max=0.0))   # safe_exp
                decay = decay.permute(0, 3, 1, 2)               # [M, H, C, C]
            else:
                decay = 1.0

            beta_b = bc.permute(0, 2, 1).unsqueeze(-1)          # [M, H, C, 1]
            a = kkt_h * decay * beta_b * tril_mask              # [M, H, C, C]

            # ---- 写回：A[b, rows, h, :BT] ----
            a = a.permute(0, 2, 1, 3)                           # [M, C, H, C]
            A[b, bos:eos, :, :] = a.reshape(Tp, H, BT)[:T_seq].to(output_dtype)

        return A


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "14_ChunkScaledDotKktFwd.json")
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

    input_groups = []
    for case in cases:
        inputs = case["inputs"]

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        k_info = inputs[0]
        beta_info = inputs[1]

        dtype = dtype_map[k_info["dtype"]]

        k = random_tensor(k_info["shape"], dtype)
        beta = random_tensor(beta_info["shape"], dtype_map.get(beta_info.get("dtype"), dtype))

        g = None
        cu_seqlens = None
        chunk_indices = None
        chunk_size = None
        output_dtype = torch.float32

        for inp in inputs[2:]:
            name = inp.get("name", "")
            if name == "g":
                g_dtype = dtype_map.get(inp.get("dtype"), dtype)
                g = random_tensor(inp["shape"], g_dtype)
            elif name == "cu_seqlens":
                cu_seqlens = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "chunk_indices":
                chunk_indices = torch.tensor(inp["value"], dtype=torch.int32)
            elif name == "chunk_size":
                chunk_size = inp["value"]
            elif name == "output_dtype":
                output_dtype = dtype_map.get(inp["value"], torch.float32)

        input_groups.append([k, g, beta, cu_seqlens, chunk_indices, chunk_size, output_dtype])
    return input_groups


def get_init_inputs():
    return []
