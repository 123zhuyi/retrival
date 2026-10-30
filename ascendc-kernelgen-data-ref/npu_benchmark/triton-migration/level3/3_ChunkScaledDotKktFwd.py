import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs chunk-scaled dot product KKT forward computation.
    Computes beta * K * K^T with optional gate accumulation and chunked processing.

    This kernel supports Q/K with different head numbers (fla assumes Hg == H).

    Golden reference: torch 原生小算子拼接实现
    对每个 (batch b, head h, chunk i_t)，记该 chunk 覆盖的行区间为
    [s, s+L)（L <= BT，BT 为 chunk_size，默认 64）：
        1. 取 k 片段 Kc = k[b, s:s+L, h // (H // Hg), :].float()    # [L, K]
        2. A_chunk = Kc @ Kc^T                                     # fp32 累乘，[L, L]
        3. 若给定 g（g_cumsum, [B, T, H]）：
               diff = gc[:, None] - gc[None, :]
               A_chunk *= exp(min(diff, 0))        # 对应 kernel 的 safe_exp
        4. A_chunk *= beta[b, s:s+L, h].float()[:, None]           # 逐行缩放
        5. 严格下三角 mask：仅保留 chunk 内 i > j 的元素，其余置 0
        6. 写入 A[b, s:s+L, h, :L]，其余列保持 0（对应 kernel 越界 load 为 0）
    varlen 模式（cu_seqlens 不为 None，约定 B == 1）：按 cu_seqlens 切分序列，
    每条序列内部独立分 chunk，语义与 kernel 的 chunk_indices 路径一致。
    精度说明：kernel 中 tl.dot 为 bf16 输入 / fp32 累加，本实现统一先升 fp32
    再 matmul，累加精度不低于 golden kernel。
    """

    FLA_CHUNK_SIZE = 64

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, k, g=None, beta=None, cu_seqlens=None, chunk_indices=None,
                chunk_size=None, output_dtype=torch.float32):
        """
        Computes beta * K * K^T.

        Args:
            k (Tensor): Key tensor of shape [B, T, Hg, K].
            g (Tensor, optional): Cumulative sum of the gate tensor of shape [B, T, H].
            beta (Tensor): Beta tensor of shape [B, T, H].
            cu_seqlens (Tensor, optional): Cumulative sequence lengths of the input tensor.
            chunk_indices (Tensor, optional): Pre-computed chunk indices.
            chunk_size (int): The chunk size. Default: 64.
            output_dtype (torch.dtype): The dtype of the output tensor. Default: torch.float32.

        Returns:
            Tensor: beta * K * K^T of shape [B, T, H, BT] where BT is the chunk size.
        """
        if chunk_size is None:
            chunk_size = self.FLA_CHUNK_SIZE
        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        BT = chunk_size
        group = H // Hg  # 每个 k 头组对应的 q/beta 头数

        A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)

        # chunk 内严格下三角 mask（kernel: where(o_t[:,None] > o_t[None,:], A, 0)）
        tril_mask = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=k.device),
                               diagonal=-1)

        # 构造待处理的 (batch_idx, bos, eos) 列表
        if cu_seqlens is not None:
            # varlen：约定 B == 1，k 为 [1, total_T, Hg, K] 的拉直序列
            cu = cu_seqlens.cpu().tolist()
            spans = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
        else:
            spans = [(b, 0, T) for b in range(B)]

        for b, bos, eos in spans:
            T_seq = eos - bos
            for h in range(H):
                hg = h // group
                for s in range(0, T_seq, BT):
                    row0, row1 = bos + s, min(bos + s + BT, eos)
                    L = row1 - row0

                    # 1+2: chunk 内 K @ K^T（fp32 累乘）
                    kc = k[b, row0:row1, hg, :].to(torch.float32)          # [L, K]
                    a = kc @ kc.transpose(0, 1)                            # [L, L]

                    # 3: 门控衰减 exp(min(gi - gj, 0))，对应 safe_exp
                    if g is not None:
                        gc = g[b, row0:row1, h].to(torch.float32)          # [L]
                        diff = gc[:, None] - gc[None, :]
                        a = a * torch.exp(torch.clamp(diff, max=0.0))

                    # 4: 逐行乘 beta
                    bc = beta[b, row0:row1, h].to(torch.float32)           # [L]
                    a = a * bc[:, None]

                    # 5+6: 严格下三角 mask 后写入；:L 之外的列保持 0
                    a = a * tril_mask[:L, :L]
                    A[b, row0:row1, h, :L] = a.to(output_dtype)

        return A


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "3_ChunkScaledDotKktFwd.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        # 输入分布为新框架标准（与 _rand_tensor 默认一致）：
        # 50% 均匀分布 U(-5, 5) + 50% 正态分布 N(mu, sigma)，
        # 其中 mu ~ U(-5, 5)，sigma ~ U(0.1, 2)。
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            # bf16 不支持 torch.normal，先 fp32 生成再降精度（与 _rand_tensor 一致）
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
        # beta 精度以 case 声明为准，缺省回退到 k 的 dtype
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