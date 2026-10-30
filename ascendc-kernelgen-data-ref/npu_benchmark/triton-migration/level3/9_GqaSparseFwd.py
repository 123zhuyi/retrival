import torch
import torch.nn as nn
import json
import os

SPARSE_BLOCK_SIZE = 128  # 一个稀疏块 == 一个 KV 页


class Model(nn.Module):
    """
    Model that performs MiniMax-M3 GQA block-sparse attention (prefill).
    vLLM _gqa_sparse_fwd_kernel (vllm/models/minimax_m3/common/ops/sparse_attn.py,
    wrapper minimax_m3_sparse_attn, block_size_q == 1 路径):
        minimax_m3_sparse_attn(q, kv_cache, topk_idx, block_table, cu_seqlens_q,
            seq_lens, prefix_lens, max_query_len, num_kv_heads, sm_scale, output)
        -> output [total_q, num_heads, head_dim]

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致):
    KV cache 布局 [num_blocks, num_kv_heads, 128, 2*head_dim],
    K=[..., :head_dim], V=[..., head_dim:]; 稀疏块大小 128 == 页大小。
    topk_idx [num_kv_heads, total_q, topk] 为 lightning indexer 选中的逻辑块号
    (无哨兵值), 实际读取块数由 query 绝对位置推出:
        q_abs        = prefix_len + j            # 第 b 个请求第 j 个 query
        valid_blocks = (q_abs + 128) // 128      # 因果可见的块数
        real_topk    = min(topk, valid_blocks)   # 仅前 real_topk 个索引被读取
    每个 query token、每个 kv_head (GQA: q 头组 [kh*G, (kh+1)*G)):
        对每个选中块 blk: page = block_table[b, blk],
        位置 pos ∈ [blk*128, blk*128+128) 有效当且仅当 pos < seq_len 且 pos <= q_abs
        logits = q @ K^T * sm_scale            # 无效位置 -inf
        o = softmax(logits) @ V
    注:
      1. kernel 内部为 base-2 softmax (sm_scale*log2e 后 exp2), 与自然底
         softmax 数学等价, golden 用 torch.softmax 表达;
      2. 无 attention sink; FP8 KV cache 与 k_scale/v_scale 反量化路径未纳入
         (kv_cache 取与 q 相同 dtype);
      3. num_q_loop/cu_seqblocks_q (block_size_q>1 预留) 与 strides 仅为调度
         参数, 未纳入;
      4. 内部 fp32 计算, 输出回铸 q.dtype。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, kv_cache, topk_idx, block_table, cu_seqlens_q,
                seq_lens, prefix_lens, sm_scale):
        total_q, H, D = q.shape
        KH = kv_cache.shape[1]
        BS = SPARSE_BLOCK_SIZE
        G = H // KH
        topk = topk_idx.shape[-1]
        B = cu_seqlens_q.numel() - 1
        cu = cu_seqlens_q.tolist()

        o = torch.zeros(total_q, H, D, dtype=torch.float32, device=q.device)
        arange_bs = torch.arange(BS, device=q.device)

        for b in range(B):
            qs, qe = cu[b], cu[b + 1]
            seq_len = int(seq_lens[b].item())
            prefix = int(prefix_lens[b].item())
            for j in range(qe - qs):
                q_abs = prefix + j
                valid_blocks = (q_abs + BS) // BS
                real_topk = min(topk, valid_blocks)
                if real_topk <= 0:
                    continue
                for kh in range(KH):
                    blks = topk_idx[kh, qs + j, :real_topk].long()
                    pages = block_table[b, blks].long()
                    kv = kv_cache[pages, kh].float()       # [rt, 128, 2D]
                    k = kv[..., :D].reshape(-1, D)         # [rt*128, D]
                    v = kv[..., D:].reshape(-1, D)
                    pos = (blks[:, None] * BS + arange_bs[None, :]).reshape(-1)
                    mask = (pos < seq_len) & (pos <= q_abs)
                    if not bool(mask.any()):
                        continue
                    q_g = q[qs + j, kh * G:(kh + 1) * G].float()   # [G, D]
                    logits = (q_g @ k.t()) * sm_scale              # [G, L]
                    logits = logits.masked_fill(~mask[None, :], float("-inf"))
                    o[qs + j, kh * G:(kh + 1) * G] = \
                        torch.softmax(logits, dim=-1) @ v

        return o.to(q.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "9_GqaSparseFwd.json")
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
        "int32": torch.int32,  # topk_idx/block_table 仅声明形状, 实际由逻辑生成
    }
    BS = SPARSE_BLOCK_SIZE

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, dtypes = {}, {}
        cu_seqlens_q = seq_lens = prefix_lens = None
        sm_scale = None
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "cu_seqlens_q":
                cu_seqlens_q = inp["value"]
            elif name == "seq_lens":
                seq_lens = inp["value"]
            elif name == "prefix_lens":
                prefix_lens = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]

        total_q, H, D = shapes["q"]
        KH = shapes["kv_cache"][1]
        topk = shapes["topk_idx"][-1]
        B = len(seq_lens)
        assert cu_seqlens_q[0] == 0 and cu_seqlens_q[-1] == total_q
        assert shapes["topk_idx"] == [KH, total_q, topk]
        assert H % KH == 0
        q_lens = [cu_seqlens_q[b + 1] - cu_seqlens_q[b] for b in range(B)]
        # prefill 语义: kv 总长 = prefix + 本次 query 长度
        assert all(s == p + ql for s, p, ql in zip(seq_lens, prefix_lens, q_lens))

        q = random_tensor(shapes["q"], dtypes["q"])
        kv_cache = random_tensor(shapes["kv_cache"], dtypes["kv_cache"])
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32)

        # 分页: 每请求逻辑块数 = ceil(seq_len/128), 页 id 随机打乱 + 3 冗余页
        nblocks = [(s + BS - 1) // BS for s in seq_lens]
        total_pages = sum(nblocks) + 3
        assert shapes["kv_cache"][0] == total_pages
        max_blocks = max(nblocks)
        perm = torch.randperm(total_pages)
        block_table = torch.zeros(B, max_blocks, dtype=torch.int32)
        off = 0
        for b, n in enumerate(nblocks):
            block_table[b, :n] = perm[off:off + n].int()
            off += n

        # topk_idx [KH, total_q, topk]: 每个 (kh, query token) 从因果可见块中
        # 无放回采样 real_topk 个逻辑块号并升序排列 (模拟 indexer 输出, 无哨兵);
        # 尾部 (kernel 不读取) 填范围内的随机块号
        topk_idx = torch.zeros(KH, total_q, topk, dtype=torch.int32)
        for b in range(B):
            qs = cu_seqlens_q[b]
            for j in range(q_lens[b]):
                q_abs = prefix_lens[b] + j
                valid_blocks = (q_abs + BS) // BS
                n_sel = min(topk, valid_blocks)
                for kh in range(KH):
                    sel = torch.randperm(valid_blocks)[:n_sel].sort().values
                    topk_idx[kh, qs + j, :n_sel] = sel.int()
                    if n_sel < topk:
                        topk_idx[kh, qs + j, n_sel:] = torch.randint(
                            0, valid_blocks, (topk - n_sel,), dtype=torch.int32)

        input_groups.append([q, kv_cache, topk_idx, block_table, cu_seqlens_q_t,
                             seq_lens_t, prefix_lens_t, sm_scale])
    return input_groups


def get_init_inputs():
    return []