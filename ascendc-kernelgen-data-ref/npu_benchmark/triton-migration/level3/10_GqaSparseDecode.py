import torch
import torch.nn as nn
import json
import os

SPARSE_BLOCK_SIZE = 128  # 一个稀疏块 == 一个 KV 页


class Model(nn.Module):
    """
    Model that performs MiniMax-M3 GQA block-sparse attention (decode).
    vLLM _gqa_sparse_decode_kernel + _merge_topk_attn_out_kernel
    (vllm/models/minimax_m3/common/ops/sparse_attn.py,
    wrapper minimax_m3_sparse_attn_decode):
        minimax_m3_sparse_attn_decode(q, kv_cache, topk_idx, block_table,
            seq_lens, num_kv_heads, sm_scale, output, decode_query_len)
        -> output [total_q, num_heads, head_dim]

    torch 原生小算子拼接参考实现(与 Triton kernel 数学语义一致):
    decode batch 按请求主序展平, total_q = num_reqs * decode_query_len
    (投机解码时 dql > 1)。KV cache 布局 [num_blocks, num_kv_heads, 128,
    2*head_dim], K=[..., :head_dim], V=[..., head_dim:]; 稀疏块 128 == 页。
    第 pid_b 个 query token:
        req_id   = pid_b // dql,  q_offset = pid_b % dql
        query_pos = seq_len - dql + q_offset     # 该 token 的绝对位置
        kv_len    = max(query_pos + 1, 0)        # 因果上界(含自身)
        real_topk = min(topk, ceil(kv_len / 128))
    每个 kv_head (GQA: q 头组 [kh*G, (kh+1)*G)):
        对前 real_topk 个选中块: page = block_table[req, blk],
        位置 pos 有效当且仅当 pos < kv_len (因果性已由 kv_len 蕴含)
        logits = q @ K^T * sm_scale
        o = softmax(logits) @ V
    注:
      1. kernel 采用 split-K (flash-decoding): topk 维切成 NUM_TOPK_CHUNKS 份
         各算部分 online softmax 再由 merge kernel 合并, 分块数由启动启发式
         决定, 属实现细节; 合并结果与单次 softmax 数学等价, golden 直接计算
         最终结果 (partials/lse_partial 为内部缓冲, 未纳入);
      2. kernel 内部为 base-2 softmax, 与自然底数学等价;
      3. 全空 padded row (kv_len <= 0, cuda graph 填充) kernel 侧为 NaN,
         golden 置 0 并避免在输入中生成此类行;
      4. 无 attention sink; FP8 KV cache 与 k_scale/v_scale 反量化路径未纳入;
      5. 内部 fp32 计算, 输出回铸 q.dtype。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, kv_cache, topk_idx, block_table, seq_lens,
                sm_scale, decode_query_len):
        total_q, H, D = q.shape
        KH = kv_cache.shape[1]
        BS = SPARSE_BLOCK_SIZE
        G = H // KH
        topk = topk_idx.shape[-1]
        dql = decode_query_len

        o = torch.zeros(total_q, H, D, dtype=torch.float32, device=q.device)
        arange_bs = torch.arange(BS, device=q.device)

        for pid_b in range(total_q):
            req_id = pid_b // dql
            q_offset = pid_b - req_id * dql
            seq_len = int(seq_lens[req_id].item())
            query_pos = seq_len - dql + q_offset
            kv_len = max(query_pos + 1, 0)
            num_blocks = (kv_len + BS - 1) // BS
            real_topk = min(topk, num_blocks)
            if real_topk <= 0:
                continue  # 全空 padded row -> 输出 0 (kernel 侧为 NaN, 见 docstring)
            for kh in range(KH):
                blks = topk_idx[kh, pid_b, :real_topk].long()
                pages = block_table[req_id, blks].long()
                kv = kv_cache[pages, kh].float()       # [rt, 128, 2D]
                k = kv[..., :D].reshape(-1, D)
                v = kv[..., D:].reshape(-1, D)
                pos = (blks[:, None] * BS + arange_bs[None, :]).reshape(-1)
                mask = pos < kv_len
                if not bool(mask.any()):
                    continue
                q_g = q[pid_b, kh * G:(kh + 1) * G].float()      # [G, D]
                logits = (q_g @ k.t()) * sm_scale                # [G, L]
                logits = logits.masked_fill(~mask[None, :], float("-inf"))
                o[pid_b, kh * G:(kh + 1) * G] = \
                    torch.softmax(logits, dim=-1) @ v

        return o.to(q.dtype)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "10_GqaSparseDecode.json")
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
        seq_lens = None
        sm_scale = None
        decode_query_len = None
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
                dtypes[name] = dtype_map[inp["dtype"]]
            elif name == "seq_lens":
                seq_lens = inp["value"]
            elif name == "sm_scale":
                sm_scale = inp["value"]
            elif name == "decode_query_len":
                decode_query_len = inp["value"]

        total_q, H, D = shapes["q"]
        KH = shapes["kv_cache"][1]
        topk = shapes["topk_idx"][-1]
        B = len(seq_lens)
        dql = decode_query_len
        assert total_q == B * dql
        assert shapes["topk_idx"] == [KH, total_q, topk]
        assert H % KH == 0
        assert all(s >= dql for s in seq_lens)  # query_pos >= 0

        q = random_tensor(shapes["q"], dtypes["q"])
        kv_cache = random_tensor(shapes["kv_cache"], dtypes["kv_cache"])
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

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

        # topk_idx [KH, total_q, topk]: 每个 (kh, query token) 从因果可见块
        # (ceil(kv_len/128) 个) 中无放回采样并升序排列, 尾部填范围内随机块号
        topk_idx = torch.zeros(KH, total_q, topk, dtype=torch.int32)
        for pid_b in range(total_q):
            req_id = pid_b // dql
            q_offset = pid_b - req_id * dql
            kv_len = seq_lens[req_id] - dql + q_offset + 1
            num_blocks = (kv_len + BS - 1) // BS
            n_sel = min(topk, num_blocks)
            for kh in range(KH):
                sel = torch.randperm(num_blocks)[:n_sel].sort().values
                topk_idx[kh, pid_b, :n_sel] = sel.int()
                if n_sel < topk:
                    topk_idx[kh, pid_b, n_sel:] = torch.randint(
                        0, num_blocks, (topk - n_sel,), dtype=torch.int32)

        input_groups.append([q, kv_cache, topk_idx, block_table, seq_lens_t,
                             sm_scale, dql])
    return input_groups


def get_init_inputs():
    return []