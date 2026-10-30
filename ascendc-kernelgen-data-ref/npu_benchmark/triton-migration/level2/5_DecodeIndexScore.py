import torch
import torch.nn as nn
import json
import os

SPARSE_BLOCK_SIZE = 128  # 一个稀疏块 == 一个 KV page（vLLM 强制页大小=128）


class Model(nn.Module):
    """
    _decode_index_score_kernel —— MiniMax M3 lightning-indexer 的 decode
    阶段 index 块打分算子（split-K，cudagraph 安全；golden reference）。

    算子出处：
      vllm-project/vllm
      vllm/models/minimax_m3/common/ops/index_topk.py
        - _decode_index_score_kernel（Triton kernel 本体）
        - minimax_m3_index_decode_score（Python 包装）

    数学语义（逐请求 r、逐 query 偏移 q_off、逐 128-token 块 blk）：
      输入  idx_q:         [total_q, H, D]      —— total_q = num_reqs * dql
      输入  index_kv_cache:[num_pages, 128, D]  —— paged index-K 缓存（单头共享）
      输入  block_table:   [num_reqs, max_block] int32
      输入  seq_lens:      [num_reqs] int32     —— 含本轮 query 的总长度
      attr  init_blocks / local_blocks / decode_query_len(dql)
      输出  score:         [H, total_q, max_block] fp32

      query_pos = seq_lens[r] - dql + q_off          （本轮 query 的绝对位置）
      kv_len    = max(query_pos + 1, 0)              （0 长度 padding 行钳位）
      nb        = ceil(kv_len / 128)                 （该 token 可见块数）

      对 blk < nb：
        s = max_{ pos = blk*128 + j, pos < kv_len }
              ( idx_q[r*dql+q_off, h, :] . index_kv_cache[block_table[r,blk], j, :] )
        强制选块（保证后续 topk 必选，local 优先级高于 init）：
        local_start = max(0, nb - local_blocks)
        s = 1e29  if blk >= local_start            （is_local，窗口末尾块）
        s = 1e30  elif blk < init_blocks           （is_init，流式开头块）
      对 blk >= nb：kernel 写入全掩码后的 -inf 或不写（垃圾），golden 统一 -inf。

      - 无 softmax scale（源码注释：decode 只消费块排序，scale 省略）；
      - 除 pos < kv_len 的可见性掩码外无其他因果约束（decode 的 key 全部在历史）；
      - spec-decode（dql > 1）时同一请求内不同 q_off 的 kv_len 逐个 +1，
        最后一个块的有效 key 数不同，per-token 掩码必须精确。

    dtype 约定：
      q / cache 为 bf16（主）或 fp8_e4m3fn（FP8 index cache 路径，见 kernel
      内注释 "fp32 accumulation is required for the fp8 (e4m3) index cache"）。
      kernel 以存储 dtype 加载、tl.dot fp32 累加；golden 统一 .float() 后
      fp32 计算（fp8 -> fp32 为精确反量化），输出 fp32。

    与 kernel 的已知差异点：
      1. 【原 kernel bug，vLLM issue #49157】BLOCK_SIZE_HQ = num_idx_heads *
         next_pow2(dql) 作为 tl.arange 长度，非 2 的幂时编译失败（如 H=6,
         dql=1）。golden 实现广义语义，任意 H/dql 均正确，case 6（H=6）
         专门覆盖该差异——评测生成算子时应支持，不应照搬原 kernel 的缺陷。
      2. 未写入/全掩码块列 kernel 可能残留未初始化垃圾，golden 统一 -inf
         （下游 topk 只读 blk < nb 的列，差异不影响正确性）。
      3. split-K 的 num_kv_chunks、BLOCK_SIZE_Q、PDL、cudagraph 固定 grid
         均为调度/工程细节，不改变逐 token 语义，不纳入。
      4. score 列数 kernel 按 round_up(max_block, 16) 对齐，golden 恰好
         max_block 列。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, idx_q, index_kv_cache, block_table, seq_lens,
                init_blocks, local_blocks, decode_query_len):
        # ---- 形状自洽校验 ----
        assert idx_q.dim() == 3, f"idx_q 应为 3D, got {idx_q.shape}"
        total_q, H, D = idx_q.shape
        P, BS, D2 = index_kv_cache.shape
        assert BS == SPARSE_BLOCK_SIZE and D2 == D, \
            f"cache 形状应为 [P, 128, {D}], got {index_kv_cache.shape}"
        B, max_block = block_table.shape
        assert seq_lens.shape[0] == B
        assert total_q == B * decode_query_len, \
            f"total_q={total_q} != num_reqs({B}) * dql({decode_query_len})"
        dql = decode_query_len

        q = idx_q.float()  # fp32 内部计算（fp8 输入精确反量化）
        score = torch.full((H, total_q, max_block), float("-inf"),
                           dtype=torch.float32, device=idx_q.device)
        arange_bs = torch.arange(BS, device=idx_q.device)
        arange_q = torch.arange(dql, device=idx_q.device)

        for r in range(B):
            seq_len = int(seq_lens[r])
            sl = slice(r * dql, (r + 1) * dql)
            q_r = q[sl]                                       # [dql, H, D]
            query_pos = seq_len - dql + arange_q              # [dql]
            kv_len = (query_pos + 1).clamp(min=0)             # [dql]
            nb_q = (kv_len + BS - 1) // BS                    # 每 token 可见块数
            nb = min(int(nb_q.max().item()), max_block)       # 本请求块循环上界
            local_start = (nb_q - local_blocks).clamp(min=0)  # [dql]
            for blk in range(nb):
                page = int(block_table[r, blk])
                k = index_kv_cache[page].float()              # [BS, D]
                qk = torch.einsum("kd,qhd->hqk", k, q_r)      # [H, dql, BS]
                pos = blk * BS + arange_bs
                visible = pos[None, :] < kv_len[:, None]      # [dql, BS]
                qk = torch.where(visible[None, :, :], qk,
                                 torch.full_like(qk, float("-inf")))
                s = qk.max(dim=-1).values                     # [H, dql]
                # 强制 init/local（仅对可见块；local 优先于 init）
                is_vis = blk < nb_q                           # [dql]
                is_local = (blk >= local_start) & is_vis
                is_init = (blk < init_blocks) & is_vis
                s = torch.where(is_local[None, :],
                                torch.full_like(s, 1e29),
                                torch.where(is_init[None, :],
                                            torch.full_like(s, 1e30), s))
                score[:, sl, blk] = s
        return score


def get_input_groups():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "5_DecodeIndexScore.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    def random_tensor(shape, dtype):
        # 标准分布：50% 均匀 U[-5,5] + 50% 正态 N(mu, sigma)。
        # index q/k 是投影层输出，无对数域/归一化等物理约束；
        # fp8 路径由 bf16 生成后 cast（取值远小于 e4m3 上限 448，无饱和）。
        if torch.rand(1).item() < 0.5:
            t = torch.empty(shape, dtype=torch.bfloat16).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            t = torch.normal(mu, sigma, shape, dtype=torch.bfloat16)
        return t.to(dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 可复现性：每个 case 固定种子
        torch.manual_seed(3407 + case_idx)

        inputs = case["inputs"]
        q_info = next(i for i in inputs if i["name"] == "idx_q")
        cache_info = next(i for i in inputs if i["name"] == "index_kv_cache")
        bt_info = next(i for i in inputs if i["name"] == "block_table")
        seq_lens = next(i for i in inputs if i["name"] == "seq_lens")["value"]
        dql = next(i for i in inputs
                   if i["name"] == "decode_query_len")["value"]
        init_blocks = next(i for i in inputs
                           if i["name"] == "init_blocks")["value"]
        local_blocks = next(i for i in inputs
                            if i["name"] == "local_blocks")["value"]

        dtype = dtype_map[q_info["dtype"]]
        idx_q = random_tensor(q_info["shape"], dtype)
        index_kv_cache = random_tensor(cache_info["shape"], dtype)

        # ---- JSON 形状与 attr 推导值自洽校验 ----
        total_q, H, D = q_info["shape"]
        P, BS, _ = cache_info["shape"]
        B, max_block = bt_info["shape"]
        assert BS == SPARSE_BLOCK_SIZE
        assert B == len(seq_lens)
        assert total_q == B * dql, \
            f"case {case_idx}: total_q={total_q} != B*dql={B * dql}"

        needed = [(sl + BS - 1) // BS for sl in seq_lens]
        assert max(needed) <= max_block, \
            f"case {case_idx}: max_block={max_block} 不足 {max(needed)}"
        # 页 id 按 kernel 语义生成：随机打乱 + 至少 3 个冗余未引用页
        assert P >= sum(needed) + 3, \
            f"case {case_idx}: 页数 {P} 不足 sum(needed)+3={sum(needed) + 3}"

        perm = torch.randperm(P, dtype=torch.int32)
        block_table = torch.zeros(B, max_block, dtype=torch.int32)
        off = 0
        sum_needed = sum(needed)
        unused = P - sum_needed
        for b in range(B):
            nb = needed[b]
            block_table[b, :nb] = perm[off:off + nb]
            off += nb
            for j in range(nb, max_block):  # 尾部填未引用页（kernel 不读）
                block_table[b, j] = perm[sum_needed + (j - nb) % unused]

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        input_groups.append([idx_q, index_kv_cache, block_table, seq_lens_t,
                             init_blocks, local_blocks, dql])
    return input_groups


def get_init_inputs():
    return []