import torch
import torch.nn as nn
import json
import os

SPARSE_BLOCK_SIZE = 128  # 一个稀疏块 == 一个 KV page（vLLM 强制页大小=128）


class Model(nn.Module):
    """
    _index_block_score_kernel —— MiniMax M3 lightning-indexer 的 prefill
    阶段 index 块打分算子（golden reference）。

    算子出处：
      vllm-project/vllm
      vllm/models/minimax_m3/common/ops/index_topk.py
        - _index_block_score_kernel（Triton kernel 本体）
        - minimax_m3_index_score（Python 包装）
      配套 topk 选块在 _topk_index_kernel（不在本算子范围）。

    数学语义（逐请求、逐 query token、逐 128-token 块）：
      输入  idx_q:         [total_q, H, D]      —— varlen 打包的 index 查询
      输入  index_kv_cache:[num_pages, 128, D]  —— paged index-K 缓存（单头共享）
      输入  block_table:   [B, max_block] int32 —— 逻辑块 -> 物理页映射
      输入  cu_seqlens:    [B+1] int32          —— 各请求 query 起点（前缀和）
      输入  seq_lens:      [B] int32            —— 各请求总 KV 长度
      输入  prefix_lens:   [B] int32            —— 本 chunk 之前的上下文长度
      输出  score:         [H, total_q, max_block] fp32

      对请求 b 的第 t 个 query（绝对位置 p = prefix_lens[b] + t）、块 blk：
        score[h, t, blk] = max_{ pos = blk*128 + j, j in [0,128), pos <= p }
                             ( idx_q[t, h, :] . index_kv_cache[block_table[b,blk], j, :] )
      - 无 softmax scale：M3 的 index score 仅用于块排序选块，常数缩放不改变
        topk 顺序，源码逐行确认两个 score kernel 均不乘 scale；
      - 因果掩码 pos <= p 同时天然挡住了页内 pos >= seq_len 的未写入垃圾
        （query 位置 p < seq_len，任何 pos >= seq_len 必在未来），与 kernel
        "无需 mask K 加载"的注释一致；
      - 块内无任何 key 可见（blk*128 > p，整块在未来）时 score = -inf。

    dtype 约定：
      q / cache 为 bf16（主）、fp16、fp32；kernel 内 tl.dot 以 fp32 累加。
      golden 统一转 fp32 计算，输出 fp32，与 kernel 的 fp32 输出约定一致。

    与 kernel 的已知差异点：
      1. kernel 对「超出本 tile 因果范围」的块列不写入（score 缓冲由
         torch.empty 分配，残留未初始化垃圾）；golden 将这些列统一定义为
         -inf。下游 _topk_index_kernel 只读 valid_blocks = ceil((p+1)/128)
         之前的列，该区域两者逐位一致，差异不影响正确性闭环。
      2. kernel 的 score 列数按 round_up(max_block, 16) 对齐分配（避免
         Triton 重编译的内存布局细节）；golden 输出恰好 max_block 列。
      3. BLOCK_SIZE_Q=64 的 query  tiling 不改变逐 token 语义，不纳入。
      4. prefill kernel 将 head 放在 grid 维，任意 H 均可（无 2 的幂约束，
         与 decode 版不同），case 2 以 H=6 覆盖。
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, idx_q, index_kv_cache, block_table,
                cu_seqlens, seq_lens, prefix_lens):
        # ---- 形状自洽校验 ----
        assert idx_q.dim() == 3, f"idx_q 应为 3D, got {idx_q.shape}"
        total_q, H, D = idx_q.shape
        P, BS, D2 = index_kv_cache.shape
        assert BS == SPARSE_BLOCK_SIZE and D2 == D, \
            f"cache 形状应为 [P, 128, {D}], got {index_kv_cache.shape}"
        B, max_block = block_table.shape
        assert cu_seqlens.shape[0] == B + 1
        assert seq_lens.shape[0] == B and prefix_lens.shape[0] == B
        assert int(cu_seqlens[-1]) == total_q

        q = idx_q.float()  # fp32 内部计算
        score = torch.full((H, total_q, max_block), float("-inf"),
                           dtype=torch.float32, device=idx_q.device)
        arange_bs = torch.arange(BS, device=idx_q.device)

        for b in range(B):
            s = int(cu_seqlens[b]); e = int(cu_seqlens[b + 1])
            q_len = e - s
            if q_len == 0:
                continue
            prefix = int(prefix_lens[b]); seq_len = int(seq_lens[b])
            # 该请求最后一个 query 可见的块数（更早 token 的块范围是其子集）
            hi = min(seq_len, prefix + q_len)
            nb = min((hi + BS - 1) // BS, max_block)
            abs_pos = prefix + torch.arange(q_len, device=idx_q.device)
            for blk in range(nb):
                page = int(block_table[b, blk])
                k = index_kv_cache[page].float()              # [BS, D]
                qk = torch.einsum("qhd,kd->hqk", q[s:e], k)   # [H, q_len, BS]
                pos = blk * BS + arange_bs
                causal = abs_pos[:, None] >= pos[None, :]     # [q_len, BS]
                qk = torch.where(causal[None, :, :], qk,
                                 torch.full_like(qk, float("-inf")))
                score[:, s:e, blk] = qk.max(dim=-1).values
        return score


def get_input_groups():
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "4_IndexBlockScore.json")
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
        # index q/k 是投影层输出，无对数域/归一化等物理约束，标准分布即可。
        if torch.rand(1).item() < 0.5:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)
        else:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 可复现性：每个 case 固定种子
        torch.manual_seed(3407 + case_idx)

        inputs = case["inputs"]
        q_info = next(i for i in inputs if i["name"] == "idx_q")
        cache_info = next(i for i in inputs if i["name"] == "index_kv_cache")
        bt_info = next(i for i in inputs if i["name"] == "block_table")
        q_lens = next(i for i in inputs if i["name"] == "q_lens")["value"]
        prefix_lens = next(i for i in inputs
                           if i["name"] == "prefix_lens")["value"]

        dtype = dtype_map[q_info["dtype"]]
        idx_q = random_tensor(q_info["shape"], dtype)
        index_kv_cache = random_tensor(cache_info["shape"], dtype)

        # ---- JSON 形状与 attr 推导值自洽校验 ----
        total_q, H, D = q_info["shape"]
        P, BS, _ = cache_info["shape"]
        B, max_block = bt_info["shape"]
        assert BS == SPARSE_BLOCK_SIZE
        assert B == len(q_lens) == len(prefix_lens)
        assert total_q == sum(q_lens), \
            f"case {case_idx}: total_q={total_q} != sum(q_lens)={sum(q_lens)}"

        seq_lens = [p + ql for p, ql in zip(prefix_lens, q_lens)]
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

        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(q_lens, dtype=torch.int32), dim=0)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
        prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32)

        input_groups.append([idx_q, index_kv_cache, block_table,
                             cu_seqlens, seq_lens_t, prefix_lens_t])
    return input_groups


def get_init_inputs():
    return []