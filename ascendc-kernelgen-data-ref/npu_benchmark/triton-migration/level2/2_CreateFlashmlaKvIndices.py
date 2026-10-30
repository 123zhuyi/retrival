import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    """
    Model that creates FlashMLA block-level kv_indices (paged KV 索引构建)。
    对标 sglang create_flashmla_kv_indices_triton
    (python/sglang/kernels/ops/kvcache/kv_indices.py:99,
    sglang/srt/layers/attention/flashmla_backend.py 调用, PAGED_SIZE=64)。

    数学定义 (对每个 batch 条目 b, 页大小 ps):
        n_pages = ceil((kv_start[b] + seq_len[b] - kv_start[b]) / ps) = ceil(seq_len[b] / ps)
        对 j = 0 .. n_pages-1:
            kv_indices[b, j] = req_to_token[req_pool_indices[b], kv_start[b] + j*ps] // ps
        其余位置 (j >= n_pages) 保持 -1 不变 (调用方预填充, kernel 不写)。

    布局约定:
        req_to_token     [max_batch, max_context_len] int32
                         req_pool 行 -> 物理 token 槽位 (page_id * ps + 页内偏移),
                         未使用位置 -1; kernel 只读取每页首 token 的槽位
        req_pool_indices [bs] int32, batch 条目 -> req_to_token 行号 (互不重复)
        page_kernel_lens [bs] int32, 每条序列的有效 KV 长度 (>= 1)
        kv_start_idx     [bs] int32 可选 (形状 [0] 表示 None):
                         每条序列在 req_to_token 行内的起始偏移, FlashMLA
                         decode/target-verify 路径恒为 None
        kv_indices_width int32 attr: 输出缓冲行宽 (= cdiv(max_len, ps))
        page_size        int32 attr: FlashMLA 页大小, 恒 64
    输出:
        kv_indices [bs, kv_indices_width] int32, 有效前缀为页号, 其余 -1

    与 kernel 的差异: 无 (纯索引映射; BLOCK 分块只是调度)。
    """

    def __init__(self):
        super().__init__()

    def forward(self, req_to_token, req_pool_indices, page_kernel_lens,
                kv_start_idx, kv_indices_width: int, page_size: int = 64):
        bs = page_kernel_lens.shape[0]
        kv_indices = torch.full((bs, kv_indices_width), -1, dtype=torch.int32)
        for b in range(bs):
            pool = int(req_pool_indices[b])
            kv_start = int(kv_start_idx[b]) if kv_start_idx.numel() else 0
            kv_len = int(page_kernel_lens[b])
            n_pages = (kv_len + page_size - 1) // page_size
            for j in range(min(n_pages, kv_indices_width)):
                slot = int(req_to_token[pool, kv_start + j * page_size])
                kv_indices[b, j] = slot // page_size
        return kv_indices


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "2_CreateFlashmlaKvIndices.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {"int32": torch.int32}

    input_groups = []
    for case_idx, case in enumerate(cases):
        # 固定随机种子保证标杆可复现: 每个 case 独立种子, 重复调用结果完全一致
        torch.manual_seed(3407 + case_idx)

        shapes, kv_indices_width, page_size = {}, None, 64
        seq_lens, kv_start = None, None
        for inp in case["inputs"]:
            name = inp.get("name", "")
            if inp.get("type") == "tensor":
                shapes[name] = inp["shape"]
            elif name == "page_kernel_lens":
                seq_lens = inp["value"]
            elif name == "kv_start_idx":
                kv_start = inp["value"]
            elif name == "kv_indices_width":
                kv_indices_width = inp["value"]
            elif name == "page_size":
                page_size = inp["value"]

        max_batch, max_ctx = shapes["req_to_token"]
        bs = len(seq_lens)
        assert all(l >= 1 for l in seq_lens)
        if kv_start:
            assert all(s + l <= max_ctx for s, l in zip(kv_start, seq_lens))
        else:
            assert all(l <= max_ctx for l in seq_lens)

        # 物理约束: req_to_token 由分页 KV cache 管理器维护 —— 每条序列占用
        # ceil(len/ps) 个物理页, 页 id 全局打乱 (含冗余空闲页), 页内槽位连续
        # (page_id * ps + 页内偏移), 未使用位置 -1
        req_pool_indices = torch.randperm(max_batch)[:bs].int()
        req_to_token = torch.full((max_batch, max_ctx), -1, dtype=torch.int32)
        # 物理页全局唯一 (整个 KV cache 共 max_batch * max_ctx/ps 个物理页, 打乱分配)
        total_pages = max_ctx // page_size
        global_pages = torch.randperm(max_batch * total_pages).tolist()
        pg_off = 0
        for b in range(bs):
            pool = int(req_pool_indices[b])
            start = kv_start[b] if kv_start else 0
            n_pages = (seq_lens[b] + page_size - 1) // page_size
            pages = global_pages[pg_off:pg_off + n_pages]
            pg_off += n_pages
            for j, pg in enumerate(pages):
                tok0 = start + j * page_size
                tok1 = min(tok0 + page_size, start + seq_lens[b])
                req_to_token[pool, tok0:tok1] = torch.arange(
                    pg * page_size, pg * page_size + (tok1 - tok0),
                    dtype=torch.int32)

        kv_start_t = (torch.tensor(kv_start, dtype=torch.int32)
                      if kv_start else torch.zeros(0, dtype=torch.int32))
        input_groups.append([req_to_token, req_pool_indices,
                             torch.tensor(seq_lens, dtype=torch.int32),
                             kv_start_t, kv_indices_width, page_size])
    return input_groups


def get_init_inputs():
    return []