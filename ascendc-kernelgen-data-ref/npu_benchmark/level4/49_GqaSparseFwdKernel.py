import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    top-k 块稀疏 GQA 注意力前向 —— extend/投机解码形态的 golden reference。
    对标 MiniMax-M3 MSA（Main Sparse Attention）主注意力：
    官方 kernel 仓 MiniMax-AI/MSA（SM100 CuTe 实现，arXiv:2606.13392）；
    官方 Triton 参考实现在 vLLM 主仓
    vllm/model_executor/models/minimax_m3/common/sparse_attention.py
    （Triton 前向 minimax_m3_sparse_attn + Triton decode
    minimax_m3_sparse_attn_decode，v0.26.x 线）。

    数学语义（与上游逐条核对一致）：
      - 分页 KV：物理页大小 128，K/V 沿末维拼接存储 [pages, Hkv, 128, 2*D]；
        逻辑块 logical_block = pos // 128 经 block_table[request, block]
        映射到物理页，页内行 = pos % 128。
      - top-k 选择：topk_idx[kv_head, token, topk] 给出每个 query token
        在其所属 kv_head 下选中的逻辑块集合（GQA 组内共享同一选择），
        -1 为尾部填充哨兵；选中块掩码与因果掩码相与：
        mask = (logical_block ∈ selected) & (pos <= prefix_len + local_idx)。
      - 打分链：scores = (q_group @ k^T) * sm_scale（fp32）→ masked_fill
        (-inf) → softmax(dim=-1) → @ values → 回铸 q.dtype 落盘。
      - 多请求 varlen：cu_seqlens_q 切分 ragged query 序列，seq_lens 为
        各请求总 KV 长度，prefix_lens = seq_len - q_len（extend 前缀）。

    覆盖边界（与上游差异，非语义偏差）：
      - golden 只做 main 分支（topk 注意力本体）；init_blocks/local_window
        规则与 merge kernel 不在本算子域内（topk 选择在上游外层完成）。
      - FP8 E4M3 KV 量化路径（K/V scale 反量化）不在 golden 声明域，
        JSON 不出现 scale 张量。
      - 哨兵约定：topk_idx 恒以 -1 填充；因 logical_blocks >= 0，-1 永不
        匹配任何逻辑块，golden 天然安全（上游 kernel 同约定）。
      - 官方 decode kernel 假设各请求均匀 decode_query_len，golden 支持
        ragged q_len（逐请求 cu_seqlens 切分），语义为超集。
      - 数值路径：逐点镜像官方 kernel 的 **online base-2 softmax**
        （orig_vllm_msa_sparse_attn.py:182 QK、:210 PV、:212 lse），
        非单遍自然底 softmax：按 topk 槽位顺序逐块迭代（kernel 顺序读
        前 real_topk 个槽位，real_topk = min(topk, (q_abs+128)//128)），
        qk = dot(q,k) * sm_scale_log2e 在 log2 域（sm_scale_log2e =
        sm_scale * 1.4426950409，kernel host 侧常量），running max m_i
        逐块以 exp2 因子重缩放 acc，lse 以
        m_ij + log2(exp2(lse - m_ij) + l_ij) 递推，末步
        acc * exp2(m - lse) 归一。无效槽位（j >= real_topk，-1 哨兵）
        以全 -inf 行喂入，更新为精确空操作（p=0、rescale=1、lse 不变），
        与 kernel 不读该槽位等价。
        16-bit 输入时 QK 为精确乘积 + fp32 累加，PV 前 P（未归一、相对
        当前块 max）舍回 v.dtype（kernel `tl.dot(p.to(v.dtype), v)` 同）；
        fp32 输入时 kernel 两处 tl.dot 走 TF32（10 位尾数），golden 对
        QK/PV 两侧操作数做 TF32 RZ 截断（位模式清低 13 位）镜像。

    布局约定：q [total_q, H, D]；kv_cache [pages, Hkv, 128, 2*D]；
    topk_idx [Hkv, total_q, topk] int32；block_table [batch, max_blocks]
    int32；cu_seqlens_q/seq_lens/prefix_lens int32；sm_scale 标量 attr。
    要求 H % Hkv == 0。返回 output [total_q, H, D]（与 q 同 dtype）。
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _tf32_round(x):
        # tl.dot fp32 操作数走 TF32（保留 10 位尾数）。该 GPU 上 fp32→tf32
        # 转换为 RZ（直接截断低 13 位）——41 号算子用 RZ 在 CPU 精确复现
        # 了 GPU 实测误差（6.5465e-3 vs 6.55e-3），RNA 不符
        xi = x.contiguous().view(torch.int32)
        return (xi & ~0x1FFF).view(torch.float32)

    def forward(
        self,
        q,
        kv_cache,
        topk_idx,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        sm_scale,
    ):
        total_q, num_heads, head_dim = q.shape
        num_kv_heads = kv_cache.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        BS = 128
        group_size = num_heads // num_kv_heads
        topk = topk_idx.shape[-1]
        batch = cu_seqlens_q.numel() - 1
        max_blocks = block_table.shape[1]
        device = q.device
        is_fp32 = q.dtype == torch.float32
        # kernel host 侧 sm_scale_log2e = sm_scale * 1.4426950409（fp64 乘积
        # 以 fp32 标量入 kernel），QK 全程 log2 域
        scale2 = sm_scale * 1.4426950409

        pid = torch.arange(total_q, device=device, dtype=torch.int32)
        cu = cu_seqlens_q.to(device)
        rows = (pid[:, None] >= cu[1:][None, :]).sum(1)             # [TQ]
        delta = prefix_lens.to(device) - cu[:-1]                    # [B]
        q_abs = pid + torch.index_select(delta, 0, rows)            # [TQ]

        # kernel: valid_blocks = (q_abs + 128) // 128, real_topk = min(topk, ·)
        real_topk = torch.clamp((q_abs + BS) // BS, max=topk)       # [TQ]

        q_g = q.float().view(total_q * num_kv_heads, group_size, head_dim)
        if is_fp32:
            q_g = self._tf32_round(q_g)

        # online base-2 softmax 状态（kernel: m_i/lse_i fp32, acc fp32）
        m_i = torch.full((total_q * num_kv_heads, group_size),
                         float("-inf"), device=device)
        lse_i = torch.full((total_q * num_kv_heads, group_size),
                           float("-inf"), device=device)
        acc = torch.zeros(total_q * num_kv_heads, group_size, head_dim,
                          device=device)

        tk = topk_idx.permute(1, 0, 2)                              # [TQ,KH,tk]
        bt_flat = block_table.to(device).reshape(-1)
        kv_flat = kv_cache.view(-1, BS * 2 * head_dim)
        col_kh = torch.arange(num_kv_heads, device=device, dtype=torch.int32)
        col_bs = torch.arange(BS, device=device, dtype=torch.int32)
        q_abs_r = q_abs.repeat_interleave(num_kv_heads)             # [TQ*KH]

        # 逐槽位迭代（kernel 顺序读前 real_topk 槽；无效槽喂全 -inf 行，
        # 更新为精确空操作）
        for j in range(topk):
            slot_ok = j < real_topk                                 # [TQ]
            blk = tk[:, :, j].clamp(min=0).long()                   # [TQ,KH]
            pages = torch.index_select(
                bt_flat, 0, (rows[:, None] * max_blocks + blk).reshape(-1)
            ).view(total_q, num_kv_heads)
            kv_idx = (pages.long() * num_kv_heads
                      + col_kh[None, :]).reshape(-1)
            kv = torch.index_select(kv_flat, 0, kv_idx).view(
                total_q * num_kv_heads, BS, 2 * head_dim).float()
            k_b = kv[..., :head_dim]
            v_b = kv[..., head_dim:]
            if is_fp32:
                k_b = self._tf32_round(k_b)

            qk = torch.bmm(q_g, k_b.transpose(1, 2)) * scale2
            pos = blk.reshape(-1)[:, None] * BS + col_bs[None, :]   # [TQ*KH,BS]
            invalid = (pos > q_abs_r[:, None]) \
                | (~slot_ok).repeat_interleave(num_kv_heads)[:, None]
            qk = qk.masked_fill(invalid[:, None, :], float("-inf"))

            m_ij = torch.maximum(m_i, qk.max(dim=-1).values)
            p = torch.nan_to_num(torch.exp2(qk - m_ij.unsqueeze(-1)),
                                 nan=0.0)
            l_ij = p.sum(dim=-1)
            alpha = torch.nan_to_num(torch.exp2(m_i - m_ij), nan=0.0)
            if is_fp32:
                p_r = self._tf32_round(p)
                v_b = self._tf32_round(v_b)
            else:
                # kernel: tl.dot(p.to(v.dtype), v) —— 未归一 P 先舍回 KV dtype
                p_r = p.to(kv_cache.dtype).float()
            acc = acc * alpha.unsqueeze(-1) + torch.bmm(p_r, v_b)
            # kernel: lse_i = m_ij + log2(exp2(lse_i - m_ij) + l_ij)
            lse_new = m_ij + torch.log2(torch.exp2(lse_i - m_ij) + l_ij)
            upd = slot_ok.repeat_interleave(num_kv_heads)[:, None]
            lse_i = torch.where(
                upd, torch.nan_to_num(lse_new, nan=float("-inf")), lse_i)
            m_i = torch.where(upd, m_ij, m_i)

        o = acc * torch.exp2(m_i - lse_i).unsqueeze(-1)
        return o.view(total_q, num_heads, head_dim).to(q.dtype)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _split_lengths(total, count):
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def _metadata(specs, case_index):
    """由 shape 确定性派生 topk_idx/block_table/cu_seqlens/seq_lens/
    prefix_lens（与 level4 _metadata 逐行同构；Generator 穿线播种）。"""
    total_q = specs["q"]["shape"][0]
    batch, max_blocks = specs["block_table"]["shape"]
    num_kv_heads, _, topk = specs["topk_idx"]["shape"]
    q_lens = _split_lengths(total_q, batch)
    capacity = max_blocks * 128
    seq_lens = []
    for request, q_len in enumerate(q_lens):
        span = max(1, min(97, capacity - q_len + 1))
        margin = (case_index * 17 + request * 23) % span
        seq_lens.append(max(q_len, capacity - margin))
    prefix_lens = [
        seq_len - q_len
        for seq_len, q_len in zip(seq_lens, q_lens)
    ]
    cu_seqlens = [0]
    for length in q_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)

    generator = torch.Generator()
    generator.manual_seed(1000 + case_index)
    pages = batch * max_blocks
    permutation = torch.randperm(
        pages, generator=generator, dtype=torch.int64
    ).to(torch.int32)
    block_table = permutation.reshape(batch, max_blocks)
    topk_idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32
    )
    q_offset = 0
    for request, q_len in enumerate(q_lens):
        prefix = prefix_lens[request]
        for local_query in range(q_len):
            absolute_position = prefix + local_query
            current_block = absolute_position // 128
            valid_count = min(topk, current_block + 1)
            selected = torch.arange(
                current_block,
                current_block - valid_count,
                -1,
                dtype=torch.int32,
            )
            topk_idx[:, q_offset + local_query, :valid_count] = selected
        q_offset += q_len
    return (
        topk_idx,
        block_table,
        torch.tensor(cu_seqlens, dtype=torch.int32),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(prefix_lens, dtype=torch.int32),
    )


def get_input_groups():
    def json_path():
        # 验证/性能脚本会把本文件拷贝成 {op_name}_torch.py，
        # 因此除同名 .json 外还需回退到去掉 _torch 后缀的名字。
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

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        topk_idx, block_table, cu_seqlens, seq_lens, prefix_lens = (
            _metadata(specs, case_index)
        )
        # 每 case 独立播种（与 case 顺序/子集无关），harness 负责上卡
        gen_q = torch.Generator().manual_seed(42 + case_index * 2)
        gen_kv = torch.Generator().manual_seed(43 + case_index * 2)
        q = (torch.randn(tuple(specs["q"]["shape"]), generator=gen_q,
                         dtype=torch.float32) * 0.25).to(
            _DTYPE_MAP[specs["q"]["dtype"]])
        kv_cache = (torch.randn(tuple(specs["kv_cache"]["shape"]),
                                generator=gen_kv,
                                dtype=torch.float32) * 0.25).to(
            _DTYPE_MAP[specs["kv_cache"]["dtype"]])
        input_groups.append([
            q,
            kv_cache,
            topk_idx,
            block_table,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            specs["sm_scale"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
