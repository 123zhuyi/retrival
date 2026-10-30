"""Paged Attention forward（lmdeploy / vLLM 风格分页 KV cache 注意力）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/paged_attention.py
    paged_attention_fwd(q, k, v, o, block_offsets, q_start_loc, q_seqlens,
                        kv_seqlens, max_seqlen, window_size, sm_scale, logit_softcapping)

语义（golden，torch 原生小算子拼接）：
    q       : [total_q, Hq, D]           变长 batch 拼接后的连续 query
    k_cache : [num_blocks, block_size, Hk, D]    分页 KV cache
    v_cache : [num_blocks, block_size, Hk, Dv]
    block_offsets : [B, max_blocks]      每个 batch 的逻辑块 → 物理块映射
    q_start_loc / q_seqlens / kv_seqlens : [B]
    输出     : [total_q, Hq, Dv]

    1) 按 block_offsets 把分页 cache 还原成逐 batch 连续的 K/V：
       k_flat[b, t] = k_cache[block_offsets[b, t // block_size], t % block_size]
    2) GQA：Hq // Hk 个 query head 共享一个 kv head。
    3) 打分 scores = q · k^T * sm_scale，因果掩码为
       kv_pos > (kv_len - q_len) + q_pos → -inf
       （history_len = kv_len - q_len，与 kernel 内 causal mask 一致；
        kv_pos >= kv_len 的位置必然满足该条件，因此天然被屏蔽）。
    4) softmax 走 fp32，out = p @ v 后回铸 q.dtype。

    kernel 内 tl.dot 为 fp16/bf16 输入 + fp32 累加，本实现统一先升 fp32 再计算。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, block_offsets, q_start_loc,
                q_seqlens, kv_seqlens, max_seqlen, sm_scale):
        device = q.device
        block_size = k_cache.shape[1]
        total_q, hq, dim = q.shape
        hk, dim_v = v_cache.shape[-2], v_cache.shape[-1]
        group = hq // hk
        batch, max_blocks = block_offsets.shape
        # padding 长度直接取自形状（block_offsets 已按最长序列分配），
        # 避免 max().item() 触发额外 reduce 与主机同步；超出真实长度的位置由掩码屏蔽。
        max_kv = max_blocks * block_size
        max_q = max_seqlen

        # 逻辑块 → 物理 token 号：block_offsets * block_size + [0, block_size)
        # （只用 mul/add 广播，避免整除/取模落到 AiCpu；全程 int32）
        span = torch.arange(max_kv, device=device, dtype=torch.int32)
        token = (block_offsets * block_size).unsqueeze(-1) + span[:block_size]
        token = token.reshape(-1)

        k_pad = torch.index_select(k_cache.reshape(-1, hk, dim), 0, token)
        v_pad = torch.index_select(v_cache.reshape(-1, hk, dim_v), 0, token)
        k_pad = k_pad.to(torch.float32).reshape(batch, max_kv, hk, dim)
        v_pad = v_pad.to(torch.float32).reshape(batch, max_kv, hk, dim_v)

        # query 侧：变长 → [B, max_q, Hk, group, D]（padding 行的输出最后丢弃）
        q_pos = span[:max_q]
        q_idx = q_start_loc.unsqueeze(1) + q_pos.unsqueeze(0)
        q_pad = torch.index_select(q, 0, q_idx.reshape(-1).clamp(max=total_q - 1))
        q_pad = q_pad.to(torch.float32).reshape(batch, max_q, hk, group, dim)

        # GQA 打分：group 维直接进 einsum，避免展开 K/V
        scores = torch.einsum("bqhgd,bkhd->bhgqk", q_pad, k_pad) * sm_scale
        # 因果掩码：kv_pos > (kv_len - q_len) + q_pos，等价于屏蔽历史窗口之外与越界位置
        limit = (kv_seqlens - q_seqlens).reshape(batch, 1, 1, 1, 1) + q_pos.reshape(1, 1, 1, max_q, 1)
        scores = scores.masked_fill(span.reshape(1, 1, 1, 1, max_kv) > limit, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhgqk,bkhd->bqhgd", probs, v_pad)

        if max_q == 1:
            # decode：每条序列只有一个 query token，q_start_loc 即 batch 序，无需散射
            return out.reshape(total_q, hq, dim_v).to(q.dtype)

        # 还原成连续变长布局（越界行写入哨兵槽后丢弃）
        valid = q_pos.reshape(1, max_q) < q_seqlens.reshape(batch, 1)
        flat_idx = torch.where(valid.reshape(-1), q_idx.reshape(-1),
                               torch.full_like(q_idx.reshape(-1), total_q)).to(torch.int32)
        buf = torch.zeros(total_q + 1, hq, dim_v, device=device, dtype=torch.float32)
        buf.index_copy_(0, flat_idx.to(torch.int64), out.reshape(-1, hq, dim_v))
        return buf[:total_q].to(q.dtype)


def get_input_groups():
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    def json_path():
        # 验证/性能脚本会把本文件拷贝成 {op_name}_torch.py，
        # 因此除同名 .json 外还需回退到去掉 _torch 后缀的名字与原始 case 文件名。
        here = os.path.dirname(os.path.abspath(__file__))
        stem = os.path.splitext(os.path.basename(__file__))[0]
        names = [stem]
        if stem.endswith("_torch"):
            names.append(stem[: -len("_torch")])
        names.append("2_PagedAttention")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k 取 N(0, 1)：注意力输入来自 RMSNorm/LayerNorm 之后的激活，
        # 打分经 1/sqrt(D) 缩放后落在 N(0, 1) 量级，softmax 不饱和、注意力模式有意义。
        # v 取 U(0, 1)（与上游 test_paged_attention.py 的 torch.rand 一致）：
        # 输出是 v 的凸组合，v 若零均值会让输出集中在 0 附近，
        # 低精度输出的相对误差被数值抵消放大到与算子实现无关的程度。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(shape)
        if name == "v_cache":
            tensor = torch.empty(shape, dtype=torch.float32)
            tensor.uniform_(0.0, 1.0, generator=generator)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        return tensor.to(dtype)

    def build_block_offsets(kv_seqlens, block_size):
        # 复刻仓上测试的块分配：batch i 的第 j 个逻辑块 → 物理块 j * B + i
        batch = len(kv_seqlens)
        num_blocks = [(int(l) + block_size - 1) // block_size for l in kv_seqlens]
        offsets = torch.zeros(batch, max(num_blocks), dtype=torch.int32)
        for i, n in enumerate(num_blocks):
            offsets[i, :n] = torch.arange(n, dtype=torch.int32) * batch + i
        return offsets

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["q"]["dtype"]]
        block_size = specs["k_cache"]["shape"][1]
        q_seqlens = torch.tensor(specs["q_seqlens"]["value"], dtype=torch.int32)
        kv_seqlens = torch.tensor(specs["kv_seqlens"]["value"], dtype=torch.int32)
        q_start_loc = torch.tensor(specs["q_start_loc"]["value"], dtype=torch.int32)
        input_groups.append([
            random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 3),
            random_tensor("k_cache", specs["k_cache"]["shape"], dtype, 43 + case_index * 3),
            random_tensor("v_cache", specs["v_cache"]["shape"], dtype, 44 + case_index * 3),
            build_block_offsets(kv_seqlens.tolist(), block_size),
            q_start_loc,
            q_seqlens,
            kv_seqlens,
            specs["max_seqlen"]["value"],
            specs["sm_scale"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
