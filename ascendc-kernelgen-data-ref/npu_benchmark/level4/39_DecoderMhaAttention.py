"""Decoder MHA Attention（SGLang 风格 decode 阶段注意力，split-KV 两段式）.

算子来源：triton-ascend-kernels/src/triton_ascend_kernels/attention/decoder_mha_attention.py
    decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices,
                         attn_logits, attn_lse, num_kv_splits, max_kv_splits, sm_scale)

语义（golden，torch 原生小算子拼接）：
    q         : [B, H, D]      每条序列一个新 token 的 query
    k_buffer  : [P, H, D]      全局 KV pool（分页/连续均可，按 token 索引）
    v_buffer  : [P, H, Dv]
    kv_indptr : [B+1] int32    每条序列在 kv_indices 上的起止
    kv_indices: [nnz] int32    每条序列命中的 KV pool token 号
    max_kv_len: int            批内最长 KV 长度（形状提示，等价于 paged attention 的
                               max_seqlen；kernel 侧不需要，仅供 golden 免去一次
                               device→host 的 max 同步）
    输出       : [B, H, Dv]

    对每条序列 b，取 idx = kv_indices[kv_indptr[b] : kv_indptr[b+1]]：
        scores = (q[b] · k_buffer[idx]) * sm_scale     # [H, L_b]
        out[b] = softmax(scores) @ v_buffer[idx]       # [H, Dv]

    kernel 走 split-KV：先分段算局部 softmax 与 lse，再在 stage2 归并，
    数学上等价于一次全局 softmax；本实现直接算全局 softmax，全程 fp32 后回铸 q.dtype。
    该 kernel 仅支持 MHA（H_q == H_kv）。
"""
import json
import os

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_buffer, v_buffer, kv_indptr, kv_indices, max_kv_len, sm_scale):
        device = q.device
        batch, heads, dim = q.shape
        dim_v = v_buffer.shape[-1]
        total = kv_indices.shape[0]

        if total == batch * max_kv_len:
            # 等长批：sum(len) == B * max(len) 只可能各条序列等长，
            # kv_indices 本身即 [B, max_kv_len] 连续布局，无需 padding 与掩码
            token = kv_indices
            valid = None
        else:
            pos = torch.arange(max_kv_len, device=device, dtype=torch.int32)
            lengths = kv_indptr[1:] - kv_indptr[:-1]
            valid = pos.reshape(1, max_kv_len) < lengths.reshape(batch, 1)
            gather = (kv_indptr[:-1].reshape(batch, 1) + pos.reshape(1, max_kv_len))
            token = torch.index_select(kv_indices, 0, gather.reshape(-1).clamp(max=total - 1))

        k = torch.index_select(k_buffer, 0, token.reshape(-1)).to(torch.float32)
        v = torch.index_select(v_buffer, 0, token.reshape(-1)).to(torch.float32)
        k = k.reshape(batch, max_kv_len, heads, dim)
        v = v.reshape(batch, max_kv_len, heads, dim_v)

        scores = torch.einsum("bhd,bkhd->bhk", q.to(torch.float32), k) * sm_scale
        if valid is not None:
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhk,bkhd->bhd", probs, v)
        return out.to(q.dtype)


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
        names.append("4_DecoderMhaAttention")
        for name in names:
            path = os.path.join(here, name + ".json")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"case 描述文件未找到: {names} @ {here}")

    def random_tensor(name, shape, dtype, seed):
        # q / k 取 N(0, 1)：来自 RMSNorm 之后的激活；
        # v 取 U(0, 1)：输出是 v 的凸组合，v 若零均值会让输出集中在 0 附近，
        # 低精度输出的相对误差被数值抵消放大到与算子实现无关的程度。
        generator = torch.Generator()
        generator.manual_seed(seed)
        shape = tuple(shape)
        if name == "v_buffer":
            tensor = torch.empty(shape, dtype=torch.float32)
            tensor.uniform_(0.0, 1.0, generator=generator)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
        return tensor.to(dtype)

    def build_kv_indices(kv_lens, pool_size, seed):
        # KV pool 中的 token 号：模拟 paged 分配后被打散的槽位（非连续），
        # 与 SGLang 的 req_to_token 表一致。
        generator = torch.Generator()
        generator.manual_seed(seed)
        total = int(sum(kv_lens))
        perm = torch.randperm(pool_size, generator=generator)[:total]
        return perm.to(torch.int32)

    with open(json_path(), "r", encoding="utf-8-sig") as file:
        cases = [json.loads(line) for line in file if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        specs = {item["name"]: item for item in case["inputs"]}
        dtype = dtype_map[specs["q"]["dtype"]]
        kv_indptr = torch.tensor(specs["kv_indptr"]["value"], dtype=torch.int32)
        kv_lens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
        pool_size = specs["k_buffer"]["shape"][0]
        input_groups.append([
            random_tensor("q", specs["q"]["shape"], dtype, 42 + case_index * 3),
            random_tensor("k_buffer", specs["k_buffer"]["shape"], dtype, 43 + case_index * 3),
            random_tensor("v_buffer", specs["v_buffer"]["shape"], dtype, 44 + case_index * 3),
            kv_indptr,
            build_kv_indices(kv_lens, pool_size, 45 + case_index * 3),
            specs["max_kv_len"]["value"],
            specs["sm_scale"]["value"],
        ])
    return input_groups


def get_init_inputs():
    return []
