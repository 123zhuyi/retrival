"""示例 flops.py —— 7_SparseFlashAttention（稀疏 + 变长 + rope）。

这是本 skill 实测用过的真实文件（910B2C，72 case），示范 MHA 示例覆盖不到的三件事：

1. **变长下 score_elems 要逐 token 累加。** 不能用 padding 后的 shape，要按
   actual_seq_lengths_query/kv 的 cumsum 定位每个 query token 属于哪个 batch，
   再取 min(topk * block_size, 该 batch 的 KV 长度)。
2. **rope 拼进 head_dim。** 参考实现里 q/k 与 rope 是 torch.cat 到最后一维的，
   所以 QK^T 的 K 维是 d_main + d_rope，不是 d_main。
3. **无 FLOPs 的段用 0 占位。** gather / combine / layout transpose 不产生语义
   FLOPs，但它们的耗时必须计入总时间。给 flops_parts 一个 0 值条目，
   mfu_decompose.py 就会单独列出这些段的耗时——本例正是靠这个发现
   case 9 的 gather 吃掉了 50.6% 的时间。

语义（model.py Model.forward）：
    1) 按 sparse_indices 从 KV 中 gather 出实际参与计算的 KV 行
    2) Q @ K^T （含 rope 拼接后的完整 head_dim）
    3) softmax / mask / scale
    4) P @ V

按 operator_mfu 口径，分子只计语义必需的核心 GEMM：
    FLOPs = 2 * score_elems * (d_main + d_rope) + 2 * score_elems * d_v

gather / combine / softmax / mask 等 memory-bound 或 elementwise 段不计入分子。

已知近似：sparse_mode=3 是 right-down causal，本文件未按 threshold 逐 token 缩减
score_elems，因为 sparse_indices 的值在运行时才生成，flops.py 只能看到 shape/attr。
实测形态下 t1 << s2（如 128 vs 8192），过计约 0.8%，对结论无影响；
若换成 t1 接近 s2 的形态，需要改成按 threshold 精确缩减。
"""


def _specs(case):
    return {item["name"]: item for item in case["inputs"]}


def _cumsum(lengths, total):
    """与参考实现 _to_cumsum 等价，返回标准 cumsum。"""
    if lengths is None or len(lengths) == 0:
        return [total]
    if lengths[-1] >= total * 0.9:
        return list(lengths)
    return [sum(lengths[: i + 1]) for i in range(len(lengths))]


def _batch_of(cumsum_list, t_idx):
    """query token t_idx 属于哪个 batch（参考实现 _get_tnd_idx 的 batch 部分）。"""
    b_idx = 0
    while b_idx < len(cumsum_list) and t_idx >= cumsum_list[b_idx]:
        b_idx += 1
    # 超出最后一个 batch 时归到最后一个 batch（与参考实现对边界处理一致）
    return min(b_idx, len(cumsum_list) - 1)


def _dims(case):
    s = _specs(case)
    q = s["query"]
    k = s["key"]
    v = s["value"]
    si = s["sparse_indices"]
    qr = s.get("query_rope")
    kr = s.get("key_rope")

    layout_query = s["layout_query"]["value"]
    layout_kv = s["layout_kv"]["value"]
    block_size = s["sparse_block_size"]["value"]

    if layout_query == "BSND":
        b, s1, n1, d_main = q["shape"]
        t1 = b * s1
        n2 = k["shape"][2] if layout_kv != "TND" else k["shape"][1]
    else:  # TND
        t1, n1, d_main = q["shape"]
        n2 = k["shape"][1] if layout_kv == "TND" else k["shape"][2]

    g = n1 // n2
    d_v = v["shape"][-1]
    topk = si["shape"][-1]

    has_rope = qr is not None and kr is not None
    d_rope = qr["shape"][-1] if has_rope else 0

    asq = s.get("actual_seq_lengths_query", {}).get("value") or [t1]
    if layout_kv == "PA_BSND":
        total_kv = k["shape"][0] * k["shape"][1]
    elif layout_kv == "BSND":
        total_kv = b * k["shape"][1]
    else:
        total_kv = k["shape"][0]
    ask = s.get("actual_seq_lengths_kv", {}).get("value") or [total_kv]

    return t1, n2, g, d_main, d_rope, d_v, topk, block_size, asq, ask


def flops_parts(case) -> dict:
    """按 kernel 类型拆分理论 FLOPs：只有 attention 核心 GEMM 一段。"""
    t1, n2, g, d_main, d_rope, d_v, topk, block_size, asq, ask = _dims(case)

    asq_cumsum = _cumsum(asq, t1)
    ask_cumsum = _cumsum(ask, sum(ask) if ask else 0)

    # 每个 query token 实际选中的 KV 行数
    total_sel = 0
    for i in range(t1):
        b_idx = _batch_of(asq_cumsum, i)
        cur_s2 = ask_cumsum[b_idx] - (ask_cumsum[b_idx - 1] if b_idx > 0 else 0)
        total_sel += min(topk * block_size, cur_s2)

    score_elems = total_sel * n2 * g

    flops = 2.0 * score_elems * (d_main + d_rope + d_v)
    # gather / combine / layout transpose 无语义 FLOPs，单列为 0 以便 mfu_decompose 显示耗时
    return {"attn": flops, "gather": 0.0, "combine": 0.0, "layout": 0.0}


def flops(case) -> float:
    return float(sum(flops_parts(case).values()))


# kernel 名 → flops_parts 段名。
# 下面的 kernel 名取自该实现 msprof task_time.csv 的 kernel_name 列，换算子必须改。
# 不确定时先跑一次 mfu_decompose.py，未映射的 kernel 会以 [warn] 列出来。
KERNEL_PARTS = {
    # attention 核心 GEMM（QK^T + PV）
    "_sfa_gathered_attn_kernel": "attn",
    "_sfa_split_attn_kernel": "attn",
    "_sfa_spec_attn_kernel": "attn",
    # 非 MFU 分子段，仅用于展示耗时
    "_sfa_gather_kv_kernel": "gather",
    "_sfa_split_combine_kernel": "combine",
    # BSND layout 转换引入的 transpose/copy
    "aclnnInplaceCopy_TransposeAiCore_Transpose": "layout",
}
