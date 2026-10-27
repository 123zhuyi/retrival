"""示例 flops.py —— 20_FlashAttentionV2（完整 MHA block）。

这是本 skill 实测用过的真实文件，示范三件容易做错的事：

1. **算子语义不等于名字。** 名字叫 FlashAttentionV2，但 model.py 的 forward 里
   有 4 个 nn.Linear（q/k/v/out 投影）。投影是语义必需的核心 GEMM，必须计入，
   漏掉会让分子小一个数量级 —— 这个 case 里投影占理论 FLOPs 的 **94.7%**。
2. **causal 要缩减 score_elems。** 上三角被 mask 成 -inf，语义上不需要计算，
   官方 operator_mfu 文档同样规定 causal/sparse 模式下 score_elems 相应缩减。
3. **分子只含语义必需量。** 实现里的冗余（两趟 QK dot、对角块内被 mask 掉的
   元素、head_dim pad 到 BLOCK_Dh 的补零）一律不计 —— 冗余会拉长耗时让 MFU 下降，
   这正是 MFU 比 PMU 占空比更真实的原因。

算子语义（model.py Model.forward）：
    q/k/v = 3 × Linear(D, D, bias=False)
    scores = q @ kᵀ → scale → (causal mask) → softmax
    out    = weights @ v
    y      = out_proj(out)
"""


def _dims(case):
    specs = {item["name"]: item for item in case["inputs"]}
    batch, seq_len, d_model = specs["x"]["shape"]
    causal = bool(specs["causal"]["value"])
    return batch, seq_len, d_model, causal


def flops_parts(case) -> dict:
    """按 kernel 的缝把理论 FLOPs 拆开。各段之和 == flops(case)。"""
    batch, seq_len, d_model, causal = _dims(case)

    # 4 次 [B*S, D] @ [D, D] 投影
    proj = 4.0 * 2.0 * batch * seq_len * d_model * d_model

    # Q@Kᵀ + P@V。pairs = 单个 (b, head) 语义上必需的 (q, k) 对数
    # 因 H × head_dim = D，两次 GEMM 合并后为 4 × B × D × pairs
    pairs = seq_len * (seq_len + 1) / 2.0 if causal else float(seq_len) * seq_len
    attn = 4.0 * batch * d_model * pairs

    return {"proj": proj, "attn": attn}


def flops(case) -> float:
    """ops-profiling `--mfu` 要求的接口。"""
    return float(sum(flops_parts(case).values()))


# mfu_decompose.py 用：kernel 名 → flops_parts 的段名。
# kernel 名取自 msprof task_time.csv 的 kernel_name 列；不确定时先跑一次
# mfu_decompose.py，未映射的 kernel 会以 [warn] 列出来。
KERNEL_PARTS = {
    "linear_kernel": "proj",
    "flash_attention_kernel": "attn",
}
