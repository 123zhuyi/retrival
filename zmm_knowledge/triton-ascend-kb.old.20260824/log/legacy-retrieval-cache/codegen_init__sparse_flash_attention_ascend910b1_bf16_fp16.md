---
query_type: codegen_init
query_keywords: "sparse_flash_attention ascend910b1 bf16 fp16 prefill decode gather topk"
created: 2026-07-24T13:30:00
hit_count: 0
---

# 知识单元列表 — SparseFlashAttention

知识单元 C1 [硬件约束] UB 容量限制
  内容: 每个 tile 数据量 ≤ 192KB（Atlas 800T/I A2）。fp16/bf16 单 tile 最大约 98304 elements，fp32 约 49152。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C2 [硬件约束] Grid 大小上限
  内容: grid_size ≤ AI Core 总数（≤ 65535）。910b2 约 20 核，910b3 约 24 核。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C3 [硬件约束] 512B 行宽对齐
  内容: matmul 块行宽须为 512B 整数倍。fp16/bf16 下 BLOCK_N 应为 256 的倍数。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C4 [硬件约束] 禁用 tl.swizzle2d
  内容: Ascend 上 tl.swizzle2d 行为未定义，需替换为 grouped_launch_diagonal 或静态调度。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C5 [性能约束] int64/int32 CMP/ADD scalar fallback
  内容: 索引比较/加法若用 int64/int32 会产生标量回退，应 .to(tl.float32) 或避免标量控制流。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C6 [性能约束] care_padding=False
  内容: 无 care_padding=False 时 MTE2→Vector 依赖导致 ILP 损失，padding 不影响正确性时应设置。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 C7 [性能约束] fp16/bf16 accumulation 升精度
  内容: 累加/归约应在 fp32 中进行，最后 cast 回输入 dtype，防止溢出并提升精度。
  来源: knowledge/pages/hardware/constraints_cheatsheet.md

知识单元 OP1 [算子分类] operator_class = "attention"
  内容: 计算图为 softmax(Q @ K.T / sqrt(d_k)) @ V，符合 Attention 类算子。
  来源: knowledge/pages/ops/attention.md

知识单元 API1 [API用法] tl.load / tl.store
  内容: 用于 Q/K/V、sparse_indices、输出加载/存储。gather 场景下使用指针张量实现 index_select 风格加载。
  来源: knowledge/pages/ops/attention.md

知识单元 API2 [API用法] tl.dot
  内容: QK^T 与 PV 均使用 tl.dot。输入为 fp16/bf16，输出建议在 fp32 累加。
  来源: knowledge/pages/ops/attention.md

知识单元 API3 [API用法] tl.exp / tl.maximum / tl.sum
  内容: online softmax 核心原语：exp、reduce_max、reduce_sum。
  来源: knowledge/pages/ops/attention.md

知识单元 API4 [API用法] tl.where
  内容: causal mask / sparse mask 使用 tl.where 将无效位置置 -inf 或 -2e35。
  来源: knowledge/pages/ops/attention.md

知识单元 SK1 [代码骨架] Flash Attention causal GQA/MQA skeleton
  内容: grid 覆盖 (batch*head*q_block)，每个 program 负责一个 Q block，KV 循环 online softmax。
  来源: knowledge/pages/skeletons/flash_attention_causal.py

知识单元 EXP1 [调优经验] Sparse Flash Attention Prefill — 60% 提升
  内容: 使用 tle.index_select_simd / 指针 gather 加载 Top-K KV；Cube 单元做 Q@K^T；online LSE softmax；workspace tiling CACHE_BLOCK=64。
  适用条件: seq_len>=1024, topk>=32, head_dim>=64, fp16/bf16
  来源: knowledge/raw/experiences/attention/exp_sparse_flash_attn_prefill_001.yaml

知识单元 EXP2 [调优经验] Sparse Flash Attention Decode — 45% 提升
  内容: tle.index_select_simd gather Top-K KV，cube1 intrinsic Q@K^T，LSE online softmax，sub_vec_id 分布加载。
  适用条件: topk>=32, head_dim>=64, decode 场景
  来源: knowledge/raw/experiences/attention/exp_sparse_flash_attn_decode_001.yaml

知识单元 EXP3 [调优经验] Fused Attention Online Softmax — 50% 提升
  内容: 分块 online softmax，避免物化完整 S×S 分数矩阵，O(S²)→O(S) 内存。
  适用条件: S>=1024, head_dim>=64, H>=8
  来源: knowledge/pages/ops/attention.md

知识单元 EXP4 [调优经验] GQA/MQA head 复用
  内容: kv_head_id = q_head_id // (num_heads // num_kv_heads)，多个 Q head 共享 KV head。
  适用条件: GQA/MQA
  来源: knowledge/pages/ops/attention.md

知识单元 EXP5 [调优经验] Causal Mask 边界错误
  内容: mask 条件为 q_pos >= kv_pos，注意 off-by-one；sparse 场景需按原始 KV 位置计算 threshold。
  来源: knowledge/pages/ops/attention.md

知识单元 EXP6 [潜在瓶颈] 不规则 gather
  内容: 按 sparse_indices 随机 gather 易导致非连续访存，应通过 block_size 保证块内连续，或使用 index_select 风格 block load。
  来源: knowledge/raw/experiences/attention/exp_sparse_flash_attn_prefill_001.yaml

知识单元 EXP7 [潜在瓶颈] Decode 误用 Prefill kernel
  内容: S=1 时 Flash Attention 分块无收益，应走 Split-K decode。本任务 S1>=1 以 prefill 为主，但需兼顾 S1=1 小 shape。
  来源: knowledge/pages/ops/attention.md

知识单元 EXP8 [调试建议] HEAD_DIM 非 64 倍数
  内容: HEAD_DIM 需 padding 到 64 倍数，否则 Cube 效率下降。本任务 D=512，满足。
  来源: knowledge/pages/ops/attention.md
