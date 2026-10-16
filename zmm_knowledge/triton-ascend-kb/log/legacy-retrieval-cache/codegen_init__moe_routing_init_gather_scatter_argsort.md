---
query_type: codegen_init
query_keywords: [moe, routing, init, gather, scatter, argsort]
created: 2026-06-08
last_updated: 2026-06-08T09:27:00
hit_count: 1
context_summary: "MoeInitRouting 算子代码生成初始化 — 基于 expert_idx 稳定排序后重排 token 特征和索引映射"
top_units: [C1, C2, OP, A1, B_gs_1, E_moe_gather_001, T_mem_1, S_token_permute]
stale: false
---

## 检索问题（Problem Statement）

为 MoeInitRouting 算子生成高性能 Triton-Ascend kernel。算子语义：
1. 按 expert_idx 稳定升序排序，携带 row_idx
2. 构建 src_to_dst 逆向映射
3. 根据排序后的位置 gather x 的行数据

**触发上下文**：代码生成初始化，从零构建 MoE 路由初始化算子知识包
**检索范围**：gather_scatter + moe routing + argsort + index permutation

---

## 知识单元列表

### 硬件约束知识单元

知识单元 C1  [硬件约束]  UB 总用量 ≤ 192KB
  每个 program 的 tile 数据量之和不超过 192KB（Atlas 800T A2）。
  fp16: 单 tile ≤ 98304 elements；fp32: ≤ 49152 elements。
  MoeInitRouting 以 token 为粒度 gather，每个 program 加载一行 H 维数据，BLOCK_H ≤ 98304 / H。
  来源: hardware/constraints_cheatsheet.md C1

知识单元 C2  [硬件约束]  Grid 维度上界 65535
  grid 必须是 tuple，各维度乘积 ≤ 65535。超出需用 grid-stride loop。
  MoeInitRouting 的 NK = N×K 个输出位置，若 NK > 65535 需 grid-stride。
  来源: hardware/constraints_cheatsheet.md C2

知识单元 C5  [性能约束]  int64 CMP/ADD scalar fallback
  int64 的比较和加法会退化为标量路径，导致 aiv_scalar_ratio 升高。
  解决：用 int32 索引，.to(tl.float32) 再做比较/加法。
  MoeInitRouting 中 row_idx 和 expert_idx 均为 int32，天然规避此问题。
  来源: hardware/constraints_cheatsheet.md P1/P2

---

### 算子分类知识单元

知识单元 OP  [算子分类]  operator_class = "gather_scatter"
  判断依据：核心操作为 argsort + gather + index inversion + row copy
  详细页面：pages/ops/gather_scatter.md
  经验目录：raw/experiences/gather_scatter/
  注意：argsort 为全局排序操作，不适合 Triton kernel 内实现；应在 host 侧完成，kernel 负责 gather 和 scatter 部分。

---

### API 知识单元

知识单元 A1  [API用法]  tl.load / tl.store
  用于从 HBM 加载 tile 到寄存器，以及写回结果。
  Ascend 特有：使用 care_padding=False 消除 MTE2→Vector 同步屏障（仅当 padding 元素被 store mask 过滤时安全）。
  MoeInitRouting 中 gather x 的行数据时，可用 care_padding=False 提升 ILP。
  API文档: raw/triton-ascend-develop-guide/triton_api/Memory_Pointer_Ops/

知识单元 A4  [API文档]  tl.where / tl.gather — 索引操作
  tl.swizzle2d 在 Ascend 上禁用。
  MoeInitRouting 通过手动索引指针计算（ptr + index[i] * stride）实现 gather。
  API文档: raw/triton-ascend-develop-guide/triton_api/Indexing_Ops/

知识单元 A10 [API文档]  tl.constexpr — 编译时常量
  仅允许在 kernel 参数声明中使用；host 侧禁止。
  使用场景：BLOCK_H, HAS_BIAS 等。
  API文档: raw/triton-ascend-develop-guide/triton_api/triton/

---

### 瓶颈预判知识单元

知识单元 B_gs_1  [潜在瓶颈]  随机索引 cache miss
  触发条件：expert_idx 随机分布时，gather 的原始行号不连续
  症状: MemBW 高，但带宽利用率低（非连续访问）
  修复: argsort 后 gather 顺序已被排序，但 x 的 gather 仍按 sorted_pos % N 随机访问
  关联经验: exp_gs_mem_005
  修复方向: techniques/memory_access_optimization.md

知识单元 B_gs_2  [潜在瓶颈]  int64 索引计算
  触发条件：大 shape 时，索引乘法可能溢出 int32
  症状: aiv_scalar_ratio 升高
  修复: 使用 int64 仅在必要时，或预计算基地址
  关联经验: exp_gs_mem_003

---

### 调优经验知识单元

知识单元 E_gather_001  [调优经验]  exp_gather_001 — 向量化 gather
  适用条件：标准 gather，fp16/bf16/fp32，num_indices ≥ 32，x_size ≥ 64
  性能提升：30%
  核心优化：向量化的指针算术 + 整行加载（vectorized_row_load）+ tl.gather 内建指令
  经验文件：raw/experiences/gather_scatter/exp_gather_001.yaml
  详细页面：pages/ops/gather_scatter.md

知识单元 E_moe_gather_001  [调优经验]  exp_moe_gather_001 — MoE Gather
  适用条件：MoE Gather / Weighted Sum，topk_num ≤ 8，BLOCK_D ≥ 128
  性能提升：20%
  核心优化：向量化元数据加载（expert_id, source_token_index）+ loop unrolling（topk_num 小循环展开）+ 内存对齐
  经验文件：raw/experiences/gather_scatter/exp_moe_gather_001.yaml
  详细页面：pages/ops/gather_scatter.md

知识单元 E_moe_scatter_001  [调优经验]  exp_moe_scatter_001 — MoE Scatter 路由
  适用条件：MoE Scatter，HIDDEN_SIZE ≥ 128，支持 FP8/FP16/BF16
  性能提升：35%
  核心优化：software pipelining（num_stages=4）+ 向量化 scatter（BLOCK_E 粒度）+ 局部前缀和 + 内存对齐填充
  经验文件：raw/experiences/gather_scatter/exp_moe_scatter_001.yaml
  详细页面：pages/ops/gather_scatter.md

---

### 调优策略知识单元

知识单元 T_mem_1  [调优策略]  索引预计算与地址复用
  适用场景：gather/scatter 中重复计算索引偏移
  核心方法：预计算 src_base_addrs = indices * hidden_dim，在 hidden 循环中复用
  关键代码：
  ```python
  src_indices = tl.load(index_ptr + token_offs, mask=token_mask, other=0)
  src_base_addrs = src_indices * hidden_dim  # 预计算基地址
  for h_start in range(0, hidden_dim, BLOCK_H):
      src_addrs = src_base_addrs[:, None] + h_offs[None, :]
  ```
  页面：pages/techniques/memory_access_optimization.md

知识单元 T_mem_2  [调优策略]  care_padding=False 消除 MTE2 屏障
  适用场景：所有 gather/scatter kernel，当越界值被 mask 过滤时
  核心方法：tl.load(..., care_padding=False)
  关键代码：x = tl.load(ptr + offs, mask=mask, other=0.0, care_padding=False)
  页面：pages/techniques/ilp.md

知识单元 T_ilp_1  [调优策略]  for-loop tiling + ILP 提升
  适用场景：hidden_dim 较大，单 program 需处理多行数据
  核心方法：以 BLOCK_H 为单位循环处理 hidden_dim，增加指令级并行
  页面：pages/techniques/ilp.md

---

### 代码骨架知识单元

知识单元 S_token_permute  [代码骨架]  token_permute_kernel
  适用任务：基于 sorted_indices 的 token 重排（MoeInitRouting 核心操作）
  骨架路径：pages/skeletons/moe_dispatch.py
  关键结构：
    1. 每个 program 处理一个输出位置（token_id = tl.program_id(0)）
    2. 读取 sorted_indices[token_id] 得到原始行号
    3. 向量化加载整行 hidden_dim 数据
    4. 向量化存储到输出位置
  ```python
  @triton.jit
  def token_permute_kernel(src_ptr, dst_ptr, indices_ptr, num_tokens, hidden_dim, BLOCK_HIDDEN: tl.constexpr):
      token_id = tl.program_id(0)
      src_idx = tl.load(indices_ptr + token_id)
      hidden_offs = tl.arange(0, BLOCK_HIDDEN)
      mask = hidden_offs < hidden_dim
      src = tl.load(src_ptr + src_idx * hidden_dim + hidden_offs, mask=mask)
      tl.store(dst_ptr + token_id * hidden_dim + hidden_offs, src, mask=mask)
  ```

---

### 矛盾与注意事项

知识单元 W1  [矛盾提示]  argsort 的 Triton 实现困境
  Triton 的并行粒度是 per-program，argsort 是全局操作，需要跨 program 协调。
  当前 Triton-Ascend 不支持高效的跨 program 排序原语。
  方案：在 host 侧（forward 中）用 PyTorch 完成 argsort，kernel 只负责 gather/scatter。
  但这可能导致 PyTorch 退化判定（Type3）。
  折中：把 argsort 视为"索引准备"而非"核心计算"，核心计算（数据搬运）在 Triton kernel 中完成。

知识单元 W2  [调试建议]  int32 索引溢出检查
  当 N * K * H 较大时，指针计算可能溢出 int32。
  建议：对大 shape 使用 int64 做指针计算，但避免 int64 的比较和加法（会 scalar fallback）。
  或分步计算：base = ptr + row * stride_row，然后 base + col。
