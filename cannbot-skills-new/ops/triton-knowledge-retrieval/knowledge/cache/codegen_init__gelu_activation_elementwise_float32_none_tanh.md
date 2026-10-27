---
query_type: codegen_init
query_keywords: ['gelu', 'activation', 'elementwise', 'float32', 'none', 'tanh']
created: '2026-08-06T09:15:00Z'
last_updated: '2026-08-06T02:50:33Z'
hit_count: 2
context_summary: "为 GELU 激活函数生成高性能 Triton-Ascend kernel，支持 none(精确) 和 tanh(近似) 两种模式，dtype=float32，arch=ascend910b2"
top_units: ["C1", "C2", "OP_elementwise", "A1", "A2", "A3", "B1", "E_gelu_001", "T_autotune", "S_elementwise"]
stale: false
---

# Knowledge Cache: codegen_init for GELU (elementwise activation)

## C1-C5: 硬件约束 (Step 1)
- **C1** [硬件约束] UB 总用量 ≤ 192KB（Atlas 800T/I A2）。fp32 最大 BLOCK ≤ 49152 elements。elementwise 算子仅需单个 tile buffer，实际可用接近全 UB。
- **C2** [硬件约束] Grid 维度上界 65535。elementwise 使用 1D grid，`grid = cdiv(n_elements, BLOCK_SIZE)`。
- **C3** [硬件约束] 512B 行宽对齐仅对 matmul 类算子强制；elementwise 无此约束。
- **C4** [硬件约束] 禁止 `tl.swizzle2d`。elementwise 无需此 API。
- **C5** [性能约束] int64 CMP/ADD scalar fallback：`tl.arange()` 默认产 int64，做 mask 比较前应 `.to(tl.float32)` 避免标量降级。

## P1-P7: 性能约束 (Step 1 补充)
- **P1** [性能约束] int64 CMP/ADD → scalar fallback
- **P2** [性能约束] int32 CMP → scalar fallback
- **P3** [性能约束] `care_padding=False` 消除 MTE2→Vector 同步屏障，适用于边界 mask 已正确处理时
- **P7** [性能约束] fp16/bf16 accumulation 未升精度 → 数值问题。GELU 涉及 exp/tanh/erf 等非线性函数，**必须**在 fp32 中计算（`x.to(tl.float32)`），最后 cast 回输入 dtype。

## OP: 算子分类 (Step 2)
- **OP_elementwise** [算子分类] operator_class = "elementwise"。GELU 是逐元素激活函数，输入输出形状相同，元素间无数据依赖，天然适合向量化并行。

## A1-A5: API 知识 (Step 3)
- **A1** [API用法] `tl.load(ptr, mask=None, other=0.0)` — 从 HBM 加载数据到寄存器。elementwise 中使用 1D pointer + offsets 模式。
- **A2** [API用法] `tl.store(ptr, val, mask=None)` — 存储结果到 HBM。
- **A3** [API用法] `tl.program_id(0)` + `tl.arange(0, BLOCK_SIZE)` — 标准 1D grid 索引计算。`offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`。
- **A4** [API用法] GELU 相关数学函数：
  - `none` 模式：`tl.erf(x)` 计算 error function → `0.5 * x * (1 + erf(x / sqrt(2)))`
  - `tanh` 模式：`tl.tanh(x)` + `tl.sqrt(x)` → `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
  - 所有 math ops 需要浮点输入，不支持 int 类型。
- **A5** [API用法] `@triton.autotune(configs, key=['n_elements'])` — elementwise 标准搜索空间 `BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096}`。来源: constraints_cheatsheet.md elementwise 搜索空间。

## B1-B3: 瓶颈预判 (Step 4)
- **B1** [潜在瓶颈] Elementwise 算子受**内存带宽**限制而非计算能力。优化重点：最大化向量化访存带宽利用率。
- **B2** [潜在瓶颈] BLOCK_SIZE 过小（< 256）→ 向量单元填不满；BLOCK_SIZE 过大（> 4096 fp32）→ 可能超 UB 容量。
- **B3** [潜在瓶颈] 小张量（总元素少）→ kernel launch overhead 显著。张量 ≤ 1024 元素时考虑 persistent kernel 或直接用 PyTorch。

## E1-E5: 经验匹配 (Step 4)
- **E_gelu_001** [调优经验] exp_gelu_fwd_001 — GELU 前向向量化实现，提升 15-25%。核心技术：
  - grid-stride loop: `for row_task_id in range(pid, num_row_tasks, grid_size)` 处理任意行数
  - autotuning BLOCK_SIZE_M: 在 1-32 范围搜索行维度 tile 大小
  - 精度保持: 输入 cast 到 fp32 后再做 GELU 计算（tanh/exp 在 fp16 下容易溢出）
  - 向量化访存: 列维度用 BLOCK_SIZE_N 分块 + masked load/store
  来源: raw/experiences/activation/exp_gelu_fwd_001.yaml
- **E_elem_vec_001** [调优经验] BLOCK_SIZE=1024/2048 向量化连续 load/store，提升 30%。来源: ops/elementwise.md
- **E_elem_vec_003** [调优经验] BLOCK_SIZE autotuning {256,512,1024,2048,4096}，自适应不同规模，提升 20%。来源: ops/elementwise.md
- **E_elem_vec_004** [调优经验] `tl.load()` 设 `other=0`，`boundary_check` 减少同步开销，提升 15%。来源: ops/elementwise.md
- **E_elem_vec_005** [调优经验] For-loop tiling: 单次 load+compute+store 拆为多小 tile 循环，提升 ILP，提升 20%。来源: ops/elementwise.md

## T1-T3: 调优策略 (Step 5)
- **T_autotune** [调优策略] 推荐 autotune 搜索 BLOCK_SIZE ∈ {256, 512, 1024, 2048, 4096}。elementwise 标准配置空间。
- **T_ilp** [调优策略] For-loop tiling 提升 ILP：将大 tile 拆为 NUM_TILES 个小 tile 循环，使硬件流水线并行处理多个 tile。
- **T_precision** [调优策略] GELU 涉及 exp/tanh/erf 等非线性函数，必须 cast 到 fp32 计算，最后转回原 dtype。

## S1: 代码骨架 (Step 6)
- **S_elementwise** [代码骨架] 通用 elementwise 模板（见 ops/elementwise.md §代码模板）。核心结构：
  ```python
  @triton.autotune(configs=[...], key=['n_elements'])
  @triton.jit
  def gelu_kernel(x_ptr, y_ptr, n_elements, APPROXIMATE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
      pid = tl.program_id(0)
      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
      # GELU computation in fp32
      x_f32 = x.to(tl.float32)
      if APPROXIMATE == 'none':
          y_f32 = 0.5 * x_f32 * (1.0 + tl.erf(x_f32 / 1.4142135623730951))
      else:  # 'tanh'
          inner = 0.7978845608028654 * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
          y_f32 = 0.5 * x_f32 * (1.0 + tl.tanh(inner))
      y = y_f32.to(x.dtype)
      tl.store(y_ptr + offsets, y, mask=mask)
  ```
