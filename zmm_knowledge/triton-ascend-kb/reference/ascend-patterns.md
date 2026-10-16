---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: reference
title: Ascend NPU 特有语法与模式
description: Ascend NPU 特有语法与模式
tags: [ascend, grid, constexpr, scalar, slice, tensor_create, parallel_pattern, 910b2, 910b3, hardware]
confidence: inferred
status: active
resource: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/hardware/ascend_patterns.md
created_at: '2026-08-10T02:56:36Z'
updated_at: '2026-08-10T02:56:36Z'
---

sources:
  - url: https://gitcode.com/zhou_huaman/cannbot-skills_5171/blob/d972095e7b5842b59302088cdaa07749eef5b901/ops/triton-knowledge-retrieval/knowledge/pages/hardware/ascend_patterns.md
    role: primary

> 迁移备注：本卡由 `cannbot-skills_5171/ops/triton-knowledge-retrieval/knowledge/pages/hardware/ascend_patterns.md` 迁移而来，原始为硬件概念页，未做内容改写，仅转 OKF frontmatter。

# Ascend NPU 特有语法与模式

> 本页汇总 Triton-Ascend 与标准 GPU Triton 的关键差异，来自 `raw/experiences/general_ascend/` 12 条经验。  
> **必读**：实现任何 kernel 前先检查本页约束，避免编译错误。

## 1. Grid 配置规范

**约束**：
- Grid 必须是 `tuple` 类型，最多 3 维
- 各维度乘积 **≤ 65535**（`x * y * z ≤ 65535`）——**默认硬件限制**
- 不能使用 4 维 Grid

> **环境变量例外**：设置 `TRITON_ALL_BLOCKS_PARALLEL=1` 时，Runtime 会尝试让所有 block 并发调度，此时 Grid 乘积可超过 65535，但仍受物理核数（910b2: 24 VEC + 24 Cube，910b3: 20+20）限制。**生产代码不应依赖此环境变量**——它会绕过硬件调度保障，可能导致 OOM 或调度死锁。若 block 数量确实需要超过 65535，使用 grid-stride loop 代替。

```python
# 正确
grid = (48,)            # 1D，乘积=48 ✓
grid = (24, 2)          # 2D，乘积=48 ✓

# 错误
grid = 48               # 不是 tuple ✗
grid = (256, 256)       # 乘积=65536 > 65535 ✗
grid = (12, 2, 2, 1)   # 4维 ✗
```

**大 shape 处理**：当 `n_blocks > 65535` 时，使用 grid-stride loop（persistent kernel）：
```python
grid = (min(n_blocks, NUM_CORES),)  # 固定 grid
# kernel 内部循环处理剩余任务
```

经验文件：exp_ascend_grid_001，exp_ascend_grid_large_shape_001

---

## 2. tl.constexpr 用法

**规则**：`tl.constexpr` 仅允许在 kernel 函数参数声明中使用，**禁止**在 host 侧使用。

```python
# 正确：kernel 参数
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr): ...

# 错误：host 侧使用
def forward():
    BLOCK_SIZE: tl.constexpr = 128  # ✗ 语法错误
    BLOCK_SIZE = 128  # ✓ host 侧直接用普通变量
```

经验文件：exp_constexpr_001

---

## 3. 禁用语法列表

| 禁止 | 替代方案 | 说明 |
|------|----------|------|
| `return` / `break` / `continue` | mask 控制执行逻辑 | Triton kernel 无提前退出 |
| `lambda` 表达式 | `tl.where` / 内联函数 | 编译不支持 |
| 链式布尔运算 `a and b and c` | 分步计算 mask 后组合 | 用 `&` 按位与代替 |
| `tensor[i]` 直接索引 | `tl.load` / `tl.store` | 必须通过内存访问 API |
| `tl.swizzle2d` | 使用 `grouped_launch_diagonal` | Ascend 不支持 |
| int64 CMP/ADD | 转换为 fp32 / int32 | 无原生 SIMD int64 指令 |
| 负数偏移 | `tl.maximum(offset, 0)` + mask | 避免负地址访问 |

经验文件：exp_api_001，exp_scalar_cast_001

---

## 4. 核心数与并行模式

**910b2 标准核心数**：
- Cube（矩阵计算）核心：通常 20/24 个
- Vector（向量计算）核心：通常 48 个
- 总 AI Core 数：通常 **20（Cube） + 48（Vector）= 68 个**，具体以 `torch_npu.npu_core_num()` 为准

```python
# 获取实际核心数
VEC_CORE_NUM = torch_npu.npu_core_num()  # 通常 48（Vector）
CUBE_CORE_NUM = 20  # 固定值

# 推荐：按核心数配置 grid
grid = (VEC_CORE_NUM,)  # 向量密集型算子
```

经验文件：exp_ascend_core_num_001，exp_ascend_parallel_pattern_001

---

## 5. 大 Shape 分块策略

当输入 shape 极大时（如超长序列、大词表），单 kernel 无法处理：

```python
# 策略：大 shape 分割为多个 sub-block，每个 program 处理一段
@triton.jit
def large_shape_kernel(ptr, n, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr):
    pid = tl.program_id(0)
    # grid-stride：每个 core 处理多个 block
    for block_id in range(pid, tl.cdiv(n, BLOCK_SIZE), tl.num_programs(0)):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        ...
```

经验文件：exp_ascend_large_shape_block_001

---

## 6. where/split/slice 操作

| 操作 | 正确写法 | 注意事项 |
|------|----------|----------|
| 条件选择 | `tl.where(cond, x, y)` | cond 必须是 mask tensor，不能是 Python bool |
| 张量切片 | `tl.load(ptr + start + tl.arange(0, size))` | 不能直接用 `[start:end]` 切片 |
| 维度分割 | 手动计算偏移，分两次 load | 无原生 split 操作 |

经验文件：exp_ascend_where_001，exp_slice_001，exp_ascend_split_001

---

## 7. Tensor 创建

| 操作 | Triton API |
|------|-----------|
| 零张量 | `tl.zeros([SIZE], dtype=tl.float32)` |
| 全填充 | `tl.full([SIZE], value, dtype=tl.float32)` |
| 整数序列 | `tl.arange(0, SIZE)` |
| 2D 广播 | `tl.arange(0, M)[:, None]` + `tl.arange(0, N)[None, :]` |

经验文件：exp_tensor_create_001

---

## 相关页面

- hardware/constraints_cheatsheet.md — 强制约束速查（UB/对齐/Grid）
- [hardware/ascend_910b.md](ascend-910b.md) — 910b2/910b3 硬件规格
- techniques/performance_tuning.md — 性能调优通用原则
- workflows/debugging.md — 调试与稳定性指南

<!-- okf:related:start -->

# 相关

- 讲解接口: [用 tl.constexpr 编译期消除运行时分支](../runbooks/compute-pipeline/constexpr-branch-elimination.md) — 主题相关：共享 constexpr
- 讲解接口: [Scheduling Optimization: hardware unit isolation](../runbooks/compute-pipeline/exp_fused_norm_rope_matmul_001.md) — 同属优化知识，主题相关
- 讲解接口: [Scheduling Optimization: hardware specific instruction](../runbooks/compute-pipeline/exp_matmul_qn_001.md) — 同属优化知识，主题相关
- 讲解接口: [Compute Optimization: constexpr dual path full vs boundary](../runbooks/compute-pipeline/exp_solve_tril_003.md) — 同属优化知识，主题相关
- 讲解接口: [算子融合模式与约束决策](../runbooks/compute-pipeline/operator-fusion-patterns.md) — 同属优化知识，主题相关
- 讲解接口: [Memory Access Optimization: make block ptr hardware acceleration](../runbooks/memory-sram/exp_causal_conv1d_update_003.md) — 同属优化知识，主题相关
- 讲解接口: [Memory Access Optimization: constexpr strides](../runbooks/memory-sram/exp_causal_conv1d_varlen_opt_001.md) — 同属优化知识，主题相关
- 讲解接口: [Memory Access Optimization: preload extract slice eliminate hbm read](../runbooks/memory-sram/exp_solve_tril_002.md) — 同属优化知识，主题相关
- 讲解接口: [BLOCK 向上取 2 的幂并声明为 constexpr](../runbooks/memory-sram/pow2-block-constexpr.md) — 主题相关：共享 constexpr
- 讲解接口: [preload + extract_slice 消除循环内同一数据重复 HBM 读取](../runbooks/memory-sram/preload-extract-slice-eliminate-reread.md) — 同属优化知识，主题相关
- 讲解接口: [负优化排除清单：微小矩阵禁 tl.dot、constexpr 展开优先、preload 按 grid 规模决策](../runbooks/meta-skills/exp_solve_tril_006.md) — 同属优化知识，主题相关
- 讲解接口: [关键长度参数必须 tl.constexpr 静态化触发启动级特化](../runbooks/portability/constexpr-launch-specialization.md) — 主题相关：共享 constexpr
- 讲解接口: [静态上限循环：while 改 for range（上限须为 tl.constexpr）](../runbooks/portability/exp_api_003.md) — 同属优化知识，主题相关
- 讲解接口: [tl.constexpr 仅限内核参数声明，Host 侧禁用](../runbooks/portability/exp_constexpr_001.md) — 同属优化知识，主题相关
- 相关主题: [Ascend 910B2/910B3 Hardware Reference](ascend-910b.md) — 主题相关：共享 910b2, 910b3, ascend, grid, hardware

<!-- okf:related:end -->
