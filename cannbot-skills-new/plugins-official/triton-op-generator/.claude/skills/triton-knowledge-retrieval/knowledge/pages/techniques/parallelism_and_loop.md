---
title: 并行与循环设计
category: techniques
tags: [grid-stride, persistent-kernel, loop-design, chunkwise, tiling-loop, multi-core, load-balancing]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# 并行与循环设计

> 本页从**优化维度**角度汇总与算子类型无关的并行策略和循环设计模式，覆盖多条经验中的共性技术。

---

## 1. Grid 设计策略

### 1.1 固定 Grid + Grid-Stride Loop（Persistent Kernel）

**问题**：总 block 数 >> 物理核数，导致多轮调度开销；或 block 数 > 65535 硬件上限。

**方案**：固定 grid = NUM_CORES，kernel 内用 grid-stride loop 处理所有任务：

```python
@triton.jit
def persistent_kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    NUM_CORES = tl.num_programs(0)   # = 实际启动的核数
    pid = tl.program_id(0)

    # grid-stride：每个 core 处理多个 block
    for block_id in range(pid, tl.cdiv(n, BLOCK_SIZE), NUM_CORES):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(ptr + offs, mask=offs < n)
        tl.store(ptr + offs, x * 2, mask=offs < n)
```

**适用场景**：
- 总 block 数 > 65535（必须使用）
- 核启动开销显著（RMSNorm 大 batch、Normalization n_rows >> 核数）
- 任务量不均匀时配合 work-stealing

经验：`exp_ascend_grid_large_shape_001`、`exp_rmsnorm_001`、`exp_gs_bin_005`

---

### 1.2 动态 Grid 大小（按实际任务数）

当任务量可提前计算时，精确匹配 grid 减少空转：

```python
# 根据实际数据量计算 grid，避免过大 padding
n_blocks = triton.cdiv(total_tokens, BLOCK_SIZE)
grid = (min(n_blocks, NUM_CORES),)   # 不超 65535，不浪费
kernel[grid](...)
```

经验：`exp_ascend_grid_001`

---

### 1.3 2D/3D Grid 多维并行

对 batch × seq_len × head 等多维并行场景，使用多维 Grid 避免手动 flat indexing：

```python
# 2D grid: (batch * seq_blocks, num_heads)
grid = (batch * triton.cdiv(seq_len, BLOCK_T), num_heads)
@triton.jit
def kernel(...):
    seq_block_id = tl.program_id(0)
    head_id      = tl.program_id(1)
    batch_id     = seq_block_id // n_seq_blocks
    seq_id       = seq_block_id %  n_seq_blocks
```

**约束**：各维度乘积 ≤ 65535，超出时退化为 1D + grid-stride。

---

## 2. 任务分块与循环设计

### 2.1 For-Loop Tiling（ILP 流水线使能）

将大 tile 拆成循环小 tile，使编译器能双缓冲流水：

```python
# 无流水：顺序 load → compute → store
x = tl.load(ptr + offs)
y = compute(x)
tl.store(out + offs, y)

# 有流水：编译器可重叠相邻迭代的 load 与 compute
for i in tl.static_range(NUM_TILES):
    offs = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(ptr + offs, mask=offs < n)
    y = compute(x)
    tl.store(out + offs, y, mask=offs < n)
```

**核心差异**：`tl.static_range` vs `range`
- `tl.static_range`：编译时展开，支持常量传播和多缓冲
- `range`：运行时循环，编译器无法静态分析迭代次数

经验：`exp_perf_004`、`exp_elem_001`

---

### 2.2 Chunkwise 分块（SSM/长序列）

对序列长度远大于 UB 容量的场景，按 chunk 分段处理，块间传递状态：

```python
@triton.jit
def chunk_fwd_kernel(..., CHUNK_SIZE: tl.constexpr):
    chunk_id = tl.program_id(0)
    # intra-chunk：在 UB 内用 GEMM 处理
    chunk_start = chunk_id * CHUNK_SIZE
    x_chunk = tl.load(x_ptr + chunk_start + tl.arange(0, CHUNK_SIZE))
    h_chunk = compute_intra_chunk(x_chunk)     # Cube GEMM

    # inter-chunk：传递状态（隐藏 state）给下一 chunk
    h_prev = tl.load(state_ptr + ...)          # 前一 chunk 的状态
    h_new = update_state(h_prev, h_chunk)
    tl.store(state_ptr + ..., h_new)
```

经验：`exp_chunk_fwd_001`（SSM chunkwise）

---

### 2.3 对角分块（Matmul 负载均衡）

标准行优先分块导致 L2 cache 热点和核间不均衡；对角分块让相邻 block 访问 A/B 矩阵不同位置，提升 L2 复用：

```python
# 对角 block_id → (m_id, n_id) 映射
@triton.jit
def matmul_diagonal_kernel(..., GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

经验：`exp_matmul_001`（对角 tiling）；详见 [techniques/core_mapping.md](core_mapping.md)

---

### 2.4 Work-Stealing 动态负载均衡

对任务量不均匀的场景（如 MoE Expert 接收 token 数差异大），用原子计数器动态分配：

```python
@triton.jit
def work_stealing_kernel(work_queue_ptr, ...):
    while True:
        task_id = tl.atomic_add(work_queue_ptr, 1)  # 原子认领任务
        if task_id >= num_tasks:
            break
        process_task(task_id)
```

经验：`exp_gs_bin_005`

---

### 2.5 1D Grid + 3D 任务分解（异构多维度 Tiling）

当两个维度都需要 tile 化且 tile 大小异构时（如 D 维度用小 tile保证连续性，T 维度用大 tile 减少调度开销），将多维任务展平为 1D grid + strided loop：

```python
# 3D 任务空间: batch × D-tiles × T-tiles
# D_CHK_SIZE=16（小 tile），T_CHK_SIZE=256（大 tile）
# 展平为 1D grid，避免 2D grid 维度积超限
pid = tl.program_id(0)
total_tasks = batch * NUM_D_CHK * NUM_T_CHK

for task_id in tl.range(pid, total_tasks, tl.num_programs(0)):
    bi = task_id // (NUM_D_CHK * NUM_T_CHK)         # batch 索引
    di = (task_id // NUM_T_CHK) % NUM_D_CHK         # D-tile 索引
    ti = task_id % NUM_T_CHK                         # T-tile 索引
    # 处理 (bi, di, ti) tile
```

**适用场景**：
- 短序列 + 大特征维度（如 Decode 阶段 conv1d：seq_len=1 但 D=4096）
- D 维度 tile 必须小（避免跨步访问）+ T 维度 tile 必须大（减少 tile 数）
- 二维 grid `(batch×NUM_D_CHK, NUM_T_CHK)` 维度积可能超 65535

**与 2D Grid 的对比**：

| 维度 | 2D Grid | 1D + 3D 分解 |
|------|---------|-------------|
| grid 维度 | `(batch×D_blocks, T_blocks)` | `(min(total_tasks, NUM_CORES),)` |
| 核利用率 | D_blocks 或 T_blocks 不能被核数整除时空转 | grid-stride 自动均衡 |
| 异构 tile 支持 | D 和 T 的 tile 数强制绑定到 grid 维度 | D 和 T 独立切分，自由度高 |
| 超 65535 处理 | 需手动折叠维度 | grid-stride 天然支持 |

经验：`exp_causal_conv1d_update_007`；详见 [ops/convolution.md](../ops/convolution.md)

---

### 2.6 2D Grid + 无 T-tiling（寄存器滑动窗口模式）

当序列维度通过**寄存器滑动窗口**处理（而非 T 维度 tile 化）时，Grid 仅需覆盖 batch 和 D 维度，T 维度在 kernel 内部顺序遍历：

```python
# Grid: (batch, D-tiles)，无 T 维度切分
grid = (batch, triton.cdiv(dim, BLOCK_D))

@triton.jit
def kernel(...):
    pid_batch = tl.program_id(0)
    pid_d = tl.program_id(1)
    # T 维度在 kernel 内顺序处理（寄存器滑动窗口）
    for t in range(seq_len):
        # 寄存器移位 + 单次 load 的窗口滑动
        ...
```

**与 T-tiling 的对比**：

| 维度 | 2D Grid + 无 T-tiling | 1D Grid + T-tiling |
|------|----------------------|---------------------|
| T 维度处理 | 寄存器滑动窗口（逐元素） | tile 化 + 重叠区 |
| Grid 维度 | `(batch, D-tiles)` | `(min(total_tasks, NUM_CORES),)` |
| 适用 seq_len | 中长序列（≥64，摊还寄存器初始化开销） | 短序列（1-16，T-tiling 减少调度开销） |
| 跨 tile 状态 | 无需（无 T 边界） | 需要 W-1 重叠区 |
| 核心优势 | 消除 T 维度 tile 边界开销 | 减少短序列调度开销 |

**选择决策**：seq_len ≥ 64 且 W ≤ 8 时优先用寄存器滑动窗口（无 T-tiling）；seq_len 小或 W 大时回退到 T-tiling + 重叠区。

经验：`exp_causal_conv1d_prefill_003`；详见 [ops/convolution.md](../ops/convolution.md)

---

## 3. 外层循环 → Grid 维度：并行度与局部性权衡

当 kernel 将某个维度（如时间 chunk `i_t`）放在循环内串行处理时，可考虑将其提升为 grid 维度以获得额外并行度。

```python
# 原始：2D grid，i_t 在循环内串行
grid = (NK * NC, B * H)
for i_t_chunk in range(NT):
    ...

# 可选：3D grid，i_t 并行
grid = (NT, NK * NC, B * H)
i_t = tl.program_id(0)
```

**优势**：
- NT 个 chunk 并行执行，增加并行度
- 消除循环控制开销
- 减少单个 program 的寄存器压力（不再需要跨 chunk 保持累加器）

**劣势**：
- program 数量增加 NT 倍，可能增加 launch overhead
- 每个 program 只处理一个 chunk，无法利用 chunk 间数据局部性
- 对短序列（NT=1）无收益

**决策原则**：
- NT 较大且序列较长 → 优先 grid 并行化
- NT=1 或 chunk 间有强数据依赖/共享状态 → 保持循环
- 可尝试混合策略：将 NT 拆为"并行组"和"组内循环"

经验：`exp_kda_bwd_wy_structure_opt_001`（KDA intra kernel 优化备选）

---

## 4. 并行模式选择指南

```
任务总量 ≤ 65535 block？
  ├── 是 → 直接 1D/2D/3D grid，grid = (n_blocks,) 或多维
  └── 否 → 必须 grid-stride loop，grid = (NUM_CORES,)

任务量是否均匀？
  ├── 均匀 → 静态分配（grid-stride 或精确 grid）
  └── 不均匀（如 MoE Expert） → work-stealing 动态均衡

循环层次？
  ├── 单 tile → 无需循环，直接 load/compute/store
  ├── 多 tile（ILP）→ tl.static_range 固定次数循环
  ├── 变长（可变 N）→ range(tl.cdiv(N, BLOCK)) 动态循环
  └── 序列状态传递 → chunkwise（块间状态持久化）

多维数据（batch × seq × head）？
  ├── 全维 ≤ 65535 → 2D/3D grid 自然并行
  ├── 超出或需异构 tile → 1D grid + 3D 任务分解（§2.5）
  ├── 寄存器滑动窗口（W≤8, seq_len≥64）→ 2D grid + 无 T-tiling（§2.6）
  ├── 有跨 tile 状态依赖（滑动窗口）→ 重叠 T-tiling（§8 of memory_access_optimization）
  └── 外层循环维度可并行化 → 提升为 grid 维度（§3）
```

---

## 5. 硬件绑定规则（Ascend 特有）

- Grid 最多 3 维，各维度乘积 ≤ 65535（详见 [hardware/ascend_patterns.md](../hardware/ascend_patterns.md)）
- `NUM_CORES = tl.constexpr(24)`（910b2 向量核数），动态获取用 `tl.num_programs(0)`（在 persistent kernel 中）
- 禁止 `tl.swizzle2d`（Ascend 不支持）；用对角分块代替
- `TRITON_ALL_BLOCKS_PARALLEL=1` 可超过 65535，但**禁止在生产代码中依赖**

---

## 相关页面

- [techniques/persistent_kernels.md](persistent_kernels.md) — Persistent kernel 详细设计与 dl.async_task
- [techniques/ilp.md](ilp.md) — For-loop tiling 与多缓冲流水线
- [techniques/core_mapping.md](core_mapping.md) — 对角 tiling 的 block_id 映射细节
- [hardware/ascend_patterns.md](../hardware/ascend_patterns.md) — Grid 配置约束与核心数获取
- [ops/normalization.md](../ops/normalization.md) — Persistent RMSNorm 应用
- [ops/gather_scatter.md](../ops/gather_scatter.md) — Work-stealing MoE dispatch
- [ops/ssm.md](../ops/ssm.md) — Chunkwise 分块 SSM
- [ops/convolution.md](../ops/convolution.md) — 1D grid + 3D 任务分解 (Update kernel 实例)；2D grid + 无 T-tiling (Prefill kernel 实例)
- [ops/solve_tril.md](../ops/solve_tril.md) — 2D grid `(NT, B×H)` 完全并行 + 1D grid 变长 chunk_indices
- [techniques/memory_access_optimization.md](memory_access_optimization.md) — 重叠 T-tiling 跨 tile 状态连续性（§8）
- [techniques/compute_and_branch.md](compute_and_branch.md) — Phase split、barrier 外提等 kernel 结构优化
