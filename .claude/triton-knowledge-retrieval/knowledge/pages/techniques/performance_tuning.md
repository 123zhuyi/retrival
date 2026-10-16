---
title: 性能调优指南（通用）
category: techniques
tags: [block-size, autotuning, performance, stride-access, grid-stride, persistent]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# 性能调优指南（通用）

> 本页汇总与算子类型无关的通用性能优化原则，来自 `raw/experiences/general_perf/` 9 条经验。

## 1. 块大小（BLOCK_SIZE）选择

**基本原则**：块大小需平衡硬件并行度与资源占用。

| 算子类型 | 推荐 BLOCK_SIZE | 说明 |
|---------|----------------|------|
| 元素级（elementwise） | 1024 / 2048 | 平衡并行度与 UB 占用 |
| 矩阵乘（GEMM） | BLOCK_M/N=256, BLOCK_K=64-128 | 适配 Cube 核心计算粒度 |
| 归约（reduction） | 256-1024 | 按归约维度大小调整 |
| Normalization | 256-512（列维度） | 列维度过大需分块 |

```python
# 推荐：使用 autotune 自动搜索
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': bs}) for bs in [256, 512, 1024, 2048, 4096]],
    key=['n_elements'],
)
@triton.jit
def kernel(ptr, n, BLOCK_SIZE: tl.constexpr): ...
```

**避坑**：
- 块过小 → 并行度不足，无法充分利用硬件算力
- 块过大 → 超出 UB（192KB），触发编译/运行失败

经验文件：[exp_perf_001](../../raw/experiences/general_perf/exp_perf_001.yaml)

---

## 2. 内存访问模式优化

### 2.1 连续访问优先

随机/stride 访问会导致内存访问不连续，缓存命中率低 50%+：

```python
# 反例：stride 访问（累积偏移开销大，缓存命中低）
offsets = rm * stride_m + rn  # 每次计算多维偏移

# 正例：连续一维访问
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 连续内存
```

**规则**：除非必须（非规则张量布局），优先连续化后一维遍历。  
经验文件：[exp_perf_007](../../raw/experiences/general_perf/exp_perf_007.yaml)

### 2.2 2D block_ptr 替代 1D 偏移

对矩阵操作，使用 `tl.make_block_ptr` 替代手动计算 2D 偏移，提升内存访问效率：

```python
# 推荐
block_ptr = tl.make_block_ptr(base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                               offsets=(m_start, k_start), block_shape=(BLOCK_M, BLOCK_K),
                               order=(1, 0))
a = tl.load(block_ptr)
```

经验文件：[exp_perf_002](../../raw/experiences/general_perf/exp_perf_002.yaml)

### 2.3 内存带宽瓶颈识别

当 PipeUtil 低、MemBW 高时：
- 检查是否有多余的 HBM 写回（中间 tensor 可消除）
- 增大 BLOCK_SIZE 提升单次 IO 效率
- 考虑算子融合消除中间 tensor

经验文件：[exp_perf_003](../../raw/experiences/general_perf/exp_perf_003.yaml)

---

## 3. 指令级并行（ILP）

使用 For-loop tiling 将单次大 tile 拆为多个小 tile 循环，使硬件流水线同时处理多个 tile 的不同阶段：

```python
@triton.jit
def ilp_kernel(ptr, n, BLOCK_SIZE: tl.constexpr, NUM_TILES: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE * NUM_TILES
    for i in tl.static_range(NUM_TILES):
        offs = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        data = tl.load(ptr + offs, mask=offs < n)
        tl.store(ptr + offs, data * 2, mask=offs < n)
```

经验文件：[exp_perf_004](../../raw/experiences/general_perf/exp_perf_004.yaml)

---

## 4. care_padding=False 优化

当 N 不是 BLOCK_SIZE 整数倍时，最后一个 tile 含 padding 元素。  
若 padding 不影响计算结果（如 padding 元素的输出被 mask 忽略），可省略 boundary check 减少 MTE2/Vector 同步屏障：

```python
# 有 boundary check（默认，稍慢）
x = tl.load(ptr + offsets, mask=offsets < n, other=0.0)

# 省略 boundary check（仅在 padding 安全时使用）
x = tl.load(ptr + offsets, mask=offsets < n)  # 不传 other
```

经验文件：[exp_perf_006](../../raw/experiences/general_perf/exp_perf_006.yaml)

---

## 5. 性能调优流程

```
1. Profiling → 识别瓶颈（MemBW? CubeUtil? PipeUtil?）
2. MemBW 瓶颈 → 增大 BLOCK_SIZE / 算子融合消中间 tensor / 连续化访问
3. CubeUtil 低 → 检查 BLOCK_M/N/K 是否适配 Cube（256 的倍数）
4. PipeUtil 低 → ILP（For-loop tiling）/ persistent kernel
5. launch overhead → persistent kernel（固定 grid，grid-stride 循环）
6. 编译失败 → 检查 UB 是否超限（192KB）、Grid 乘积是否 ≤65535
```

---

## 6. autotune 配置空间扩展

单一 autotune 配置往往无法覆盖所有 shape/register pressure 下的最优解。Ascend 上尤其需要探索 `num_warps` 和 `num_stages`：

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=4),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['BK', 'NC', 'BT'],
)
```

**经验值**：
- 标量占比高的 kernel：更多 warp（8）可隐藏标量延迟
- 寄存器压力大的 kernel：高 warp 可能导致 spill，需实测
- 内存瓶颈的 kernel：`num_stages=2/3` 软流水通常最优

经验：`exp_perf_017`

---

## 7. KDA / Linear Attention 反向调优实例

`chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2` 在 Ascend 上的调优路径：

| 轮次 | 优化 | Task Duration | 关键发现 |
|------|------|---------------|----------|
| R1 | BK/BV autotune + barrier 外提 + other=0.0 | 127,168 μs | 更大 tile + 更少 barrier |
| R2 | Phase split + h/dh 手动指针 | 91,260 μs | block_ptr 对 h/dh 开销大 |
| R3 | 全部手动指针 | 90,050 μs | 消除 block_ptr 开销 |
| R5 | 移除 b_dv compile_hint | 67,286 μs | compile_hint 导致巨大开销 |
| R6 | num_warps=8 | 50,867 μs | 更多 warp 隐藏标量延迟 |
| R7 | Config tuning | 50,700 μs | BK=128, BV=64, num_warps=8 |

经验：`exp_kda_bwd_wy_scalar_opt_001`、`exp_kda_bwd_wy_structure_opt_001`

---

## 相关页面

- [hardware/ascend_910b.md](../hardware/ascend_910b.md) — 硬件规格（UB/Grid/带宽）
- [hardware/ascend_patterns.md](../hardware/ascend_patterns.md) — Ascend 特有语法规范
- [techniques/ilp.md](ilp.md) — ILP 详解
- [techniques/persistent_kernels.md](persistent_kernels.md) — Persistent kernel 设计
- [techniques/data_types.md](data_types.md) — int32/int64 scalar fallback 修复
- [techniques/compute_and_branch.md](compute_and_branch.md) — min()/tl.cdiv 标量消除
- [ops/ssm.md](../ops/ssm.md) — KDA 反向 kernel 调优实例
- [workflows/profiling.md](../workflows/profiling.md) — 性能分析与瓶颈识别
