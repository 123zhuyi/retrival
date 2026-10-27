---
query_type: perf_tuning
query_keywords: [VecUtil_low, elementwise_add, broadcast, fast_path_separation, block_size_tuning]
created: '2026-05-26T09:32:00Z'
last_updated: '2026-08-05T23:30:00'
hit_count: 4
context_summary: "Elementwise scaled-add with broadcasting suffers from 0.007x speedup. Bottleneck = VecUtil_low: stride-based 4D index decomposition (div/mod per element) is expensive even for non-broadcast cases. Need dual-path: 1D fast path for equal shapes, broadcast-aware kernel only when needed."
top_units: ["T_dual_path", "T_block_size_tune", "T_simd_friendly_index"]
stale: false
---

# Perf Tuning: VecUtil_low for Elementwise Add with Broadcast

## 瓶颈识别

- **Baseline speedup**: 0.0070x (≈150x slower than PyTorch)
- **Root cause**: 当前 kernel 为每个元素做 `i0 = off // (S1*S2*S3); rem = ...; i1 = ...` 等
  4 轮 div/mod，即使 18/50 等-shape cases 也走同样路径
- **指标推断**: VecUtil_low (vector unit 被标量除模运算饱和), MTE2_high (12 个 scalar arg
  + stride pointer arithmetic 增加了寻址开销)

## Top-3 P2 调优知识单元

### 1. T_dual_path — 双路径分离（最关键，命中优化点 #2 控制流简化）

- **原理**：对 elementwise op，shape 相等的 case 占大头（18/50 = 36%）。
  对它们使用纯 1D flatten kernel；只对真正广播的 case 走 stride 路径。
- **应用**：host 端 `if x.shape == y.shape: launch fast_kernel else: launch bcast_kernel`
- **预期收益**：等-shape case 路径下，per-element 计算从 ~10 ops 降到 ~3 ops，
  接近 PyTorch 自身 add 性能（实测 elemwise add 在 Ascend Triton 通常可达 1.5–3x）

### 2. T_block_size_tune — BLOCK_SIZE 调大利用 48 VEC cores（命中优化点 #1 切分）

- **原理**：48 VEC core (24 AI Core × 2 VEC) 是物理上限。若 `grid > 48` 调度开销大；
  `grid < 48` 核没用满
- **应用**：对总元素 N，目标 `grid ≈ 48`，故 BLOCK_SIZE ≈ N / 48，
  再向上取 2 的幂（保 256B 对齐）
- **当前问题**：固定 BLOCK_SIZE=4096，对大 shape (N=67M) → grid≈16384 远超 48 核

### 3. T_simd_friendly_index — 用乘加替代 div/mod（命中优化点 #7 算术化简）

- **原理**：Ascend VEC 上 div/mod 比 mul/add 慢 ~5-10x。
  在 broadcast 路径中，若维度数已知 (4)，可以预计算累积乘积，但 div/mod 本身无法避免
- **替代方案**：通过双路径（见 #1）让多数 case 直接绕过 div/mod；
  广播路径再单独优化（broadcast cases 仅 32 个且通常更小）

## 应用策略到 latency-optimizer 优化点映射

| 优化点 # | 名称 | 本次是否启用 | 预期收益 |
|----------|------|-------------|---------|
| #1 | 切分/核数调优 (autotune BLOCK_SIZE) | ✅ 启用 | 中-高 |
| #2 | 控制流简化（dual-path） | ✅ 启用，最高优先级 | 高 |
| #7 | 算术化简（避免 div/mod） | 部分启用（通过 #2 间接） | 中 |

## 参考案例

- elemwise-cast: BLOCK_SIZE=65536 时 grid=2048 性能最优（注：cast 算子单 buffer，可容
  更大 block；本 add 算子三 buffer 容量减半）
- elemwise-broadcast-2d: 双层切分 NUM_BLOCKS=40 + SUB_M=512 → 平衡 UB 利用率
