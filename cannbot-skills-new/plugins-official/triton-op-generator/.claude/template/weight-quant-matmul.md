---
name: weight-quant-matmul
description: 权重反量化矩阵乘类算子（WeightQuantBatchmatmul：int8 权重在线反量化 + GEMM）的 Triton Ascend 优化经验合集，含 Layer 1 设计约束 / Layer 2 算法骨架 / Layer 3 关键 kernel
metadata:
  type: reference
---

# 权重反量化矩阵乘类算子优化经验

本文档归档 `npu_weight_quant_batchmatmul` 的 Triton-Ascend 重实现经验。算子语义：
`out[M,N] = x[M,K] @ dequant(weight[K,N] int8) + bias`，`dequant(w) = (w + antiquant_offset) * antiquant_scale`，
支持 pertensor(scale[1]) / perchannel(scale[1,N]) / pergroup(scale[G,N], group_size) 量化、可选 offset/bias、fp16/bf16 输入。

- **§1 通用经验**：本类算子共有工程约束
- **§2 WeightQuantBatchmatmul**：Layer 1-3 + 性能基准 + 陷阱

> ⚠️ **关键区分**：本算子是 GEMM(计算密集) + 在线反量化(vector)，优化哲学是「cube 利用率 + 反量化域精度对齐」，与 DynamicQuant(memory-bound 2-pass) / SwigluQuant(activation UB 压力) 完全不同，**禁止混用经验**。

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| WeightQuantBatchmatmul | `matmul` + `elementwise`(fused dequant) | int8 权重在线反量化 + GEMM，cube 密集 | 反量化域精度对齐 + 大 tile(BLOCK_M/K) 提升 cube 利用率 + swizzle2d L2 复用 |

---

## §1 通用经验（首次生成必须遵守）

### W1 反量化域必须按 x.dtype 分流（精度硬约束）
- **必须** fp16 输入在 fp16 域反量化：`(w.to(f16) + off.to(f16)) * scale.to(f16)`；bf16 输入在 fp32 域反量化后截断 bf16：`((w.to(f32) + off.to(f32)) * scale.to(f32)).to(bf16)`。
- **Why:** 对齐 torch_npu op 的 cube L0C 行为（bf16 路径 fp32 域避免加法尾数损失再截断）。域不匹配会导致 MERE 超阈值（fp16 9.77e-4 / bf16 7.81e-3）。48/48 verify 验证。

### W2 GEMM 操作数保持 low_dtype，fp32 累加
- **必须** `acc += tl.dot(x_low, w_dq_low)`，cube 原生 fp16×fp16->fp32 / bf16×bf16->fp32。low_dtype 值在 fp32 精确可表，乘积精确，fp32 累加 == 标杆 fp32 split-K。
- **禁止** 把操作数先 `to(f32)` 再 `tl.dot(fp32,fp32)`——非 cube 原生模式，慢且无精度收益。

### W3 pertensor scale 物化为连续 [1,N]，禁止 stride-0 广播 load
- **必须** pertensor scale=[1] 经 `.view(1,1).expand(1,N).contiguous()` 物化为连续 [1,N]（stride (N,1)），走正常向量 load。
- **禁止** 用 expand 得 stride-0 视图直接 `tl.load(ptr + offs*0)`——Ascend 编译器无法 tile，触发 UB overflow（实测 790KB > 192KB UB，iter_0 教训）。

### W4 pergroup BLOCK_K | group_size
- **必须** pergroup 时 BLOCK_K 整除 group_size（group_size 为 2 的幂），保证每个 K-tile 落在单一 group，scale 只取一行 `scale[k//gs, :]`。选 `BLOCK_K = min(group_size, 128)`（fp16）/ `min(group_size, 128)`（bf16, 见 W7）。

### W5 固定 CUBE_CORE_NUM 启动 + 核内循环
- **必须** `grid=(CUBE_CORE_NUM,)`，CUBE_CORE_NUM 在 `__init__` 经 `torch_npu.npu.npu_config.get_device_limit(0).get('cube_core_num', 20)` 获取，kernel 内 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 交错处理多块。
- **禁止** 硬编码核数；禁止 `grid=(NUM_BLOCKS,)` 启动所有块。

### W6 forward() 只做 shape/分派/host 预计算
- **禁止** forward 中 torch 计算（Type-3 退化）；所有反量化+matmul 在 `@triton.jit` kernel 内。`.contiguous()`/`.view()`/`.expand()`/`.stride()` 允许。

### W7 bf16 融合反量化（单表达式）解锁 BLOCK_K=128
- **必须** bf16 反量化写成单表达式 `((w_i8.to(f32) + off_f32) * scale_f32).to(bf16)`，让编译器把 fp32 中间量留寄存器不物化到 UB。否则 bf16 BLOCK_K=128 会 UB overflow（fp32 中间量 [BK,BN] 过大）。
- **Why:** bf16 BLOCK_K=64 -> 128 对齐 fp16，bf16 case 加速比 0.48 -> 0.65（opt_iter_3->4）。

### W8 BLOCK_M=256 对 M>=256 case（总 block-iter 减半）
- **必须** M>=256 时 BLOCK_M=256（L0A/L0C 满载可编译），总 block-iter 减半，更大 dot 改善 cube 利用率。
- **效果:** opt_iter_2/3 大 M case 0.27 -> 0.5+，bf16/medium 0.48 -> 0.65+。

### W9 swizzle2d 块重排 + BLOCK_K=128
- **必须** `tl.swizzle2d(block_m, block_n, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SW)` 重排块提升 L2 复用；fp16 BLOCK_K=128（非 pergroup）减半 K 迭代。BLOCK 向上取 2 的幂且 `tl.constexpr`。

### W10 precompute dispatch（预反量化分派，gs<=64 / 大 M 中等 N 启用）
- **必须** 对 pergroup gs<=64（fused 被迫 BK=32/64）或 (M>=1024 且 N<=8192) 的 case，改用 precompute 路径：单独 `_dequant_kernel` 把 int8 权重反量化成 low_dtype [K,N]，再 `_matmul_kernel` 纯 matmul（BK=128 不受 gs 约束 + 移除 K-loop 反量化阶段减 sync）；其余用 fused。
- **禁止** 对 N>8192 的大 case 用 precompute：w_dq [K,N] 内存开销（128MB@N=16384）主导反而回退（opt_iter_9 教训）。
- **禁止** precompute 的 dequant 用 per-element gather（`offs_k//gs` [BK,1]）：触发 MLIR 编译器 assert crash；用 single-row scale（g_idx=k_start//gs, BK|gs）。
- **效果:** gs<=64 case 0.27/0.48 -> 0.56/0.73；大 M 中等 N 0.55 -> 0.73；geomean 0.6814 -> 0.7571。
- **AST 陷阱:** forward 内禁 `while`/`min()`（被 Type-3 误报），把 tile 配置放 helper（`_pc_blocks`），forward 只调 helper + kernel。

---

## §2 WeightQuantBatchmatmul 算子

**算子类别**: `matmul` + fused `elementwise`(dequant)
**典型特征**: x[M,K] fp16/bf16 + weight[K,N] int8 + scale[G,N] + 可选 offset/bias + group_size；输出 [M,N] 同 x.dtype
**性能基准**: 48/48 verify pass，几何平均加速比 **0.7571x** vs torch_npu.npu_weight_quant_batchmatmul（Phase 3 基线 0.3505x，Phase 4 优化 2.16×；未达 target 0.8，因大 M+大N case 受 cube 流水 sync-bound 限制，破例归档）

### §2.1 Layer 1: 设计约束（硬性边界）

见 §1 W1-W9。核心：W1(反量化域)/W2(low_dtype+fp32)/W3(pertensor 物化)/W4(pergroup BLOCK_K)/W7(bf16 融合)/W8(BLOCK_M=256)。

### §2.2 Layer 2: 算法骨架

```python
# Host (forward):
# 1. x/weight contiguous; M,K=x.shape; N=weight.shape[1]
# 2. scale2d = _normalize_scale_offset(scale, N, per_group)  # pertensor 物化 [1,N]
# 3. block_m, block_n, block_k = _choose_blocks(per_group, gs, is_bf16, M)  # M>=256->256; fp16/bf16 BK=128
# 4. out = empty([M,N], dtype=x.dtype)
# 5. kernel[(CUBE_CORE_NUM,)](x, weight, scale2d, offset2d, bias, out, M,N,K, strides..., gs,
#        HAS_OFFSET, HAS_BIAS, PER_GROUP, IS_BF16, num_cores, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SW=8)

# Kernel:
# pid = tl.program_id(0)
# for block_idx in range(pid, NUM_BLOCKS, num_cores):
#     block_m_raw = block_idx // NUM_BLOCKS_N; block_n_raw = block_idx - block_m_raw*NUM_BLOCKS_N  # 禁 %
#     block_m, block_n = tl.swizzle2d(block_m_raw, block_n_raw, NUM_BLOCKS_M, NUM_BLOCKS_N, GROUP_SW)
#     acc = zeros([BM,BN], f32)
#     for k in range(0, K, BK):
#         x_tile = load(x[BM,BK])  # low_dtype
#         w_i8 = load(weight[BK,BN])  # int8
#         g_idx = (k//gs) if PER_GROUP else 0
#         scale_row = load(scale[g_idx, BN])  # pertensor: 物化后 stride 1
#         offset_row = load(offset[g_idx, BN]) if HAS_OFFSET else zeros
#         # 反量化域分流 (W1/W7)
#         if IS_BF16: w_dq = ((w_i8.to(f32)+off_f32)*scale_f32).to(bf16)
#         else:       w_dq = (w_i8.to(f16)+off_f16)*scale_f16
#         acc = tl.dot(x_tile, w_dq, acc=acc)  # W2: low_dtype -> f32
#     if HAS_BIAS: acc += bias.to(f32)
#     store(out, acc.to(low_dtype))
```

### §2.3 Layer 3: 关键技巧

#### L3.1 反量化域分流 + 融合（W1/W7）
```python
if IS_BF16:
    off_f32 = offset_row.to(tl.float32)[None, :]
    scale_f32 = scale_row.to(tl.float32)[None, :]
    w_dq = ((w_i8.to(tl.float32) + off_f32) * scale_f32).to(tl.bfloat16)  # 单表达式, fp32 留寄存器
else:
    w_f16 = w_i8.to(tl.float16)
    w_dq = (w_f16 + offset_row.to(tl.float16)[None, :]) * scale_row.to(tl.float16)[None, :]
acc = tl.dot(x_tile, w_dq, acc=acc)  # low_dtype x low_dtype -> fp32 acc
```

#### L3.2 scale 归一化（W3）+ BLOCK 自适应（W4/W8）
```python
def _normalize_scale_offset(t, n, per_group):
    if per_group: return t.view(1,-1) if t.dim()==1 else t
    if t.dim()==2 and t.shape[0]==1: return t  # perchannel [1,N]
    return t.view(1,1).expand(1,n).contiguous()  # pertensor 物化, 禁 stride-0

def _choose_blocks(per_group, gs, is_bf16, M):
    block_n = 128; block_m = 256 if M >= 256 else 128; cap = 128
    if per_group:
        bk = cap
        while bk > gs: bk //= 2  # BLOCK_K | gs
        block_k = bk
    else: block_k = cap
    return block_m, block_n, block_k
```

### §2.4 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| fp16 小/中 (M=128~512) | ~20 | 0.72 ~ 1.21 | 多数 >1.0 (超越原生 op) |
| fp16 大 (M=1024~2048, N=4096~16384) | ~6 | 0.48 ~ 0.58 | cube 流水气泡, 原生 op 69% peak |
| bf16 (M>=256) | ~13 | 0.48 ~ 0.66 | fp32 反量化开销, BLOCK_K=128 后接近 fp16 |
| bf16 小 (M=128) / pergroup gs=32 | ~3 | 0.25 ~ 0.50 | 小 tile 强制, 调度开销主导 |
| 全量 | 48 | **0.7571x** (geomean) | Phase 3 0.3505x -> Phase 4 2.16× (precompute dispatch) |
| 全量 (pergroup-only precompute + autotune, 2026-07-21) | 48 | **0.7479x** (geomean) | Phase 3 0.6776x -> Phase 4 1.10×；大 shape 仍受 cube 流水瓶颈约束 |
| 全量 (ascend910b3, 2026-07-30) | 48 | **0.8112x** (geomean) | Phase 3 基线 0.8112x，Phase 4 3 轮优化（Kernel 分裂/Autotune/全点检查）均无提升，确认当前代码已达最优 |
| 全量 (Ascend910_9372 / ascend910b3-family, 2026-08-06) | 48 | **0.8547x** (geomean) | Phase 3 基线 0.7831x (fused-only)，Phase 4 W10 precompute-for-gs<=64 单轮 1.09x 提升至 0.8547x；超 target 0.8；大 shape 仍受 cube 流水限制 ~0.54-0.61 |

**关键结论**:
1. 反量化域对齐 + cube 原生 low_dtype->fp32 是精度通过且高效的核心。
2. BLOCK_M=256 (M>=256) + BLOCK_K=128 + bf16 融合反量化是性能主杠杆（0.35->0.68）。
3. 大 shape case 受 cube 流水气泡限制，Triton DSL 无法暴露 CCE 级流水控制，speedup 上限约 0.6~0.7（原生 op 已 69% peak），是已知架构限制。
4. **新经验（2026-07-21）**: 对 pergroup gs<=64 启用 precompute 路径，并对 fused kernel 使用 @triton.autotune（BLOCK_M/BLOCK_K/GROUP_SW），可在不破坏精度前提下将 geomean 从 0.6776x 提升至 0.7479x；非 pergroup 的大 M 中等 N case 的 precompute 纯 matmul 路径在当前实现下会产生 NaN/inf，应避免。
5. **新经验（2026-07-30, ascend910b3）**: 在 ascend910b3 上 48/48 verify 通过，geomean 0.8112x 已达上限。对所有 27 个优化点逐一检查：Kernel 分裂（点18）和 Autotune（点13）均无提升，其余 25 点未命中。大 shape 瓶颈为 cube 流水气泡的架构限制，Triton DSL 无法进一步优化。
6. **新经验（2026-08-06, Ascend910_9372）**: 当 Phase 3 fused 基线未启用 precompute 时（geomean 0.7831x），Phase 4 对 pergroup gs<=64 启用 W10 precompute 路径（`_dequant_kernel` BLOCK_DK=32|gs + `_matmul_kernel` BLOCK_K=128）可单轮提升至 0.8547x（gs=32: 0.27->0.81、0.30->0.78；gs=64: 0.52->0.86），超 target 0.8。与 2026-07-30 "kernel-split 无提升" 不矛盾：彼时基线已含 precompute。**结论：precompute-for-gs<=64 是 fused 基线缺失时的关键杠杆；fused 基线已含 precompute 时无需再分裂。**

### §2.5 常见陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| pertensor UB overflow | stride-0 广播 load 无法 tile | W3: 物化 [1,N] 连续 |
| bf16 BLOCK_K=128 UB overflow | fp32 中间量 [BK,BN] 物化到 UB | W7: 单表达式融合, fp32 留寄存器 |
| 精度 MERE 超阈值 | 反量化域不匹配 (bf16 用 fp16 域或反之) | W1: 按 x.dtype 分流 |
| bf16 BLOCK_N=64 回退 | N-block 翻倍, 小 case 调度开销主导 | BLOCK_N >= 128 (opt_iter_1 教训) |
| 大 case cube 利用率低 | K-loop 流水气泡 | BLOCK_M=256 + BLOCK_K=128 减迭代; 架构限制接受 |
| pergroup group 越界 | BLOCK_K 不整除 group_size 跨组 | W4: BLOCK_K | group_size, 单组单 scale 行 |
| 分核(小tile增并行)回退 | 小/中 case 是 overhead-bound(每块cube工作小,launch/sync主导), 减tile增block数使overhead翻倍 | 禁止对 under-utilized case 减 tile; 大 tile(opt_iter_10)最优 |
| 非 pergroup 大 M 中等 N precompute 产生 NaN/inf | 纯 matmul 路径在当前实现下触发 UB 状态异常（具体根因待进一步 IR 分析） | 仅对 pergroup gs<=64 启用 precompute；非 pergroup 大 case 保留 fused 路径 |
| autotune 配置过多拖慢编译 | 每增加一个 config，所有 shape 需多编译一次 kernel | 限制 config 数量（本实现 6 个），优先探索 GROUP_SW 和 BLOCK_M |
| Kernel 分裂导致性能回退 (2026-07-30) | 大 shape 专用 kernel 用 BLOCK_M=128, BLOCK_N=256 导致 M 块数翻倍和调度开销增加；小 pergroup 预计算双 kernel 启动开销仍存在 | 对 ascend910b3 的 fused 路径已是最优，不建议分裂 |
