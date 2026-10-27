---
title: 内存访问优化
category: techniques
tags: [memory-access, coalescing, block-ptr, alignment, cache-reuse, software-pipeline, hbm, ub]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# 内存访问优化

> 本页从**内存访问维度**汇总 HBM→UB 带宽优化策略，覆盖对齐、连续访问、block_ptr、cache 复用、软件流水线等模式。

---

## 1. 连续内存访问优先

### 1.1 避免 stride 跳跃访问

随机/stride 访问导致 HBM 访问不连续，缓存命中率低 50%+：

```python
# 反例：多维 stride 偏移（不连续，cache miss 高）
m_offs = rm[:, None] * stride_m
n_offs = rn[None, :] * stride_n
ptr = A_ptr + m_offs + n_offs   # 每次都是离散地址

# 正例：连续一维访问（每次 load/store 是连续内存区间）
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offs, mask=offs < n)
```

经验：`exp_perf_007`

---

### 1.2 索引排序改善局部性

Scatter/Gather 前对索引排序，使相邻 program 写入相邻地址：

```python
# 排序后 scatter：相邻写入相邻地址，HBM burst 合并
sorted_order = torch.argsort(index)
sorted_index = index[sorted_order]
sorted_src   = src[sorted_order]
# 现在 scatter 是近似连续写
```

经验：`exp_gs_mem_005`（30% 提升）

---

## 2. 2D Block Pointer（tl.make_block_ptr）

对矩阵操作，使用 `tl.make_block_ptr` 替代手动 2D 偏移，降低地址计算开销并获得 HBM 预取提示：

```python
# 手动偏移（低效）
a_offs = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_am + \
         (k_off + tl.arange(0, BLOCK_K))[None, :] * stride_ak

# block_ptr（高效，编译器可优化预取）
a_ptr = tl.make_block_ptr(
    base=A_ptr,
    shape=(M, K),
    strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, k_off),
    block_shape=(BLOCK_M, BLOCK_K),
    order=(1, 0)   # row-major: 最后一维连续
)
a = tl.load(a_ptr)
a_ptr = tl.advance(a_ptr, (0, BLOCK_K))  # 在 K 循环中推进
```

**约束**：
- `order=(1, 0)` 表示行优先（内存连续维度在后）
- strides 单位为元素数（不是字节）
- 步长最小单位仍需满足 512B 对齐

经验：`exp_perf_002`

---

## 3. 内存对齐

### 3.1 512B 行宽对齐

Ascend 910b 要求**尾轴（最内层维度）步长**是 512B 的整数倍（fp16: 256 elements；fp32: 128 elements）：

```python
# 错误：N=300, fp16 → stride=300*2=600B，不是 512B 倍数
# 正确：pad N 到 512B 整数倍
N_aligned = triton.next_power_of_2(N)  # 或手动 ceil_div(N, 256)*256
stride_n = N_aligned
```

**诊断**：若遇到编译错误 "illegal memory access" 或精度问题，首先检查 stride 对齐。

---

### 3.2 128B 输出对齐（int8/fp8）

量化输出 int8 每次 store 须 128B 对齐（64 个 int8 elements）：

```python
BLOCK_SIZE_INT8: tl.constexpr = 128  # 128 elements = 128B for int8
# 确保 BLOCK_SIZE_INT8 是 64 的倍数
```

---

### 3.3 ALIGNED constexpr 路径选择

当 rope_dim、hidden_dim 等维度已知对齐时，编译时选择无边界检查的快速路径：

```python
@triton.jit
def kernel(..., ALIGNED: tl.constexpr):
    if ALIGNED:
        x = tl.load(ptr + offs)          # 无 mask，最快
    else:
        x = tl.load(ptr + offs, mask=offs < dim)  # 带边界检查
```

---

## 4. 消除中间 Tensor（HBM 读写优化）

### 4.1 算子融合消除 HBM 写回

中间 tensor 写到 HBM 再读回是最大的带宽浪费：

```
未融合：GEMM → [HBM 写] → Activation → [HBM 读] → Norm → [HBM 写]
已融合：GEMM → [L0C 寄存器] → Activation → [寄存器] → Norm → [HBM 写]
```

每消除一对 HBM 读/写，节省约 15-20% 延迟。

经验：`exp_perf_003`；详见 [techniques/operator_fusion.md](operator_fusion.md)

---

### 4.2 增大 BLOCK_SIZE 提升 IO 效率

当 PipeUtil 低、MemBW 高时，通过增大 BLOCK_SIZE 摊薄每次 IO 的启动开销：

| BLOCK_SIZE | 每次 load 字节数 | IO 效率 |
|-----------|-----------------|---------|
| 256 | 512B (fp16) | 低（起步开销占比高） |
| 1024 | 2048B (fp16) | 中 |
| 2048 | 4096B (fp16) | 高（接近 HBM burst 长度） |

**约束**：BLOCK_SIZE × dtype_bytes ≤ UB 192KB 余量。

---

## 5. 软件流水线（Software Pipeline）

在 `tl.range` 的 `num_stages` 参数启用软件流水线，使下一轮的 load 与当前轮的 compute 重叠：

```python
# 2级流水：prefetch 下一 K tile 同时计算当前 K tile
for k in tl.range(0, K, BLOCK_K, num_stages=2):
    a = tl.load(a_ptr + ...)
    b = tl.load(b_ptr + ...)
    acc += tl.dot(a, b)
```

**级数选择**：
- `num_stages=1`：无流水（默认）
- `num_stages=2`：双缓冲，最常用
- `num_stages=3`：三缓冲，需 UB 有足够余量存放 3 个 tile

**UB 占用计算**：`num_stages × (BLOCK_M × BLOCK_K + BLOCK_K × BLOCK_N) × dtype_bytes ≤ 192KB`

经验：`exp_qwen2vl_mrope_001`、`exp_rope_npu_001`（RoPE 软流水）

---

## 6. cos/sin 等常量的寄存器驻留

对 head 循环内不变的小型常量张量（cos/sin 表的当前行），在循环外一次加载到寄存器：

```python
# 低效：cos/sin 在 head 循环内每次重新加载
for h in range(num_heads):
    cos = tl.load(cos_ptr + pos * head_dim + ...)  # 重复 HBM 读

# 高效：在 head 循环外加载，循环内直接使用寄存器
cos = tl.load(cos_ptr + pos * head_dim + ...)      # 1 次加载
sin = tl.load(sin_ptr + pos * head_dim + ...)
for h in range(num_heads):
    q = tl.load(q_ptr + ...)
    q_rot = q * cos - q_shift * sin                # 使用寄存器 cos/sin
```

同理适用于：bias 向量、scale 因子、RMSNorm weight、量化 scale。

---

## 7. 内存访问诊断表

| profiling 症状 | 诊断 | 修复方向 |
|---------------|------|---------|
| `aiv_mte2_time` > 50% 总时间 | HBM→UB 带宽瓶颈 | 增大 BLOCK_SIZE / 连续化 / 消中间 tensor |
| `mte2_stall` 高 | 非连续访问频繁 stall | 排序索引 / block_ptr / 检查 stride 对齐 |
| `PipeUtilization` < 75% | load/compute 无重叠 | 加 num_stages=2 软流水 / for-loop tiling |
| 编译错误 "illegal memory access" | stride 不对齐 | 检查尾轴 stride 是否 512B 整数倍 |
| int8 store 精度异常 | 输出未 128B 对齐 | BLOCK_SIZE_INT8 用 128 的倍数 |
| W 循环展开后性能退化（小 D 时） | 独立 tl.load 替代 block_ptr | 保留 block_ptr + extract_slice 混合方案（§9） |
| T-tiling 后 tile 边界结果异常 | 未加载跨 tile 历史 | 叠加 (W-1) 重叠区（t_start - (W-1)，§8） |
| 小 W 短窗口 kernel 带宽利用率低 | 每步重复加载全窗口 | 寄存器移位复用（§10），W=4 时可消除 75% load |
| 循环内逐行 tl.load 重复读取已加载的块 | 数据已在寄存器 + Triton 不支持动态索引 | preload + extract_slice（§11），从寄存器提取替代 HBM 读 |
| kernel launch 编译错误（CANN 不兼容参数） | num_stages/sync_solver 等依赖 TMA 特性 | 审计并删除 CANN 8.5.0 不兼容参数 |
| `aiv_scalar_ratio` 高 + 大量 block_ptr load/store | boundary_check 生成逐元素 int32 比较 | 改用手动指针 + fp32 mask（§12） |
| 转置视角加载且 `aiv_mte2_time` 高 | 列方向 stride 过大，非合并访存 | 行优先 load + `tl.trans` 寄存器转置（§13） |
| barrier 前后重复加载同一数据 | 跨阶段未复用已加载数据 | 保持寄存器复用，实测 spill 情况（§14） |

---
---

## 8. 重叠 T-tiling（序列处理通用模式）

### 8.1 核心思想

对序列数据做 T 维度切分时，相邻 tile 之间保留 (W-1) 个元素的重叠区，保证跨 tile 的滑动窗口操作连续：

```
T-tile 0: [history | t0 ..... t255]
T-tile 1:          [t254, t255 | t256 ..... t511]
                   ↑----------↑
                W-1 个重叠元素（无需从状态缓存重新加载）
```

每个后续 tile 向前多加载 (W-1) 个 token（buffer = T_CHK_SIZE + W - 1），前 (W-1) 列自然等于上一个 tile 的结尾。仅首个 tile 需要从外部状态缓存读取历史。

```python
# T-tile 0: 从状态缓存拼接历史
x_b = tl.insert_slice(
    tl.load(state_ptr, offsets=(0, state_len - (W-1)), block_shape=(D_CHK, (W-1)+T_CHK)),
    tl.load(x_ptr, offsets=(0, 0), block_shape=(D_CHK, T_CHK)),
    (0, W-1), (D_CHK, T_CHK))

# T-tile i (i>0): 向前偏移 W-1，自然包含前 tile 结尾
x_b = tl.load(x_ptr, offsets=(0, ti*T_CHK - (W-1)), block_shape=(D_CHK, T_CHK+W-1))
```

### 8.2 适用场景

- 序列处理 kernel 做了 T 维度 tile 化（如 Causal Conv1D、SSM chunk scan）
- 滑动窗口 / 卷积 / 状态机等需要跨 tile 历史上下文的算子
- W 越小（2-4），额外 buffer 开销越低（~1-3% UB 增量），收益越明显

### 8.3 收益分析

相比 naive 方案（每个 tile 独立从状态缓存取 W-1 个历史 token），节省 `(num_T_tiles - 1) × (W-1) × D_CHK_SIZE` 次全局内存 load。典型场景节省 15-25% 内存带宽。

经验：`exp_causal_conv1d_update_004`；详见 [ops/convolution.md](../ops/convolution.md)

---
---

## 9. make_block_ptr vs 标量 tl.load — 硬件本质差异

### 9.1 为什么不能混用

`tl.make_block_ptr` 在 Ascend NPU 上映射为**硬件 block transfer 指令**（类似 DMA），一条指令搬运整个 tile，包含：
- 自动边界检查（`boundary_check`）
- 自动零填充（`padding_option="zero"`）
- 硬件预取提示（MTE2 可异步预取下一 tile）

4 次独立 `tl.load(weight_ptr + d_offs * width + offset, mask=mask_w)` 每条都是**独立标量地址 load**，丧失上述所有硬件加速。

### 9.2 性能差异量级

在 D≥2048 的 Causal Conv1D Update kernel 中，4 次独立 load 替代 1 次 block_ptr 可导致 2-3x 的性能倒退（与 W=4 循环展开的组合效应下）。

### 9.3 混合方案（保留 block_ptr 加速 + 手动索引）

需要手动索引权重列时（如循环展开），**先用 block_ptr 加载整个 tile，再用 extract_slice 切分**，而非退化到独立标量 load：

```python
# 推荐：保留硬件加速
w = tl.load(make_block_ptr(weight_ptr, ..., block_shape=(D_CHK, W)))
w0 = tl.extract_slice(w, (0, 0), (D_CHK, 1), (1, 1))  # view，零数据移动
w1 = tl.extract_slice(w, (0, 1), (D_CHK, 1), (1, 1))
# ...
# 反模式：标量 load（丧失硬件加速）
w0 = tl.load(weight_ptr + d_offs * W + 0, mask=mask_w)  # 每条独立地址计算
w1 = tl.load(weight_ptr + d_offs * W + 1, mask=mask_w)
```

经验：`exp_causal_conv1d_update_003`；详见 [ops/convolution.md](../ops/convolution.md)

---

## 10. 寄存器驻留 + 移位代替重复加载（滑动窗口模式）

### 10.1 核心思想

对于固定小窗口（W=2-8）的序列处理，将窗口元素全部加载到命名寄存器后，通过**寄存器移位**（而非重复加载）实现窗口滑动。每次迭代仅需加载 1 个新元素，而非 W 个：

```
迭代 i:   [col0, col1, col2, col3] → 计算 → store
迭代 i+1:         [col1, col2, col3, new_col] → 计算 → store
                ↑ 移位（零 load）  ↑ 仅此 1 次 load
```

### 10.2 代码示例

```python
# 初始化：一次加载 W 个窗口元素
col0 = tl.load(ptr + 0*stride, mask=mask)
col1 = tl.load(ptr + 1*stride, mask=mask)
col2 = tl.load(ptr + 2*stride, mask=mask)
col3 = tl.load(ptr + 3*stride, mask=mask)  # 当前元素

for i in range(seq_len - W):
    acc = col0*w0 + col1*w1 + col2*w2 + col3*w3
    tl.store(out + i*stride, acc, mask=mask)
    col0 = col1; col1 = col2; col2 = col3           # 移位（零 HBM 操作）
    col3 = tl.load(ptr + (i+W) * stride, mask=mask) # 仅 1 次新 load
```

### 10.3 适用条件

- W 固定且小（≤8），所有窗口元素可驻留寄存器
- 窗口访问模式为严格移位（最旧丢弃，其余左移，末尾新元素）
- seq_len 足够大（≥64），初始化的 W 次 load 可被摊还
- 需要滑动窗口连续性的场景（卷积、移动平均、SSM 前缀处理）

### 10.4 收益

W=4 时消除 75% 的 HBM 读（从每步 4 次 load → 1 次 load + 3 次寄存器间移位）。移位是寄存器间拷贝，零延迟、零带宽消耗。

**与重叠 T-tiling（§8）的互补关系**：寄存器滑动窗口用于**无 T-tiling** 的 Prefill kernel（逐 token 顺序处理）；重叠 T-tiling 用于**有 T-tiling** 的 Update kernel（跨 tile 边界连续性保证）。两者解决不同维度的问题，可在同一 kernel 族中并存。

经验：`exp_causal_conv1d_prefill_002`；详见 [ops/convolution.md](../ops/convolution.md)

---

## 11. Preload + extract_slice 消除冗余 HBM 重读

### 11.1 核心思想

当矩阵块已在循环外通过 `tl.load(make_block_ptr(...))` 加载到寄存器后，循环内不应再用 `tl.load(A + row_offset)` 逐行重新从 HBM 读取同一数据。虽然 Triton 不支持 `b_A[i, :]` 动态寄存器索引，但可用 `tl.extract_slice` 从寄存器提取：

```python
# 循环外：一次性预加载整个块到寄存器
b_A = tl.load(make_block_ptr(A, ..., block_shape=(BLOCK, BLOCK)))
neg_A = -b_A  # 预取反后的块

# 循环内：从寄存器提取，零 HBM 访问
for i in range(2, BLOCK):
    b_a = tl.extract_slice(neg_A, (i, 0), (1, BLOCK), (1, 1))
    b_a = tl.reshape(b_a, (BLOCK,))
    # ... 使用 b_a 计算 ...
```

### 11.2 为什么不用 b_A[i, :]？

Triton 报错 `ValueError('unsupported tensor index: int32[]')`——`i` 是运行时 int32，block tensor 不支持动态索引。`tl.extract_slice` 是 Triton 支持的寄存器块子集提取原语，offset 为运行时变量时生成动态偏移指令（开销仍远低于 HBM load）。

### 11.3 适用条件

- 数据已在循环外的寄存器中（通过 block_ptr 或一次性 load 获取）
- 循环内需要按行/按列/按子块访问同一数据
- 循环次数 ≥ 2（否则预加载的额外寄存器操作不摊还）

### 11.4 收益与权衡

| 场景 | 收益 |
|------|------|
| 大 grid（B≥4, T≥1024） | 显著（1.27x-1.37x），HBM load 节省累积效应放大 |
| 小 grid（B=1, T=512） | 微小（~1.01x），额外寄存器操作与省下的 load 几乎抵消 |
| 多对角块（如 64×64 有 4 个对角块） | 每块独立预加载，各省循环体内 HBM 读取 |

**与 §6 cos/sin 寄存器驻留的关系**：§6 是对**循环不变标量/小向量**的驻留；本节是对**已在寄存器中的矩阵块**避免重复 HBM 读取。两者互补：标量驻留解决"不需要重新加载"，extract_slice 解决"不能动态索引寄存器块"。

经验：`exp_solve_tril_002`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

---

## 12. block_ptr boundary_check 的标量代价

`tl.make_block_ptr` 在 Ascend 上通常能生成高效的硬件 block transfer，但 **`boundary_check` 会引入逐元素 int32 边界比较**，在 tile 较小或调用频繁时成为 scalar 开销的主要来源。

```python
# 高标量开销：小 tile + 频繁调用
for i_j in range(i_i):
    p_k = tl.make_block_ptr(k_local, (T, K), (H*K, 1),
                            (i_t*BT + i_j*BC, i_k*BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))   # 每次循环都生成标量边界检查
```

**优化方案**：改为手动指针 + 向量化 mask，mask 计算外提到循环外（若边界在循环内不变）：

```python
o_i = tl.arange(0, BC)
o_k = tl.arange(0, BK)
m_row = (i_t*BT + o_i).to(tl.float32) < T_local.to(tl.float32)
m_col = o_k.to(tl.float32) < K.to(tl.float32)
mask_2d = m_row[:, None] & m_col[None, :]

for i_j in range(i_i):
    row_off = i_t*BT + i_j*BC + o_i
    p_k = k_local + row_off[:, None] * H*K + (i_k*BK + o_k)[None, :]
    b_k = tl.load(p_k, mask=mask_2d, other=0.0)
```

**决策原则**：
- 完整 tile 且已知对齐：优先用 `block_ptr` 无 `boundary_check` 或 `ALIGNED` constexpr 路径
- 边界 tile 或频繁小 tile load/store：改用手动指针 + fp32 mask
- 单次大块加载：保留 `block_ptr` 以获得硬件预取提示

经验：`exp_perf_011`、`exp_ascend_blockptr_boundary_scalar_001`

---

## 13. 转置 load 合并化

当 kernel 需要以**转置视角**读取一个行优先张量时，直接用 `stride=(1, H*BT)` 的 block_ptr 会导致列方向跨大步访问，严重不连续。

```python
# 反模式：非合并转置 load
p = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT),
                      (i_i*BC, i_t*BT+i_j*BC), (BC, BC), (0, 1))
b = tl.load(p, boundary_check=(0, 1))   # 列方向 stride = H*BT
```

**优化方案**：先按原始行优先布局连续加载，再用 `tl.trans` 在寄存器内转置：

```python
row_idx = i_t*BT + i_j*BC + o_i[:, None]   # 连续方向
col_idx = i_i*BC + o_i[None, :]            # 将被转置为行方向
mask = (row_idx < T)[:, None] & (col_idx < BT)[None, :]
p_row = dAqk + row_idx * H*BT + col_idx
b_row = tl.load(p_row, mask=mask, other=0.0)
b = tl.trans(b_row)                        # 寄存器转置，零 HBM 额外开销
```

经验：`exp_perf_015`、`exp_kda_bwd_intra_memory_opt_001`

---

## 14. 跨阶段冗余 load 消除

同一数据在不同计算阶段被重复加载会浪费 HBM 带宽。若数据已在寄存器中且 `tl.debug_barrier()` 未强制 spill，可直接复用。

```python
# Phase 1.5 加载 b_q
b_q = tl.load(p_q, mask=mask, other=0.0)
# ... 使用 b_q 计算 dq2/dg2 ...

tl.debug_barrier()

# Phase 2.5 直接使用寄存器中的 b_q，无需重新加载
# ... 继续使用 b_q ...
```

**适用条件**：
- barrier 前后数据确实相同
- 寄存器压力未导致 barrier 强制 spill
- 需实测验证 MTE2 是否下降

经验：`exp_perf_016`、`exp_kda_bwd_intra_memory_opt_001`

---

## 相关页面

- [techniques/tiling.md](tiling.md) — UB 容量限制与 tile size 计算
- [techniques/ilp.md](ilp.md) — care_padding=False + for-loop 流水
- [techniques/operator_fusion.md](operator_fusion.md) — 算子融合消中间 tensor
- [hardware/ascend_910b.md](../hardware/ascend_910b.md) — UB 192KB、HBM 1.8TB/s、512B 对齐
- [hardware/constraints_cheatsheet.md](../hardware/constraints_cheatsheet.md) — C1-C5 强制约束
- [ops/rope.md](../ops/rope.md) — cos/sin 复用 + ALIGNED 路径实例
- [ops/convolution.md](../ops/convolution.md) — make_block_ptr vs scalar load 实测分析、重叠 T-tiling 实例、寄存器驻留 + 移位实例
- [ops/solve_tril.md](../ops/solve_tril.md) — preload + extract_slice 消除循环内 HBM 重读实例
- [ops/gather_scatter.md](../ops/gather_scatter.md) — 索引排序 + 向量化 scatter 实例
- [ops/ssm.md](../ops/ssm.md) — Chunkwise 分块 SSM（序列 T-tiling + 状态传递）
- [techniques/ilp.md](ilp.md) — 滑动窗口寄存器复用（小固定窗口序列通用模式）
- [techniques/compute_and_branch.md](compute_and_branch.md) — constexpr 双路径展开（配合 extract_slice 使用）
- [techniques/data_types.md](data_types.md) — int32→fp32 比较避免 scalar fallback（配合手动指针 mask）
