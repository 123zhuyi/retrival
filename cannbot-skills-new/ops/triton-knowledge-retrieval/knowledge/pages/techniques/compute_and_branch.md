---
title: 计算与分支处理优化
category: techniques
tags: [constexpr, branch-elimination, fp32-accumulator, type-cast, fma, mask-as-branch, numerical-stability]
hardware: [910b2, 910b3]
last_updated: 2026-07-07
---

# 计算与分支处理优化

> 本页从**计算与分支**维度汇总消除低效分支、提升计算精度、利用 FMA 指令等模式，与算子类型无关。

---

## 1. constexpr 编译时分支消除

### 1.1 tl.constexpr 参数特化

运行时 if/else 分支导致两路代码都被编译且存在分支预测开销。用 `tl.constexpr` 将布尔参数提升为编译时常量，使编译器完全删除未选中的分支：

```python
# 低效：运行时分支
@triton.jit
def kernel(ptr, n, use_bias, BLOCK: tl.constexpr):
    if use_bias:      # 运行时分支：两路都需编译
        bias = tl.load(...)
        out = x + bias
    else:
        out = x

# 高效：constexpr 分支（编译时选择）
@triton.jit
def kernel(ptr, n, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    if HAS_BIAS:      # 编译时消除，只生成所选路径的指令
        bias = tl.load(...)
        out = x + bias
    else:
        out = x
```

**常见 constexpr 开关模式**：

| 开关名 | 含义 | 省去 |
|--------|------|------|
| `HAS_RESIDUAL` | 是否有残差加 | 残差 load + add 指令 |
| `INVERSE` / `BACKWARD_PASS` | 前向/反向共用 kernel | 方向判断和不同运算路径 |
| `IS_NEOX_STYLE` | RoPE 布局（分块 vs 交错） | 布局转换分支 |
| `HAS_CAUSAL_MASK` | 是否有因果掩码 | mask 加法和 tril 检查 |
| `ALIGNED` | 维度是否对齐 | boundary check mask |
| `IS_VARLEN` | 是否变长序列 | 长度查表和偏移计算 |
| `OUTPUT_TYPE` | 输出数据类型 | 类型转换和 cast 指令 |
| `DIM_IS_MULTIPLE` | dim 能否被 D_CHK_SIZE 整除 | 所有 load/store 的 D 维度 mask 运算 |
| `W_IS_4` | W==4 且 D≥阈值时展开卷积循环 | 3 次 tl.trans + 8 次 extract_slice 数据编排开销 |

经验：`exp_constexpr_001`；`exp_rope_001`（INVERSE）；`exp_triton_rope_001`（IS_NEOX_STYLE）；`exp_causal_conv1d_update_005`（DIM_IS_MULTIPLE）；`exp_causal_conv1d_update_002`（W_IS_4）

---

### 1.2 do_not_specialize 避免过度特化

对**经常变化的标量参数**（如 seq_len），标注 `do_not_specialize` 防止为每个值重新编译：

```python
@triton.jit(do_not_specialize=["seq_len", "batch_size"])
def kernel(ptr, seq_len, batch_size, BLOCK: tl.constexpr):
    # seq_len 不触发重编译，减少 JIT overhead
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < seq_len
    ...
```

经验：`exp_rope_forward_001`

---

### 1.3 mask 作为分支替代

用 `tl.where` 和 mask 替代条件分支，保持 SIMD 统一执行路径：

```python
# 分支方式（串行化，产生指令发散）
if pid < special_case:
    result = compute_special()
else:
    result = compute_normal()

# mask 方式（全员执行，无分发）
is_special = (pid < special_case)
r_special = compute_special()   # 所有 program 都算（后被 mask 丢弃）
r_normal  = compute_normal()
result = tl.where(is_special, r_special, r_normal)
```

经验：`exp_ascend_where_001`

---

### 1.4 DIM_IS_MULTIPLE — 编译期 mask 消除

当某个维度大小能被 tile size 整除时（如 `dim % D_CHK_SIZE == 0`），通过 constexpr 开关跳过该维度所有 load/store 的 mask 检查：

```python
# host 端
DIM_IS_MULTIPLE = (dim % D_CHK_SIZE == 0)

# kernel 端
@triton.jit
def kernel(..., DIM_IS_MULTIPLE: tl.constexpr, D_CHK_SIZE: tl.constexpr):
    d_offs = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
    if not DIM_IS_MULTIPLE:
        mask_d = d_offs < dim
        x = tl.load(ptr + d_offs, mask=mask_d)
    else:
        x = tl.load(ptr + d_offs)  # 无 mask，编译器消除边界检查
```

**适用条件**：tile size 能整除维度大小（如 dim=4096, D_CHK_SIZE=16 或 BLOCK_D=1024）。

**收益**：减少 5-15% 标量指令（取决于 tile 数量），D 维度大时更显著。

**适用范围**：所有做 D/BD 维度 tile 化的 kernel（matmul、conv1d、elementwise、normalization 等）。

经验：`exp_causal_conv1d_update_005`；详见 [ops/convolution.md](../ops/convolution.md)

---

### 1.5 W_IS_4 — 循环展开 + D 阈值控制

将小型固定次数循环（如 W=4 的卷积核循环）展开为多路独立表达式，消除循环开销和中间转置操作。**关键**：必须配合 D 维度阈值控制，避免小 D 场景退化：

```python
# host 端：仅在 D 足够大时启用展开
W_IS_4 = (width == 4 and dim >= 1024)

# kernel 端：constexpr 分支
@triton.jit
def kernel(..., W_IS_4: tl.constexpr):
    if W_IS_4:
        # 展开：4 路独立乘加，消除 3 次 trans + 8 次 extract_slice
        out = x_s0*w0 + x_s1*w1 + x_s2*w2 + x_s3*w3
    else:
        # 通用路径：for 循环，保留 block_ptr 硬件加速
        for owi in tl.range(0, width):
            out += x_slice * w_slice
```

**阈值选择依据**（实测数据）：

| D | 加速比 | 策略 |
|---|--------|------|
| ≥ 2048 | 2.3x - 3.1x | 启用展开 |
| ≤ 256 | 0.41x - 0.75x | 回退到 for 循环 |

**退化根因**：小 D 场景 tile 数少（2-16 个），展开带来的转置节约不足以抵消独立标量 load 的启动开销。解决方案见 [memory_access_optimization.md §9](memory_access_optimization.md)（混合方案：block_ptr + extract_slice）。

经验：`exp_causal_conv1d_update_002`；详见 [ops/convolution.md](../ops/convolution.md)

---

### 1.6 指针基址预计算（循环不变量外提 / LICM）

将循环内重复计算的地址分量（base + start_offset + dim_offs * stride）提到循环外预计算为基址指针，循环内仅做简单偏移 `+ i * stride`：

```python
# 反模式：循环内重复计算不变地址分量
for i in range(seq_len):
    ptr = x_ptr + start * stride_t + d_offs * stride_d + i * stride_t
    x = tl.load(ptr, mask=mask)

# 正模式：循环不变量外提
x_ptrs = x_ptr + start * stride_t + d_offs * stride_d  # 外提
y_ptrs = y_ptr + start * stride_t + d_offs * stride_d
for i in range(seq_len):
    x = tl.load(x_ptrs + i * stride_t, mask=mask)       # 仅简单偏移
    tl.store(y_ptrs + i * stride_y, acc, mask=mask)
```

**收益**：seq_len=2048 时节省约 4096 次乘加地址计算。编译器通常能自动做部分 LICM，但显式外提能保证 Triton 编译器在 Ascend 后端不做保守假设。

**通用性**：适用于所有含循环体的 kernel（序列处理、matmul K 循环、normalization 列循环）。

经验：`exp_causal_conv1d_prefill_004`；详见 [ops/convolution.md](../ops/convolution.md)

---

### 1.7 Pad Slot 提前退出（无效计算跳过）

通过间接索引映射 + 单次检查，在 kernel 入口处跳过无效序列的完整计算：

```python
# indirect mapping: batch_id → cache_slot
cache_idx = tl.load(cache_indices_ptr + pid_batch)
if cache_idx == pad_slot_id:   # -1 = 无效填充槽
    return                      # 立即退出，释放 compute unit
```

此模式的基础开销仅 1 次 load + 1 次整数比较（~10 cycles），远小于执行完整 kernel 循环。batch 中有 ≥10% 填充序列时收益显著。

**通用性**：适用于所有使用间接索引映射的 kernel（KV-cache page table、embedding lookup with dynamic indices、MoE routing），只要能用单次 load + compare 判断是否需要跳过计算。

**与 constexpr 分支消除（§1.1）的区别**：constexpr 消除的是编译时已知路径（算子配置级）；pad slot 退出是运行时数据驱动的跳过（batch 构成级）。两者互补：constexpr 消除固定路径分支，pad slot 跳过动态无效数据。

经验：`exp_causal_conv1d_prefill_005`；详见 [ops/convolution.md](../ops/convolution.md)

---

### 1.8 冗余结构 mask 消除（矩阵性质保证的零元）

当矩阵具有**已知结构零元**（严格下三角、对角、上三角、带状等）时，对已为零元素施加的保护性 `tl.where(mask, val, 0.)` 是冗余 ALU 指令。Triton 编译器在运行时变量 `i` 驱动的循环中可能无法自动优化掉这类 mask：

```python
# 反模式：A 是严格下三角矩阵，A[i,j≥i] 本身就是 0
for i in range(2, BT):
    b_a = -tl.load(A + offset)
    b_a = tl.where(o_i < i, b_a, 0.)  # ← 冗余！j≥i 位置已为零

# 优化后：依赖矩阵数学性质保证零值
for i in range(2, BT):
    b_a = -tl.load(A + offset)
    # b_a[j≥i] 由严格下三角结构保证为 0
```

**识别条件**：
- 矩阵具有编译时已知的结构零元模式（严格下三角、上三角、对角、标准 mask 矩阵）
- `tl.where` 的 condition 与该结构零元条件一致
- 数据来源保证结构零元位置确实是 0（如从初始化到当前操作未改动）

**收益**：每条 `tl.where` 省 16 次 cmp + 16 次 select（16 元素向量），14 次迭代累计 448 条标量指令（实测 BT=16: 1.33x-1.87x）。

**注意**：此优化必须确认"矩阵结构零元在代码路径中始终为零"。如果中间操作可能向零元位置写入非零值，则不可删除。

经验：`exp_solve_tril_001`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

---

### 1.9 Constexpr 双路径（完整 tile vs 边界 tile 展开控制）

当 kernel 处理固定大小 tile 时，绝大多数 chunk 是完整 tile，仅最后一个可能是边界不完整 chunk。用 constexpr 分支实现双路径：

```python
neg_A = -b_A  # 预加载到寄存器

if T - chunk_offset >= BT:
    # 快速路径（99% chunk）：BT 是 constexpr → 编译器完全展开循环
    for i in range(2, BT):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, BLOCK), (1, 1))
        ...
else:
    # 慢速路径（边界 chunk）：运行时循环，但仍从寄存器读
    for i in range(2, min(BLOCK, T - chunk_offset)):
        b_a = tl.extract_slice(neg_A, (i, 0), (1, BLOCK), (1, 1))
        ...
```

**关键设计**：
- 快速路径 `range(2, BT)` 中 BT 是 constexpr → extract_slice 的 offset 参数为编译期常量，生成静态偏移（零 ALU）
- 慢速路径也使用 extract_slice（从寄存器读取，零 HBM 访问），与快速路径共享同一数据来源
- 两个路径的循环体代码可以完全一致（复用），区别仅在于编译优化程度

**被排除的方案**：统一运行时循环→失去 constexpr 展开优化，性能回退。

**通用性**：适用于所有以固定 tile 处理数据的 kernel，只要完整 tile 占绝大多数（≥95%）。

经验：`exp_solve_tril_003`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

---

## 2. 数值精度与累加器策略

### 2.1 fp32 累加器

归约、点积、LayerNorm 的中间累加必须用 fp32，避免 fp16/bf16 的累积误差：

```python
# 点积：累加器 fp32
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # fp32 累加
for k in range(0, K, BLOCK_K):
    a = tl.load(...).to(tl.float32)
    b = tl.load(...).to(tl.float32)
    acc += tl.dot(a, b)
tl.store(..., acc.to(tl.float16))  # 最后降精度输出

# 归约：fp32 中间
x = tl.load(..., mask=...).to(tl.float32)
mean = tl.sum(x) / n_elements        # fp32 sum
var  = tl.sum((x - mean) ** 2) / n_elements
```

**原则**：`dtype=tl.float32` 的 zeros/累加只在 UB 内，不影响 HBM 存储大小。

经验：`exp_stability_001`（LayerNorm fp32）；`exp_stability_002`（softmax 数值稳定）

---

### 2.2 exp overflow 防护（减去最大值）

softmax 等含 exp 的计算须先减去最大值：

```python
# 危险：直接 exp 可能 overflow（inf → NaN）
scores = q @ k.T
probs = tl.softmax(scores)  # 内部没有减 max 时 overflow

# 安全：先减 max（online softmax 模式）
m_i = tl.max(scores, axis=1)       # per-row max
scores_stable = scores - m_i[:, None]
exp_scores = tl.exp(scores_stable)  # 无 overflow 风险
```

经验：`exp_stability_003`；详见 [techniques/online_algorithms.md](online_algorithms.md)

---

### 2.3 rsqrt 优先于 sqrt + div

归一化中 `rsqrt` 是单条硬件指令，比 `sqrt + 1/x` 快且精度更高：

```python
# 低效：两步
std = tl.sqrt(var + eps)
x_norm = x / std

# 高效：单指令
rstd = tl.rsqrt(var + eps)
x_norm = x * rstd
```

经验：`exp_stability_004`

---

## 3. 类型转换优化

### 3.1 int64 → int32/fp32 避免 scalar fallback

Ascend 910b 上 int64 的比较和加法会退化为标量路径，导致 `aiv_scalar_ratio` 升高：

```python
# 危险：int64 地址计算（scalar fallback）
offset = row_id * stride_n + col_id  # 若结果 > 2^31，编译器会用 int64

# 安全：分步保持 int32 范围，或提前转 fp32
row_id_32 = row_id.to(tl.int32)
offset = row_id_32 * stride_n + col_id   # 保持 int32
```

**诊断**：profiling 中 `aiv_scalar_ratio` > 10% → 检查 int64 路径。

经验：`exp_scalar_cast_001`；详见 [techniques/data_types.md](data_types.md)

---

### 3.2 输入输出类型解耦

kernel 内部全程 fp32 计算，输入 fp16/bf16 在首次 load 后 upcast，输出在最后 store 前 downcast：

```python
@triton.jit
def kernel(x_ptr, out_ptr, ..., IN_DTYPE: tl.constexpr, OUT_DTYPE: tl.constexpr):
    x = tl.load(x_ptr + offs).to(tl.float32)    # upcast
    # ... 全程 fp32 计算 ...
    result_fp32 = compute(x)
    tl.store(out_ptr + offs, result_fp32.to(OUT_DTYPE))  # downcast
```

---

## 4. FMA 与指令合并

### 4.1 FMA（Fused Multiply-Add）

将 `a*b + c` 写成单一表达式，编译器会生成 FMA 指令（无中间寄存器，精度更高）：

```python
# 两条指令（编译器可能无法合并）
tmp = a * b
out = tmp + c

# 单条 FMA（明确触发）
out = tl.fma(a, b, c)     # 显式 FMA API
# 或直接写表达式
out = a * b + c            # 多数情况下 Triton 编译器自动 FMA
```

经验：`exp_rope_llama4_npu_001`（复数旋转 FMA）

---

### 4.3 tl.dot vs VEC 广播乘 — 微小矩阵的 CUBE 调度陷阱

对微小矩阵乘法（M=1 或 N=1 或 K≤16），使用 `tl.dot`（CUBE 单元）反而不如 VEC 单元广播乘+归约高效。CUBE 单元的调度和启动开销在微小尺寸下远大于 VEC 直接计算：

```python
# 反模式（14-20x 退化）：1×16 × 16×16 强行用 tl.dot
b_a_2d = tl.reshape(b_a, (1, 16))
b_a = tl.dot(b_a_2d, b_A)      # CUBE 调度开销 >> 计算量

# 正模式：VEC 广播乘 + 归约
b_a = b_a + tl.sum(b_a[:, None] * b_A, 0)  # b_a[16] × b_A[16,16] → VEC
```

**决策原则**：

| 矩阵规模 | 推荐方案 | 原因 |
|---------|---------|------|
| M≥16, N≥16, K≥16 | tl.dot（CUBE） | 计算量足够摊还 CUBE 调度开销 |
| M=1 或 N=1 或 K≤16 | VEC 广播乘 + sum 归约 | CUBE 调度开销 >> 微小计算量 |

经验：`exp_solve_tril_006`；详见 [ops/solve_tril.md](../ops/solve_tril.md)

---

### 4.2 减少中间寄存器

多步算术尽量合并为单一表达式，减少寄存器 spill：

```python
# 分步（3 个中间变量）
cos_x = x1 * cos
sin_x = x2 * sin
o1 = cos_x - sin_x

# 合并（1 个结果）
o1 = x1 * cos - x2 * sin   # 编译器可 FMA 并减少寄存器占用
```

---

经验：`exp_rope_llama4_npu_001`（复数旋转 FMA）

---

## 5. 标量循环 → 矩阵乘（提升 Cube 利用率）

当 kernel 中出现 `for j in range(BC)` 逐行标量循环，且循环体内是**标量×向量 broadcast 乘法**时，应考虑改为矩阵乘以利用 Cube 单元。

```python
# 反模式：BC 次标量循环
for j in range(BC):
    b_dAqk = tl.load(dAqk_local + o_dA + j, mask=m_dA, other=0)
    b_kj   = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
    m_i = o_i[:, None] >= j
    b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
    p_kj += H*K

# 正模式：一次性加载 BC×BC 块 + 2 次 tl.dot
b_dAqk_diag = tl.load(p_dAqk_diag, mask=dA_diag_mask, other=0.0)
b_k_exp = b_k * b_exp_neg_g
b_dq2 += tl.dot(b_dAqk_diag, b_k_exp) * b_exp_g
```

**适用条件**：
- 循环次数固定且较小（如 BC=16）
- 循环体内计算可表达为矩阵乘（`b_dA[BC,BC] @ b_k[BC,BK]`）
- 数值稳定性允许（如 gate 归一化点需保持一致）

**收益**：16 次标量循环 → 2 次 `tl.dot`，AIC MAC ratio 从 2.1% 量级显著提升。

经验：`exp_kda_bwd_scalar_loop_to_dot_001`

---

## 6. 运行时 min()/tl.cdiv 的标量消除

`min()` 和 `tl.cdiv()` 在运行时求值会产生标量分支或整数除法取余，在 Ascend 上标量降级。

### 6.1 min() → tl.where

```python
# 反模式
offset = min(BC // 2, T_local - i_ti - 1)

# 正模式
half_bc = BC // 2
remaining = T_local - i_ti - 1
offset = tl.where(half_bc < remaining, half_bc, remaining)
```

### 6.2 tl.cdiv → 编译期常量

```python
# 反模式
for i_k in range(tl.cdiv(K, BK)):
    ...

# 正模式（K/BK 为 constexpr）
NK = K // BK + (1 if K % BK != 0 else 0)
for i_k in range(NK):
    ...
```

经验：`exp_perf_013`、`exp_perf_014`

---

## 7. 移除冗余 compile_hint

对 block_ptr 确定性访存或普通 load/store 添加 `compile_hint(b_x, "mayDiscretememaccess")` 会限制编译器优化。

```python
# 反模式
b_h = tl.load(p_h, boundary_check=(0, 1))
tl.extra.cann.extension.compile_hint(b_h, "mayDiscretememaccess")

# 正模式
b_h = tl.load(p_h, boundary_check=(0, 1))
# 不加 hint
```

经验：`exp_ascend_compile_hint_001`；在 wy_dqkg_fused 上移除 b_dv hint 单独降低约 25% 延迟。

---

## 8. 循环不变量外提 / LICM

将循环内重复的地址分量、mask、stride 乘积提到循环外，减少每轮迭代的开销。

```python
# 反模式
for i_v in range(NV):
    p_v_new = v_new + BT_arange[:, None] * V + BV_arange
    ...

# 正模式
base_v_new = v_new + BT_arange[:, None] * V   # 外提
for i_v in range(NV):
    p_v_new = base_v_new + i_v * BV
    ...
```

常见可外提项：
- `BT_arange[:, None] * V` / `H*K` / `H*V` stride 预计算
- `BK_mask` / `BV_mask`（若 K/V 为 BK/BV 整数倍可完全消除）
- 基址指针 `base_ptr + batch_offset`

经验：`exp_kda_bwd_wy_structure_opt_001`

---

## 5. 分支与计算优化决策树

```
有运行时 if/else？
  └── 是 → 能否提升为 tl.constexpr？
        ├── 是 → 改为 constexpr 参数，完全消除分支
        └── 否 → 改用 tl.where（保持 SIMD 一致性）

有固定次数的小循环（W≤4 的卷积核、固定迭代次数）？
  ├── 是 → D 维度是否 ≥ 阈值（如 1024）？
  │       ├── 是 → constexpr 展开为多路独立表达式（§1.5）
  │       └── 否 → 保留 for 循环 + block_ptr（避免独立 load 退化）
  └── 否 → 保持通用 for 循环

有维度能被 tile size 整除？
  └── 是 → 加 DIM_IS_MULTIPLE / ALIGNED constexpr 消除 mask（§1.4）

有 exp/log/sqrt？
  ├── softmax 类 → 先减最大值再 exp（online softmax）
  ├── 归一化 → rsqrt 替代 sqrt+div
  └── 其他 → 检查是否会 overflow（x > 80 → exp → inf）

有累加/归约？
  └── 统一用 fp32 累加器，最后 downcast 到输出类型

有 int64 索引？
  └── 改为 int32 分步计算，或提前 fp32 偏移

有重复编译（kernel JIT 慢）？
  └── 频繁变化的标量参数加 do_not_specialize

有循环体内重复地址计算（base + start_offset + dim_offs * stride）？
  └── 外提为基址指针，循环内仅做 + i * stride（§1.6 LICM）

有变长 batch 中的 padding 序列？
  └── kernel 入口单次 load+cmp 检查 pad_slot_id，匹配立即 return（§1.7）

有已知结构零元的矩阵（三角、对角、带状）？
  └── 检查 tl.where(mask, ...) 的 mask 是否与结构零元条件一致 → 一致则删除（§1.8）

有运行时 min() / tl.cdiv()？
  └── 改为 tl.where 或编译期常量（§6）

有冗余 compile_hint？
  └── 移除，除非明确需要控制异步/离散访存（§7）

有循环体内重复地址计算 / stride 乘法？
  └── 外提到循环外（§8 LICM）

有完整 tile 占绝大多数（≥95%）的循环处理？
  └── constexpr 双路径：完整 tile 走 range(2, CONSTEXPR_BT)，边界走 range(2, min(BT, remaining))（§1.9）

有微小矩阵乘法（M=1 或 N=1 或 K≤16）？
  └── 优先 VEC 广播乘 + sum 归约，而非 tl.dot（§4.3）

有逐行标量循环且可表达为矩阵乘？
  └── 改为 tl.dot 提升 Cube 利用率（§5）
```

---

## 相关页面

- [techniques/data_types.md](data_types.md) — int64 scalar fallback 完整分析
- [techniques/online_algorithms.md](online_algorithms.md) — Online Softmax / 在线均值方差公式
- [techniques/ilp.md](ilp.md) — care_padding=False 消除同步屏障
- [hardware/ascend_patterns.md](../hardware/ascend_patterns.md) — constexpr 规范、禁用语法
- [workflows/debugging.md](../workflows/debugging.md) — 数值稳定性问题排查
- [ops/normalization.md](../ops/normalization.md) — fp32 累加器 + rsqrt 应用实例
- [ops/softmax.md](../ops/softmax.md) — exp overflow 防护实例
- [ops/rope.md](../ops/rope.md) — constexpr INVERSE/IS_NEOX_STYLE 实例
- [ops/convolution.md](../ops/convolution.md) — DIM_IS_MULTIPLE + W_IS_4 (D 阈值控制) 完整分析；指针基址预计算（LICM）；Pad slot 提前退出
- [ops/solve_tril.md](../ops/solve_tril.md) — 冗余结构 mask 消除（严格下三角）；constexpr 双路径；tl.dot vs VEC 广播乘决策
- [ops/ssm.md](../ops/ssm.md) — KDA/linear attention 反向：标量循环→tl.dot、Phase split、barrier 外提
- [techniques/memory_access_optimization.md](memory_access_optimization.md) — block_ptr vs scalar load 混合方案（§9）；preload + extract_slice 消除 HBM 重读（§11）
- [techniques/ilp.md](ilp.md) — 滑动窗口寄存器复用（小固定窗口序列通用模式）
