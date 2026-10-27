---
id: exp_split_qkv_rmsnorm_rope_opt_001
category: normalization
subcategory: fused_split_norm_rope
created: 2026-05-06
status: active
hardware: ascend910b2
dtype: bfloat16
---

# 经验：split_qkv_rmsnorm_rope 四合一优化（constexpr + int32 + shared_load + load_order）

## 适用条件

- 算子类型：QKV split + RMSNorm(Q,K) + RoPE(Q,K) + V passthrough 融合
- 数据类型：bfloat16
- 硬件：Ascend 910B2/B3/B4
- 形状：q_hidden=6144(48 heads), kv_hidden=1024(8 heads), head_dim=128, bsz=12
- GQA 结构：Q/K heads 不对称（HEAD_RATIO=6）

## 性能提升

- **基线 Triton → 优化后**：1.35x（0.0599ms → 0.0443ms）
- **PyTorch → 优化后**：4.43x（0.1964ms → 0.0443ms）

## 核心优化

### 优化 1：所有维度参数 constexpr 化（E2, T2）

**修改前**：
```python
def kernel(qkv_ptr, ..., q_hidden, kv_hidden, total_hidden, eps, HEAD_DIM: tl.constexpr, ...):
    # q_hidden, kv_hidden, total_hidden, eps 作为运行时标量
```

**修改后**：
```python
def kernel(qkv_ptr, ..., 
           q_hidden: tl.constexpr, 
           kv_hidden: tl.constexpr, 
           total_hidden: tl.constexpr, 
           eps: tl.constexpr, 
           HEAD_DIM: tl.constexpr, ...):
    # 编译器折叠常量偏移计算
```

**收益**：编译器在编译时展开 `k_start = q_hidden`、`v_start = q_hidden + kv_hidden` 等常量表达式，消除运行时分支。

### 优化 2：int32 offset 计算（T1, B1）

**修改前**：
```python
h = tl.arange(0, HALF_DIM)  # 默认 int64
batch_idx = (pid // Q_NUM_HEADS)  # int64
q_src = batch_idx * total_hidden + q_head_idx * HEAD_DIM  # int64 运算
```

**修改后**：
```python
h = tl.arange(0, HALF_DIM).to(tl.int32)  # 显式 int32
batch_idx = (pid // Q_NUM_HEADS).to(tl.int32)
q_src = (batch_idx * total_hidden + q_head_idx * HEAD_DIM).to(tl.int32)
```

**收益**：消除 int64 scalar fallback，aiv_scalar_ratio 从 >10% 降至 <3%，VecUtil 提升 8-15%。

### 优化 3：cos/sin 提前加载一次，Q/K 共用（E1, E3, B2）

**修改前**：
```python
# Q 处理
sin_lo = tl.load(sin_ptr + sc_base + h).to(tl.float32)
cos_lo = tl.load(cos_ptr + sc_base + h).to(tl.float32)
# ... Q RoPE 使用 sin_lo, cos_lo

# K 处理（在 if 分支内）
sin_lo = tl.load(sin_ptr + sc_base + h).to(tl.float32)  # 重复加载
cos_lo = tl.load(cos_ptr + sc_base + h).to(tl.float32)  # 重复加载
# ... K RoPE 使用 sin_lo, cos_lo
```

**修改后**：
```python
# 在 Q 处理前，提前加载 cos/sin（循环外）
sc_base = (batch_idx * HEAD_DIM).to(tl.int32)
sin_lo = tl.load(sin_ptr + sc_base + h).to(tl.float32)
sin_hi = tl.load(sin_ptr + sc_base + h + HALF_DIM).to(tl.float32)
cos_lo = tl.load(cos_ptr + sc_base + h).to(tl.float32)
cos_hi = tl.load(cos_ptr + sc_base + h + HALF_DIM).to(tl.float32)

# Q 处理：直接使用已加载的 sin_lo, cos_lo
q_rope_lo = (-qn_hi) * sin_lo + qn_lo * cos_lo

# K 处理：复用寄存器中的 sin_lo, cos_lo（无需重新加载）
k_rope_lo = (-kn_hi) * sin_lo + kn_lo * cos_lo
```

**收益**：消除冗余 HBM 读，aiv_mte2_time 下降 10-20%。

### 优化 4：load 顺序优化（E2, T3）

**修改前**：
```python
# Q 处理时才加载 q_weight/q_bias
qw_lo = tl.load(q_weight_ptr + h).to(tl.float32)
qb_lo = tl.load(q_bias_ptr + h).to(tl.float32)
```

**修改后**：
```python
# 在 cos/sin 加载后，立即加载 weight/bias（提前发射）
sin_lo = tl.load(sin_ptr + sc_base + h).to(tl.float32)
cos_lo = tl.load(cos_ptr + sc_base + h).to(tl.float32)
qw_lo = tl.load(q_weight_ptr + h).to(tl.float32)  # 提前
qb_lo = tl.load(q_bias_ptr + h).to(tl.float32)    # 提前

# Q 处理时直接使用（无需等待 load）
qn_lo = q_lo * rstd_q * qw_lo + qb_lo
```

**收益**：weight/bias load 与 Q data load 并行，隐藏内存延迟。

## 关键代码片段

```python
@triton.jit
def split_qkv_rmsnorm_rope_fused_kernel(
    qkv_ptr, sin_ptr, cos_ptr, q_weight_ptr, k_weight_ptr, q_bias_ptr, k_bias_ptr,
    q_out_ptr, k_out_ptr, v_out_ptr,
    q_hidden: tl.constexpr,      # 优化1: constexpr
    kv_hidden: tl.constexpr,
    total_hidden: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr, HALF_DIM: tl.constexpr,
    Q_NUM_HEADS: tl.constexpr, KV_NUM_HEADS: tl.constexpr, HEAD_RATIO: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = (pid // Q_NUM_HEADS).to(tl.int32)  # 优化2: int32
    q_head_idx = (pid % Q_NUM_HEADS).to(tl.int32)
    
    h = tl.arange(0, HALF_DIM).to(tl.int32)  # 优化2: int32
    
    # 优化3: cos/sin 提前加载一次
    sc_base = (batch_idx * HEAD_DIM).to(tl.int32)
    sin_lo = tl.load(sin_ptr + sc_base + h).to(tl.float32)
    sin_hi = tl.load(sin_ptr + sc_base + h + HALF_DIM).to(tl.float32)
    cos_lo = tl.load(cos_ptr + sc_base + h).to(tl.float32)
    cos_hi = tl.load(cos_ptr + sc_base + h + HALF_DIM).to(tl.float32)
    
    # 优化4: weight/bias 提前加载
    qw_lo = tl.load(q_weight_ptr + h).to(tl.float32)
    qw_hi = tl.load(q_weight_ptr + h + HALF_DIM).to(tl.float32)
    qb_lo = tl.load(q_bias_ptr + h).to(tl.float32)
    qb_hi = tl.load(q_bias_ptr + h + HALF_DIM).to(tl.float32)
    
    # Q 处理（使用已加载的 cos/sin/weight/bias）
    q_src = (batch_idx * total_hidden + q_head_idx * HEAD_DIM).to(tl.int32)
    q_lo = tl.load(qkv_ptr + q_src + h).to(tl.float32)
    q_hi = tl.load(qkv_ptr + q_src + h + HALF_DIM).to(tl.float32)
    var_q = (tl.sum(q_lo * q_lo, axis=0) + tl.sum(q_hi * q_hi, axis=0)) * (1.0 / HEAD_DIM)
    rstd_q = tl.math.rsqrt(var_q + eps)
    qn_lo = q_lo * rstd_q * qw_lo + qb_lo
    qn_hi = q_hi * rstd_q * qw_hi + qb_hi
    q_rope_lo = (-qn_hi) * sin_lo + qn_lo * cos_lo  # 复用 sin_lo, cos_lo
    q_rope_hi = qn_lo * sin_hi + qn_hi * cos_hi
    
    # K 处理（复用 cos/sin，无需重新加载）
    if q_head_idx % HEAD_RATIO == 0:
        k_head_idx = (q_head_idx // HEAD_RATIO).to(tl.int32)
        kw_lo = tl.load(k_weight_ptr + h).to(tl.float32)
        # ... K RMSNorm
        k_rope_lo = (-kn_hi) * sin_lo + kn_lo * cos_lo  # 复用 sin_lo
```

## 适用场景

- QKV split + 分别 RMSNorm + RoPE 融合算子
- GQA 结构（Q/K heads 数量不同）
- 多个 tensor 共享同一位置编码（cos/sin）
- head_dim ≥ 64，bfloat16/fp16

## 注意事项

1. **constexpr 化前提**：确认参数在 kernel 生命周期内不变
2. **int32 安全范围**：最大 offset = bsz × total_hidden < 2^31（本例 12×8192=98304，安全）
3. **cos/sin 寄存器驻留**：确保 UB 容量足够（本例 sin/cos 共 512B，远低于 192KB）
4. **load 顺序调整**：确保无数据依赖冲突

## 相关知识单元

- E2: exp_fused_split_qk_norm_001（constexpr + 顺序 load，35%）
- E3: exp_rope_npu_001（cos/sin 寄存器驻留，35%）
- T1: int32 offset 分步计算（8-15%）
- T2: constexpr 化高频参数（5-12%）
- T3: HBM 地址递增 load 顺序（10-20%）

## 测试数据

| 配置 | 延迟 | 加速比 |
|------|------|--------|
| PyTorch 基线 | 0.1964 ms | 1.0x |
| Triton 基线（未优化） | 0.0599 ms | 3.28x |
| Triton 优化后 | 0.0443 ms | **4.43x** |
| 优化提升（vs Triton 基线） | - | **1.35x** |

## 来源

- Session: split_qkv_rmsnorm_rope_bf16_run1
- 工作目录: triton_ascend_output/op_0_split_qkv_rmsnorm_rope_20260505_2328_9409
- 生成时间: 2026-05-06
- 知识库检索: perf_tuning__VecUtil_low_rmsnorm_rope_constexpr_load_order.md
