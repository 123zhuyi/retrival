---
query_type: perf_tuning
query_keywords: [VecUtil_low, rmsnorm_rope, constexpr, int32_offset, load_order]
created: 2026-05-05
last_updated: 2026-06-02T14:40:00
hit_count: 3
context_summary: "split_qkv_rmsnorm_rope_fused_kernel 性能优化：VecUtil_low 瓶颈，聚焦 constexpr 参数化、int32 offset 向量化、load 指令重排序，q_hidden=6144(48heads), kv_hidden=1024(8heads), head_dim=128, bf16, 910b2"
top_units: [E1, E2, E3, T1, T2]
stale: false
---

## 检索问题（Problem Statement）

性能优化：split_qkv_rmsnorm_rope 融合 kernel 已达 3.42x 加速，寻找进一步优化点

**触发上下文**：瓶颈分析 — implementation_latency=0.058ms，kernel=split_qkv_rmsnorm_rope_fused_kernel 占 58%，q_hidden=6144(48heads), kv_hidden=1024(8heads), head_dim=128, bf16  
**检索范围**：VecUtil_low：标量参数未 constexpr 化、int32 offset 计算、load 顺序未优化

---

## 知识单元列表

### 硬件约束知识单元（C 系列）

**C1** `[硬件约束]` UB 容量与 VecUtil 关系
- **内容**：head_dim=128，bf16 → 128×2=256B，单 program 持有 Q+K+V head 数据时最大约 3×256B=768B，远低于 192KB 上限。说明当前 VecUtil_low 不是 UB 溢出导致，而是指令级并行不足或标量路径占用。
- **来源**：`pages/hardware/constraints_cheatsheet.md C1`

**C2** `[硬件约束]` int64 scalar fallback（aiv_scalar_ratio 根因）
- **内容**：Ascend 910B 上 int64 的 CMP/ADD/MUL 退化为标量路径。`tl.arange(0, HEAD_DIM)` 默认返回 int64。若 offset 计算为 `pid * q_hidden + head_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)`，该表达式结果为 int64，会触发 scalar fallback，导致 `aiv_scalar_ratio` 升高、VecUtil 降低。
- **诊断**：`aiv_scalar_ratio > 10%` 直接检查 int64 路径
- **来源**：`pages/techniques/data_types.md` + `pages/hardware/constraints_cheatsheet.md C5`

**C3** `[性能约束]` 运行时分支 vs constexpr 分支
- **内容**：若 `Q_HIDDEN`、`KV_HIDDEN`、`HEAD_DIM` 等参数作为普通标量传入，编译器无法折叠循环边界和偏移计算，生成冗余指令。将其提升为 `tl.constexpr` 后，编译器能在编译时展开常量表达式，消除运行时分支。
- **来源**：`pages/techniques/compute_and_branch.md §1.1`

### 瓶颈预判知识单元（B 系列）

**B1** `[潜在瓶颈]` VecUtil_low — int64 标量偏移计算
- **触发条件**：kernel 中存在 `pid * stride_batch + head_idx * HEAD_DIM + arange(0, HEAD_DIM)` 形式的地址计算，若 `stride_batch = q_hidden + kv_hidden + v_hidden = 8192`，则 `pid(最大575) × 8192 = 4,718,592 > 2^21`，中间结果仍在 int32 范围，但 tl.arange 默认 int64 导致整体升级。
- **症状**：profiling 中 `aiv_scalar_ratio > 10%`，`aiv_vec_ratio < 65%`
- **修复方向**：`.to(tl.int32)` 强制截断 + int32 分步计算
- **关联经验**：exp_perf_007, exp_rope_npu_001

**B2** `[潜在瓶颈]` cos/sin 重复 HBM 加载
- **触发条件**：Q head 和 K head 分别加载 cos/sin（即 sin/cos 在 Q 处理和 K 处理中各加载一次）
- **症状**：`aiv_mte2_time` 偏高，内存带宽利用率高于预期
- **修复方向**：cos/sin 在 Q 处理前加载一次，Q 和 K 处理中共用寄存器内的 cos/sin 值
- **关联经验**：exp_fused_qkv_rmsnorm_rope_001（redundant_load_optimization）, exp_rope_npu_001

**B3** `[潜在瓶颈]` RMSNorm weight 重复加载
- **触发条件**：Q weight（6144 维）和 K weight（1024 维）如果每个 program 都从 HBM 加载对应 head_dim 切片，则 Q weight 加载 48次×128元素，K weight 加载 8次（或按 pid 全量加载）
- **症状**：weight load 成为 MTE2 热点
- **修复方向**：每个 program 只加载本 head 对应的 128 元素 weight 切片（而非全量），确认偏移计算正确

**B4** `[潜在瓶颈]` load 顺序非连续
- **触发条件**：若 Q/K/V 的 load 指令不按 HBM 地址顺序排列（如 Q→V→K），会破坏 prefetch 效果
- **修复方向**：按 HBM 地址递增顺序安排 load：Q（offset=0）→ K（offset=q_hidden）→ V（offset=q_hidden+kv_hidden），使 prefetcher 预判命中

**B5** `[潜在瓶颈]` 非 constexpr 的标量参数阻塞编译优化
- **触发条件**：`q_hidden`、`kv_hidden`、`HEAD_DIM` 作为运行时标量传入时，编译器必须生成动态偏移计算代码，无法在编译期折叠 `k_start = q_hidden` 或 `v_start = q_hidden + kv_hidden` 等常量
- **修复方向**：将不随 batch 变化的参数声明为 `tl.constexpr`

### 调优经验知识单元（E 系列）

**E1** `[调优经验]` `exp_fused_qkv_rmsnorm_rope_001` — cos/sin 冗余加载消除
- **适用条件**：QKV split + RMSNorm + RoPE 融合，Q/K 共享同一 token 位置的 cos/sin，bf16，head_dim≥64
- **性能提升**：25%（冗余 HBM 读压消除后 MTE2 占比下降）
- **核心优化**：Q 和 K 的 RoPE 使用同一 token 位置的 cos/sin，应在处理 Q 前加载一次 pos_idx/cos/sin，K 处理时直接复用寄存器值
- **适用阶段**：load 顺序重组 + 寄存器驻留 cos/sin
- **经验文件**：`raw/experiences/normalization/exp_fused_qkv_rmsnorm_rope_001.yaml`
- **关键字段**：`redundant_load_optimization` — 加载 pos_idx、cos、sin 移至 Q/K 循环外

**E2** `[调优经验]` `exp_fused_split_qk_norm_001` — constexpr 分支消除 + 顺序 load
- **适用条件**：Q/K 分别 RMSNorm（含 bias 开关），LoRA rank 场景，fp16/bf16，head_dim≥64
- **性能提升**：35%
- **核心优化**：`Q_HAS_BIAS: tl.constexpr` + `K_HAS_BIAS: tl.constexpr` 消除运行时分支；**严格顺序加载** Q segment → K-nope segment → K-pe segment，最大化 DRAM burst 命中
- **适用阶段**：kernel 参数声明重构 + load 顺序调整
- **经验文件**：`raw/experiences/normalization/exp_fused_split_qk_norm_001.yaml`
- **关键字段**：`compile_time_branching`（constexpr）+ `memory_access_pattern`（顺序 load）

**E3** `[调优经验]` `exp_rope_npu_001` — cos/sin 寄存器驻留 + 软流水
- **适用条件**：Ascend NPU RoPE，head_dim≥64，fp16/bf16，多 head 场景
- **性能提升**：35%
- **核心优化**：`cos_sin_reuse` — cos/sin 在外层（token 级）加载一次，内层所有 head 复用；`software_pipelining` — `num_stages=2` 使下一 token 的 load 与当前计算重叠
- **适用阶段**：将 cos/sin load 提至最外层 + `tl.range(..., num_stages=2)`
- **经验文件**：`raw/experiences/rope/exp_rope_npu_001.yaml`

**E4** `[调优经验]` `exp_rmsnorm_rope_001` — sub_blocking + V passthrough 重叠
- **适用条件**：RMSNorm + RoPE 融合，fp16/bf16，head_dim≥64，910b2/910b3
- **性能提升**：25%
- **核心优化**：`sub_blocking_strategy` — 用 `dl.parallel` 将 BLOCK_SIZE 分为 SUB_BLK，提升寄存器复用；`v_tensor_passthrough` — V tensor 的 load/store 与 Q/K 的 RMSNorm+RoPE 计算**重叠**，隐藏 V 的内存延迟
- **适用阶段**：V passthrough 提前 load，与 Q/K 计算交错排列
- **经验文件**：`raw/experiences/normalization/exp_rmsnorm_rope_001.yaml`

**E5** `[调优经验]` `exp_fused_norm_rope_001` — grid-stride + 精确 fp32 控制
- **适用条件**：RMSNorm + 部分 RoPE，seq_len 任意，num_heads≥1，head_dim≥64，fp16/bf16
- **性能提升**：25%
- **核心优化**：`grid_stride_scheduling` — grid-stride loop 处理任意 seq_len，避免 tail wave；`precision_management` — variance 计算用 fp32，其余向量操作保持 bf16，最大化吞吐
- **经验文件**：`raw/experiences/normalization/exp_fused_norm_rope_001.yaml`

### 调优策略知识单元（T 系列）

**T1** `[调优策略]` int32 分步 offset 消除 scalar fallback
- **适用场景**：任何 kernel 中存在大数值 pid × stride 的地址计算（stride > 1024）
- **核心方法**：
  1. `offs = tl.arange(0, HEAD_DIM).to(tl.int32)` — 强制 arange 为 int32
  2. `batch_offset = (batch_idx * stride_batch).to(tl.int32)` — 单独截断高位乘法
  3. `q_offset = batch_offset + (q_head_idx * HEAD_DIM).to(tl.int32)` — 分步累加
- **预期收益**：`aiv_scalar_ratio` 从 >10% 降至 <3%，VecUtil 提升 8-15%
- **关键参数**：`stride_batch = q_hidden + kv_hidden + v_hidden`（本算子 = 8192），`pid_max = bsz * q_heads = 576`，最大 offset = 576×8192 = 4,718,592，int32 范围内（<2^31）
- **页面**：`pages/techniques/data_types.md §3.1` + `pages/techniques/compute_and_branch.md §3.1`

**T2** `[调优策略]` constexpr 化高频参数（Q_HIDDEN / KV_HIDDEN / HEAD_DIM）
- **适用场景**：融合 kernel 中不随 batch 维变化的整数维度参数
- **核心方法**：将 `Q_HIDDEN`、`KV_HIDDEN`、`HEAD_DIM`、`HALF_DIM`、`Q_NUM_HEADS`、`KV_NUM_HEADS` 全部声明为 `tl.constexpr`
- **收益**：
  - 编译器在编译时折叠 `k_start = Q_HIDDEN`、`v_start = Q_HIDDEN + KV_HIDDEN`，消除运行时整数加法
  - if-else（如 `HAS_BIAS`）退化为单一代码路径，消除分支预测开销
  - autotune 候选数减少（同一参数集只编译一次）
- **预期收益**：5-12% 整体延迟降低
- **页面**：`pages/techniques/compute_and_branch.md §1.1`

**T3** `[调优策略]` HBM 地址递增 load 顺序（连续 prefetch）
- **适用场景**：同一 program 内需加载多个连续内存段（Q/K/V + weight + cos/sin）
- **核心方法**：按 HBM 物理地址升序排列所有 `tl.load` 调用：
  1. `qkv_ptr + batch_offset + [0 .. q_hidden)` → Q segment
  2. `qkv_ptr + batch_offset + q_hidden + [k_head_idx*HEAD_DIM .. (k_head_idx+1)*HEAD_DIM)` → K segment
  3. `qkv_ptr + batch_offset + q_hidden + kv_hidden + [v_head_idx*HEAD_DIM ...]` → V segment
  4. `cos_ptr + batch_idx*HEAD_DIM`、`sin_ptr + batch_idx*HEAD_DIM` → 仅一次
  5. `weight_ptr + head_idx*HEAD_DIM` → Q/K weight 在 Q/K 数据 load 后立即加载
- **预期收益**：prefetcher 命中率提升，`aiv_mte2_time` 下降 10-20%
- **页面**：`pages/techniques/memory_access_optimization.md §6`

**T4** `[调优策略]` care_padding=False 消除 MTE2→Vector 同步屏障
- **适用场景**：head_dim=128 是 2 的幂，arange(0, HEAD_DIM) 不需要边界 mask 时
- **核心方法**：将所有 `tl.load(ptr + offs, mask=offs < HEAD_DIM)` 改为 `tl.load(ptr + offs, care_padding=False)`（无需 mask 时）或在 store 处过滤时使用 `care_padding=False`
- **注意**：HEAD_DIM=128 是 2 的幂，无 boundary 问题，可安全去除 mask + care_padding
- **预期收益**：消除 MTE2→Vector 同步屏障，VecUtil 提升 5-10%
- **页面**：`pages/techniques/ilp.md §Solution 1`

**T5** `[调优策略]` V passthrough 与 Q/K 计算交错（内存延迟隐藏）
- **适用场景**：V segment 不做 RMSNorm/RoPE，仅做 load+store passthrough，但这部分内存访问可以与 Q/K 计算重叠
- **核心方法**：在 Q/K RMSNorm 计算（compute bound）期间，提前发出 V 的 load 指令，使 MTE2(V) 与 Vector(Q norm) 并行
- **代码结构**：
  ```
  # 先 load V（发射 prefetch）
  v = tl.load(v_ptr + offs)
  # 然后做 Q RMSNorm+RoPE（计算密集）
  var_q = tl.sum(q * q) / HEAD_DIM; ...
  # 然后做 K RMSNorm+RoPE
  var_k = tl.sum(k * k) / HEAD_DIM; ...
  # 最后 store（此时 V 已在寄存器）
  tl.store(v_out + offs, v)
  ```
- **预期收益**：V passthrough 延迟被 Q/K 计算隐藏，净收益 5-15%
- **页面**：`pages/techniques/memory_access_optimization.md §4.1`

**T6** `[调优策略]` do_not_specialize 避免 seq_len/batch_size 触发重编译
- **适用场景**：`bsz`、`seq_len` 等在不同推理请求间频繁变化的参数
- **核心方法**：`@triton.jit(do_not_specialize=["bsz"])` 防止每个 batch size 都触发 JIT 重编译，降低首次推理延迟
- **页面**：`pages/techniques/compute_and_branch.md §1.2`

### API 知识单元（A 系列）

**A1** `[API用法]` tl.arange → int32 强制类型
- **内容**：`offs = tl.arange(0, HEAD_DIM).to(tl.int32)` — HEAD_DIM=128 是 constexpr，arange 结果在编译期已知范围，`.to(tl.int32)` 安全且消除 int64 scalar 路径
- **来源**：`pages/techniques/data_types.md §Common Pattern`

**A2** `[API用法]` tl.constexpr 参数声明
- **内容**：`HEAD_DIM: tl.constexpr, Q_HIDDEN: tl.constexpr, KV_HIDDEN: tl.constexpr, Q_NUM_HEADS: tl.constexpr, KV_NUM_HEADS: tl.constexpr, HALF_DIM: tl.constexpr`。所有在 kernel 启动前已知且不变的整数维度均应声明为 constexpr。
- **来源**：`pages/techniques/compute_and_branch.md §1.1` + `pages/api/triton_api_cheatsheet.md A10`

**A3** `[API用法]` tl.load care_padding=False
- **内容**：当 HEAD_DIM=128（2的幂）且无边界越界风险时，去掉 mask 或设 `care_padding=False`，消除 MTE2→Vector 同步屏障
- **安全条件**：offs 恒 < HEAD_DIM（因 HEAD_DIM 是 2 的幂且 arange 范围恰好覆盖）
- **来源**：`pages/techniques/ilp.md §Solution 1`

**A4** `[API文档]` dl.compile_hint — multibuffer 显式开启
- **内容**：`dl.compile_hint("multibuffer=true")` 指示编译器强制开启双缓冲，使 load 与 compute 重叠；当 kernel 自动推断不足时显式使用
- **来源**：`pages/techniques/ilp.md §Compiler Hints`

### 代码骨架知识单元（S 系列）

**S1** `[代码骨架]` int32 offset 分步计算示例（适配本算子）
```python
@triton.jit
def split_qkv_rmsnorm_rope_fused_kernel(
    qkv_ptr, q_out_ptr, k_out_ptr, v_out_ptr,
    sin_ptr, cos_ptr, q_weight_ptr, k_weight_ptr,
    bsz, stride_batch,
    Q_NUM_HEADS: tl.constexpr,  # ← constexpr
    KV_NUM_HEADS: tl.constexpr,  # ← constexpr
    Q_HIDDEN: tl.constexpr,     # ← constexpr
    KV_HIDDEN: tl.constexpr,    # ← constexpr
    HEAD_DIM: tl.constexpr,     # ← constexpr
    HALF_DIM: tl.constexpr,     # ← constexpr
    eps,
):
    pid = tl.program_id(0)
    batch_idx = pid // Q_NUM_HEADS
    q_head_idx = pid % Q_NUM_HEADS
    k_head_idx = q_head_idx % KV_NUM_HEADS

    # int32 分步 offset（避免 int64 scalar fallback）
    offs_hd = tl.arange(0, HEAD_DIM).to(tl.int32)
    batch_off = (batch_idx * stride_batch).to(tl.int32)

    # 1. 提前发射 V load（延迟隐藏）
    v_start = (Q_HIDDEN + KV_HIDDEN + k_head_idx * HEAD_DIM).to(tl.int32)
    v = tl.load(qkv_ptr + batch_off + v_start + offs_hd)  # care_padding 可省

    # 2. Load cos/sin 一次（Q/K 共用）
    sin_v = tl.load(sin_ptr + batch_idx * HEAD_DIM + offs_hd).to(tl.float32)
    cos_v = tl.load(cos_ptr + batch_idx * HEAD_DIM + offs_hd).to(tl.float32)

    # 3. Load Q + weight → RMSNorm + RoPE
    q_start = (q_head_idx * HEAD_DIM).to(tl.int32)
    q = tl.load(qkv_ptr + batch_off + q_start + offs_hd).to(tl.float32)
    w_q = tl.load(q_weight_ptr + q_start + offs_hd).to(tl.float32)
    var_q = tl.sum(q * q) / HEAD_DIM
    q_norm = q * tl.rsqrt(var_q + eps) * w_q
    # RoPE (分段 lo/hi 避免负偏移)
    q_lo = tl.load(qkv_ptr + batch_off + q_start + tl.arange(0, HALF_DIM).to(tl.int32)).to(tl.float32)
    q_hi = tl.load(qkv_ptr + batch_off + q_start + (tl.arange(0, HALF_DIM) + HALF_DIM).to(tl.int32)).to(tl.float32)
    q_rope_lo = q_lo * cos_v[:HALF_DIM] - q_hi * sin_v[:HALF_DIM]
    q_rope_hi = q_hi * cos_v[HALF_DIM:] + q_lo * sin_v[HALF_DIM:]

    # 4. Load K + weight → RMSNorm + RoPE（cos/sin 复用）
    k_start = (Q_HIDDEN + k_head_idx * HEAD_DIM).to(tl.int32)
    k = tl.load(qkv_ptr + batch_off + k_start + offs_hd).to(tl.float32)
    w_k = tl.load(k_weight_ptr + (k_head_idx * HEAD_DIM).to(tl.int32) + offs_hd).to(tl.float32)
    var_k = tl.sum(k * k) / HEAD_DIM
    k_norm = k * tl.rsqrt(var_k + eps) * w_k
    # RoPE on K（同一 cos/sin）
    k_lo = tl.load(qkv_ptr + batch_off + k_start + tl.arange(0, HALF_DIM).to(tl.int32)).to(tl.float32)
    k_hi = tl.load(qkv_ptr + batch_off + k_start + (tl.arange(0, HALF_DIM) + HALF_DIM).to(tl.int32)).to(tl.float32)
    k_rope_lo = k_lo * cos_v[:HALF_DIM] - k_hi * sin_v[:HALF_DIM]
    k_rope_hi = k_hi * cos_v[HALF_DIM:] + k_lo * sin_v[HALF_DIM:]

    # 5. Store outputs
    tl.store(q_out_ptr + pid * HEAD_DIM + tl.arange(0, HALF_DIM).to(tl.int32), q_rope_lo.to(tl.bfloat16))
    tl.store(q_out_ptr + pid * HEAD_DIM + (tl.arange(0, HALF_DIM) + HALF_DIM).to(tl.int32), q_rope_hi.to(tl.bfloat16))
    tl.store(k_out_ptr + (batch_idx * KV_NUM_HEADS + k_head_idx) * HEAD_DIM + offs_hd, k_rope_lo.to(tl.bfloat16))  # lo
    # ...
    tl.store(v_out_ptr + (batch_idx * KV_NUM_HEADS + k_head_idx) * HEAD_DIM + offs_hd, v)
```
- **关键设计点**：int32 offset + constexpr 参数 + V 提前 load + cos/sin 一次加载
- **来源**：综合 exp_fused_qkv_rmsnorm_rope_001 + exp_rope_npu_001 + exp_fused_split_qk_norm_001

### 矛盾提示知识单元（X 系列）

**X1** `[矛盾提示]` constexpr 与 autotune 的兼容性
- **内容**：将 `Q_HIDDEN`、`KV_HIDDEN` 声明为 constexpr 后，autotune 不能将这些参数列入候选配置，仅 `BLOCK_T` 等实际 tile 参数可 autotune。本算子 batch=12、head_per_program=1，constexpr 化维度参数不影响 autotune 覆盖度，安全。

**X2** `[矛盾提示]` int32 overflow 边界
- **内容**：最大 offset = `bsz(12) × stride_batch(8192) + q_hidden(6144) = 98304 + 6144 = 104448`，远低于 int32 上限（2^31=2,147,483,648）。int32 分步计算完全安全，无需 int64。

**X3** `[矛盾提示]` RMSNorm 已 load 的 Q/K 与 RoPE 再次 load 的 lo/hi 重复
- **内容**：骨架中 RMSNorm 用 `q = load(q_start + offs_hd)` 完整加载 128 元素，RoPE 又分两次 load lo[0:64] 和 hi[64:128]。可优化为：用 q_norm（已在寄存器）直接拆 lo/hi，避免重复 HBM 读取。
- **修复**：`q_lo = q_norm[:HALF_DIM]; q_hi = q_norm[HALF_DIM:]`（需 tl.split 或 slice 操作）

---

## Top-3 调优经验（按 final_rank 排序）

### 第一名：E2 — `exp_fused_split_qk_norm_001`（constexpr + 顺序 load）
- **推荐理由**：直接命中 VecUtil_low 的两大根因 — constexpr 分支消除 + 严格顺序 load。性能提升 35%，适用条件完全匹配（Q/K 分别 RMSNorm，bf16，head_dim=128）
- **立即行动**：将 `Q_HAS_BIAS`、`K_HAS_BIAS`、`Q_HIDDEN`、`KV_HIDDEN`、`HEAD_DIM` 全部声明为 `tl.constexpr`；调整 load 顺序为 Q→K→V（或 V→cos/sin→Q→K 以隐藏 V 延迟）

### 第二名：E1 — `exp_fused_qkv_rmsnorm_rope_001`（cos/sin 冗余加载消除）
- **推荐理由**：`redundant_load_optimization` 直接命中 B2 瓶颈。cos/sin 从每次 Q/K 各加载一次（2次 HBM 读）改为一次加载，对 MTE2 压力显著改善
- **立即行动**：将 cos/sin load 提至 Q 处理之前，K 处理时直接引用已在寄存器中的 sin_v/cos_v

### 第三名：T1 — int32 offset 分步计算（消除 scalar fallback）
- **推荐理由**：aiv_scalar_ratio 是 VecUtil_low 的最常见根因，而 `tl.arange` 默认 int64 在 Ascend 910B 上直接导致 scalar fallback。修复成本极低（仅添加 `.to(tl.int32)`），但收益稳定（VecUtil +8-15%）
- **立即行动**：将所有 `tl.arange(0, HEAD_DIM)` 改为 `.to(tl.int32)`，所有 pid × stride 的中间计算显式 `.to(tl.int32)` 截断

---

## 复用提示

- 相似检索场景：`perf_tuning` + `VecUtil_low` + 任意 normalization+rope 融合算子 → 本缓存 90% 适用
- 差异场景：若瓶颈为 CubeUtil_low（matmul 场景）→ 参考 `perf_tuning__CubeUtil_low_matmul_diagonal_tiling.md`
- 若 aiv_scalar_ratio < 5% 但 VecUtil 仍低 → 考虑 for-loop tiling（`pages/techniques/ilp.md §Solution 2`）
