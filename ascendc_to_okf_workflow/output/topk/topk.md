---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: TopK（沿轴选 k 极值，选择/排序家族首卡）Ascend C → Triton 语义转化卡
description: values,indices=topk(x,k,dim,largest,sorted,indices_dtype)——沿 dim 轴每行选 k 个极值，值 dtype=x、索引 dtype={i32,i64}。官方三算法族（radix 多趟基数排序 / bitonic 小 k / merge sort）+ 五模式分核（SINGLE_CORE/MULT_CORE_OPTIM/SORT_AND_TOP_K/NON_LAST_SMALL_AXIS）+ sort_with_index 联合排序基座。三族在仓无半壳，唯一外部依赖 AscendC::Sort 库调用（golden 语义补偿）。Triton 主路线=发射级迭代 max+argmax 选择：v2 kernel 内动态 k 循环被后端 scf.for 缺陷挂死（vector core timeout 507034）→ 改 wrapper 循环 k 次发射无环单选 kernel + torch.scatter_ 掩码（count-insens：tl.topk/sort 因 vsort 仅 f16/f32 且 int64 编译失败废弃）。全序键=值位模式零扩展 int64（浮点 sign-flip 保序/NaN=max；u64 ^INT64_MIN），largest=False ~key 序反转。tie 左most。②门关键防坑：两行段 tile kernel 取值须用局部 lane 比对（offs==m 全局比对 tb>0 恒假→v=0）、非尾轴 movedim 后必须 contiguous 化（reshape 保留 transpose 步幅→kernel 行主序读错行）、NPU scatter_ 不支持 u64（走 int64 位视图）。
tags: [topk, top_k_v2, selection, radix_sort, bitonic_sort, merge_sort, iterative_max_argmax, multi_core_optim, sort_and_top_k, launch_iteration, key_mapping, nan_max, tie_leftmost, argmax, vsort_limit, scatter_limit, persist_core, ascend950, ascend910b, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-math/math/top_k_v2/（CANN ops-math 本地快照，arch35 radix/bitonic/merge 三族 + tiling 双文件 + golden.py）
created_at: "2026-09-18T00:00:00Z"
updated_at: "2026-09-20T00:00:00Z"
---

sources:
  - url: src/ops-math/math/top_k_v2/op_kernel/arch35/radix_sort_top_k.h
    role: primary
  - url: src/ops-math/math/top_k_v2/op_kernel/arch35/top_k_small_size_bitonic_sort.h
    role: primary
  - url: src/ops-math/math/top_k_v2/op_host/top_k_v2_tiling_base.cpp
    role: primary
  - url: src/ops-math/math/top_k_v2/tests/assets/golden.py
    role: primary
  - url: src/ops-math/math/topk/op_api/topk.cpp
    role: secondary

# TopK（沿轴选 k 个极值）

沿 dim 轴（负轴归一到尾轴）对 x 每行选 k 个最值。def 表：x∈{i8/u8/i16/u16/i32/u32/i64/u64/f16/f32/bf16}（11 种），1-8 维 ND；**k 为 0D tensor 输入**（i32/i64，ValueDepend）；values dtype=x、indices dtype=indices_dtype（i32/i64，默认 i32）；Attr sorted=true/dim=-1/largest=true/indices_dtype=DT_INT32/sort_policy=0。对应 `torch.topk`。**aclnn 三岔分派**：超尺寸走 AICPU；**f16/bf16+N≫k+910b/910_93 走姊妹算子 radix_top_k**（基数排序）；else AICPU。确定性问题：官方 tiling 满核约束 + MULT_CORE_OPTIM 二段局选归并。

## 1. 功能与全序键（golden `_radix_keys` / `_value_better` 同构）

- `values[r,j]`=选中值（dtype 同 x）；`indices[r,j]`=**轴内原始下标**（非排序后位置）。largest=True 从大到小，False 从小到大，sorted=True 有序输出；sorted=False 无序（骨架恒输出有序=超集合法）。
- **全序键**（比较域，值本体不改动，选择骨架无精度变换）：
  - 浮点（f32/f16/bf16）：位模式 sign-flip——`u = bits ^ (sign ? 0x7FFFFFFF : 0x80000000)`（16bit 用 0x7FFF/0x8000）；**NaN → u=全 1（最大键）**；-0.0/+0.0 键不同但比较相邻（golden signed_zero 登记）。
  - uint64：`u = bits ^ INT64_MIN`（有符号容器保序）。
  - int8~int64：零/符号扩展即有符号序；uint8~uint32：零扩展 + ^SIGN 转有符号序。
  - **统一实现**：值位模式零扩展 int64 无符号序键 u64，浮点先 sign-flip，容器内 `us = u64 ^ SIGN_BIT64` 得 signed-order。largest=False 只需 `~key` 序反转（双射），NaN 两向均正确（largest 先选 / False 后选）。
- **tie 语义**：`_value_better` 严格比较（largest 用 `>`）→ 稳定 → 等值子集保持输入序 → **输出序 = 输入序（左most 先出）**。实现：argmax 键 = `BLOCK - offs`（左大右小）夺左most。

## 2. 原型与约束

| 项 | 值 |
|---|---|
| x | 11 dtype（i8~u64/f16/f32/bf16），ND 1-8 维 |
| k | **0D tensor 输入**（i32/i64，ValueDepend，host 读标量），k∈[0,N] |
| values | dtype=x，shape=x 轴维→k |
| indices | dtype=indices_dtype（i32/i64 默认 i32），shape 同 values |
| attrs | sorted=true、dim=-1、largest=true、indices_dtype=3、sort_policy=0 |
| 特殊 | k 越界抛错（golden ValueError）；M=0 → 空输出；k=0 → 空；k=N → 全排等价（应走 sort 卡） |

**aclnn 分面（登记不译）**：IsAiCoreSupport（超限走 AICPU）、IsRadixTopKSupport（f16/bf16+N≫k+910b 走姊妹 radix_top_k）——姊妹算子策略不映射，Triton 统一迭代选择覆盖其语义。

## 3. 官方算法族 → Triton 判定

| 官方 | 存在理由 | Triton 判定 |
|---|---|---|
| radix 多趟（b8~b64×NUM_PASS：bin 计数→前缀和→按位重排） | 大 N 主路径 | ❌ 无 Triton 对应，语义由迭代选择覆盖 |
| SmallBitonicSort 32 槽位网（k≤32） | 小 k 稳定+精确索引 | ⚠️ 网络不同、输出序同构（严格比较→左most，golden 证） |
| merge sort 多态（more/intra/simd） | 非尾轴/大 k 归并 | ❌ 机制不译，迭代选择覆盖 |
| MULT_CORE_OPTIM 二段（局选 k' 候选→单核归并） | 每核减少搬运 | ✅ Triton 两行段同构（stage1 tile 局选→stage2 归并） |
| SORT_AND_TOP_K（尾轴≥10M 全排截断） | 巨量 k 或 N | ⚠️ k≪N 不触发；k≈N 走 sort 卡 |
| NON_LAST_SMALL_AXIS（轴≤2048 免转置） | 非尾轴少搬一次转置 | ❌ 性能机制不译（wrapper movedim+contiguous 语义等价） |
| sort_with_index 联合排序基座 | (value,index) 搬运 | ✅ 候选 (value,idx) 对搬运同构 |
| AscendC::Sort 库精排（radix 末段） | 库不在仓 | ⚠️ 半壳，选择环末段无需精排（直接有序输出），golden 补偿 |
| 五模式 tilingkey 分派体系 | host 满核/分派 | ❌ 分派不译（shape 自适应覆盖） |

**主路线判定树**：packed-key `tl.topk` ❌（vsort 仅 f16/f32，int64 编译失败，int32 装不下）；`tl.sort` f32 值域+回查 ❌（tie 破歧困难，sort 卡另证）→ **迭代 max+argmax 选择 ✅**。

## 4. kernel 结构（v3 发射级迭代，无环单选 kernel）

```python
grid = (M,)                       # 一行一 program，k 次发射 + k 次 host 同步
@triton.jit
def _pick_kernel(x_ptr, v_ptr, i_ptr, N, j, sv, si,
                 LARGEST: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64)*N + offs, mask=msk, other=0)
    cur = sortable_key(x, DT)         # §1 全序键，int64 无符号保序
    if not LARGEST: cur = ~cur
    sel = tl.where(msk, cur, tl.full((BLOCK_N,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)          # int64 max reduce（probe7 证正确，标量化仅性能）
    m = tl.argmax(tl.where(sel == mx, BLOCK_N-offs, 0), axis=0)  # 左most；返回块内 lane
    v = tl.sum(tl.where(offs == m, x, 0))  # 取 m 处值（x[m] 标量索引不支持？→ all-zero-sum 单取）
    tl.store(v_ptr + pid*s_v + j, v); tl.store(i_ptr + pid*s_i + j, m)
# wrapper：xv=xt.reshape(M,N).contiguous()；for j: _pick_kernel[(M,)](xv,...); xv.scatter_(1, m_idx, type_min)
```

- **无 kernel 内动态 k 循环**（v2 `scf.for` × 归约被后端缺陷挂死 507034）——k 趟=发射级。
- 掩码：`torch.scatter_(1, picked_idx, type_min)` 沉底（largest 填 type-min/-inf，False 填 type-max/+inf）。**NPU scatter_ 不支持 u64**→走 int64 位视图 `view(torch.int64)` + `mv - 2**64 if mv>=2**63` 沉底值（u64 沉底位模式）。
- 单行段 N≤BLOCK_N：BLOCK_N=min(2048,next_pow2(N))（int64 容器 UB 预算）。
- **两行段 N>BLOCK_N**（官方 MULT_CORE_OPTIM 同构）：`_pick_tile_kernel[(M×T,)]` 每 tile 局选 k' 候选→workspace；`_pick_merge_kernel[(M,)]` 每行归并 k 趟。stage 1 取值用**局部 lane**（`tl.arange(0,BLOCK_N)==m`），全局索引 `tb*BLOCK_N+m` 单独存——**不可用 `offs==m` 全局比较**（tb>0 恒假→v=0 事故）。

## 5. 计算语义 → Triton（核心映射）

- **比较域全序键，值本体不改动**→无计算域提升（与 layer_norm/sum 的 fp32 域纪律相反，纯选择）。浮点 values 位级直拷。
- NaN=最大（largest 先选/False 后选），键映射天然给出，无特判分支。
- tie 左most 由 argmax 键 `BLOCK-offs`（严格比较稳定同构）。
- indices=轴内原始下标，与 sorted/dim 无关；indices_dtype 在 store 端 cast。
- **非尾轴 dim≠-1**：wrapper `movedim(dim,-1)` + **必须 `.contiguous()`**——`reshape` 保留 transpose 步幅 (1,32)，若用 `.clone()`（数据复制不改正布局）→ kernel 按行主序读错行（②门捕获，探测 14 实证）。

## 6. BLOCK / 预算

- 单行段 BLOCK_N=min(2048,next_pow2(N))；每 program 一行：cur int64 + x 原始 + sel ≈ 24B×BN，普通 UB 预算。
- 无 dot/scan/atomic——preflight 高危项天然不命中；`%` 仅两行段 task 解码（`//` + `a-(a//b)*b` 代数规避，C-001）。
- 性能两态：k 次发射 + k 次 host 同步 = ~ms 级 launch 开销（正确性优先降级）；BLOCK 选型/瓶颈定型未定型。

## 7. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| radix 多趟（b8~b64×NUM_PASS） | 迭代 max+argmax 选择 | ⚠️ 算法不同、语义同构（exact 索引+tie+NaN，②门验） |
| SmallBitonicSort 32 槽网 | 同左选择环 | ⚠️ 网络不同、输出序同构（左most tie） |
| MULT_CORE_OPTIM 二段（局选+归并） | stage1+stage2 两行段 | ✅ 结构同构 |
| SORT_AND_TOP_K（≥1e7 全排截断） | 不走（k≪N）；k≈N 走 sort 卡 | ⚠️ 登记 |
| NON_LAST_SMALL_AXIS 免转置 | wrapper movedim+contiguous | ❌ 性能机制不译（语义等价） |
| sort_with_index 联合排序 | 候选 (value,idx) 对搬运 | ✅ 同构 |
| AscendC::Sort 库精排 | 选择环无需精排 | ⚠️ 半壳，机制推断 |
| 五模式 tilingkey 分派 | 单 kernel 自适应 | ⚠️ 分派不译 |
| k tensor/indices_dtype/largest/sorted | wrapper 全复刻 | ✅ |

## 8. 精度策略与验收

- 值无精度变换：整型 exact 相等；浮点位级相等（同输入直拷）。
- 索引：k≤32&&sorted → **exact 比对**（含 golden 五注入模式 tie/NaN 期望）；k>32 → 语义比较（合法+gather+无重复）；uint 走 view+stable sort 参照。
- 注入模式：duplicate/all_equal/signed_zero/nan/infinity（golden 标配）；uint16/32/64 torch 原生不支持→view 有符号参照。
- 基准：CPU torch.topk；NaN 对照 golden `_value_better`。
- 覆盖：单行段全 dtype×largest×k 档×N 形态×注入×dim 三态×indices_dtype 已②门全绿（见 P4）。
- **边界受限**：两行段 N>BLOCK_N 大 N、u64 dtype——代码层已修、②门未重验（见 P4 §边界）。

## 9. P2.5 实测摘要（①+②门，2026-09-18/20，910_9372+CANN 9.1.0）

- **新后端事实回流**：① C-027（`hivm.hir.vsort` hw sort 指令仅 f16/f32，tl.sort/topk int64 键编译失败）→ packed-key 路线废弃；② C-028（tl.argmax int64 编译>7min CPU + 错值；int32 正常）→ 用 int32 flag-argmax 或 int64 max+int32 argmax 两步。已入 backend_contracts.md。
- **关键防坑（②门捕获，通用）**：
  1. kernel 内动态 k 循环（scf.for）挂死 → 发射级迭代；
  2. 两行段 tile 取值须局部 lane（offs==m 全局比对→v=0）；
  3. 非尾轴 movedim 后须 `.contiguous()`（reshape 保留 transpose 步幅→错读行）；
  4. NPU scatter_ 禁 u64→int64 位视图；
  5. 键映射负值分支 XOR 掩码须全 1（`~b`），0x7FFFFFFF 笔误与正值碰撞。
- ②门单行段全绿（含非 2 幂 N=700、k=N、dim=0/1/2、注入模式）；两行段/u64 边界受限。