---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Histc（等宽直方图，scatter 累加家族）Ascend C → Triton 语义转化卡
description: y[b]=#{i:bin(x_i)=b}，[min,max] 等宽 nbins 段、越界归边。官方三档 LoadMode（UB_FULL 核私有 UB 直方图 / UB_NOT_FULL 分段重扫 / GM_ATOMIC 逐元素 GM 原子），SIMT 512 线程 + asc_atomic_add。Triton 结构性 ❌：UB 可寻址 scratch、SetAtomicAdd+DataCopyPad 写回、SyncAll 清零屏障均不可译。P2.5 ①门实测定案三路线：tl.atomic_add 向量 scatter 重复地址丢更新（全同地址 1024→128=1/8，静默错值）禁用；主路径=program 独占 bin 的等值比较矩阵计数（无原子，BLOCK_X×BLOCK_BINS≤128×64 实测 UB 硬上限）；兜底=逐元素 1 宽向量 atomic（0-d 标量张量算术编译不过，必须 1 宽向量）。浮点/整型双 bin 式（fp32 invRange 乘 vs int64 乘除截断，int64 中间必须保留）；特殊值链按 golden 对齐（kernel 在 max=+inf 时误丢 −inf，源内矛盾登记）。
tags: [histc, histogram_fixed_width, histogram, scatter_atomic, atomic_add_trap, bin_ownership_matrix, special_values, inf_nan, int64_bin_calc, simt, geir_only, ascend950, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-math/math/histogram_fixed_width/（CANN ops-math 本地快照，GEIR-only 无 aclnn 层，仅 ascend950 AICore）
created_at: "2026-09-18T00:00:00Z"
updated_at: "2026-09-18T00:00:00Z"
---

sources:
  - url: src/ops-math/math/histogram_fixed_width/
    role: primary

# Histc（y[b] = #{i : bin(x_i) = b}，等宽直方图）

以 range=[min,max] 划 nbins 个等宽区间统计元素落区计数，**越界归边**（<min 计首 bin，>max 计末 bin，README）。对应 `torch.histc(input, bins, min, max)`——语义对齐，但 **torch 返回输入同 dtype（浮点），本算子 y 固定 int32**（def.cpp:38-42，InitValue(0)）。GEIR-only（无 aclnn 层，仅 examples/test_geir_*.cpp）；仅 ascend950 AICore（def.cpp:49——README 平台表称多平台，系仓外实现，本卡只管 950）。

## 1. 功能与公式（详设 §1，核心资产）

- **浮点域**（x=f32/f16→fp32 计算域）：`idx = int32((clamp(v,min,max) − min) · invRange · bins)`，截断取整后 clamp[0,bins−1]（DoBin，simt_full_load.h:81-96）。
- **整型域**（x=i32/i64→**int64 计算域**）：`idx = int32((clamp(v,min,max) − min) · bins // range)`——`(c−min)·bins` 可超 int32，**int64 中间必须保留**，否则溢出错 bin（histogram_fixed_width.cpp:35-41 ComputeTypeOf）。
- **特殊值**（浮点域；按 golden 对齐——⚠️ 源内矛盾见 §7）：
  - +inf / NaN → 丢弃；−inf 且 min 有限且 max 有限 → 丢弃；
  - min=−inf → 全部有效元素进**末 bin**（−inf 元素保留随之入末 bin）；
  - max=+inf → 全部有效元素进**首 bin**（golden 口径：−inf 元素保留入首 bin）。
- 两域截断行为不同（fp32 乘-截断 vs int64 除-截断），**勿跨域统一**；golden（spec.py:26-105）与 DoBin 逐式同构，验收以 golden 为准。

## 2. 原型与约束（详设 §2/§3）

| 项 | 值 |
|---|---|
| x | {F32, F16, I64, I32}，ND（def.cpp:22-26） |
| range | tensor [2]，**dtype 必须 == x dtype**（tiling:86-91）；**min<max 强制**（tiling:108-115） |
| nbins | I32 标量 tensor，ValueDepend（host tiling 读值，tiling:230 TilingInputsDataDependency） |
| y | I32 [nbins]，InitValue(0) |
| 平台 | ascend950 AICore（GEIR-only） |

wrapper 按 Python int/float 标量接 nbins/min/max（GEIR 图模式常量场景）；**kernel 内 bins/min/max/invRange 为运行时标量参数，不 constexpr 化**（C-017 教训）。

## 3. 分核策略（详设 §4 → Triton grid）

Ascend C 事实：x 均分核（formerLength=ceil(total/coreNum)，每核连续段，tiling:171-173）；y 清零均分+SyncAll；host 三档 LoadMode 决策（tiling:161-169）：
- `bins < (ubSize−32KB SIMT dcache)/4B` → **UB_FULL**：核私有 UB 直方图 + UB 原子加 + 一次 `SetAtomicAdd<int32>+DataCopyPad` GM 原子写回（full_load.h:162-210）；
- 否则 `totalLength > bins/100`（GM_ATOMIC_ADD_FACTOR=100——「GM 逐元素原子 ≈ 100× UB 路径代价」官方启发式）→ **UB_NOT_FULL**：UB 分段私有直方图，**每段重扫全量 x**（读放大换容量）；
- 否则 → **GM_ATOMIC**：逐元素 `asc_atomic_add` GM（零 UB buffer）。

Triton 映射：**三路线 constexpr `HIST_IMPL` 切换（①门实测定案）**：

| 路线 | 机制 |  verdict |
|---|---|---|
| 0 | `tl.atomic_add` 向量 scatter | ❌ **实测错值禁用**——向量内重复地址丢更新（§4） |
| 1 | **矩阵计数（主路径）**：grid=cdiv(bins,BLOCK_BINS)，program 独占 BLOCK_BINS 个 bin，扫全量 x tile 等值比较累加 `cnt += tl.sum((idx[:,None]==bin_ids[None,:])&valid[:,None], axis=0)`，连续 store | ✅ 无原子无冲突（**bin 所有权唯一**=官方核私有直方图的所有权等价）；bins 大时比较代价 numel×BLOCK_BINS 退化 |
| 2 | **逐元素 1 宽向量 atomic（兜底）**：persistent-core 连续段，元素循环 `tl.atomic_add(y_ptr+idx, 1, mask=valid)`（idx 为 [1] 向量） | ✅ = 官方档 2 GM_ATOMIC 真同构（跨核并发由 GM 硬件原子保证）；慢但正确，bins 大/numel 巨大时启用 |

选型判定树：bins ≤ ~2048 且 numel 不巨大 → 路线 1；bins 大或 numel 巨大 → 路线 2；**路线 0 任何场景禁用**。
y 清零 = wrapper `torch.zeros`（流内序，替代官方清零+SyncAll——Triton 不需要核间屏障）。

## 4. BLOCK / tile（①编译门实测硬约束）

- **路线 1：BLOCK_X×BLOCK_BINS ≤ 128×64**——128×128 实测 `ub overflow, requires 1606400 bits vs 1572864 available`，BiShengIR 自动关 code-motion/multi-buffer 重编译后仍失败（C-005 常规 UB 预算实例；比较矩阵 i1→i32 累加，128×64 安全）。落地 (BLOCK_X=128, BLOCK_BINS=min(64, next_pow2(bins)))。
- **路线 0 实测探针数据**（910_9372+CANN 9.1.0）：全同地址 1024 次更新落 **128=恰好 1/8**（疑 8-lane 冲突组内只一 lane 生效）；两值交替 4096→1280；地址全异 63/63 正确——**地址可重复语义（直方图/bincount 类）下向量 scatter atomic 不可用**（已回流契约 C-024）。
- **路线 2：0-d 标量张量算术编译不过**——`tl.load(x_ptr+off)` 得 0-d 张量，ConvertTritonIRToLinalgIR 报 `failed to legalize unresolved materialization from () to 'tensor<1xf32>'`；标量语义必须 1 宽向量（`off + tl.arange(0,1)`）承载（已回流契约 C-025）。

## 5. 计算语义 → Triton（核心映射）

```python
# 双域 bin 计算（IS_INT constexpr 分支，C-023：dtype 分支 wrapper 传入）
if IS_INT:   # i32/i64 → int64 中间域必须保留
    v = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.int64)
    idx = ((tl.minimum(tl.maximum(v, min_v), max_v) - min_v) * bins // range_v).to(tl.int32)
else:        # f16/f32 → fp32 域（COMPUTE_TYPE=float 同构）
    v = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    valid = mask & ~is_pinf & ~is_nan & ~(is_ninf & (min_ninf == 0) & (max_pinf == 0))
    idx = ((tl.minimum(tl.maximum(v, min_v), max_v) - min_v) * inv_range * bins).to(tl.int32)
idx = tl.minimum(tl.maximum(idx, 0), bins - 1)
idx = tl.where(min_ninf != 0, bins - 1, idx)   # 标量级分支，where 广播
idx = tl.where(max_pinf != 0, 0, idx)
```
- int64 bin 计算的 4a 标量化预警**非回归**——官方 SIMT 本就逐元素标量，卡片勿当缺陷警。
- 特殊值 valid 第三项的 `~max_pinf` 按 golden 对齐（§7 源内矛盾）。

## 6. 可达性总表（详设 §6）

| Ascend C | Triton | 可达性 |
|---|---|---|
| x 均分核连续段 | persistent-core 连续段（路线 1/2） | ✅ 同构（C-011 范式） |
| 档 2 GM_ATOMIC 逐元素原子加 | 路线 2：1 宽向量逐元素 atomic | ✅ 实测正确（36 case 全过） |
| 档 2 的向量 scatter 直译 | 路线 0 | ❌ 实测错值（C-024） |
| 档 0 UB_FULL 核私有 UB 直方图 | 无可寻址 UB scratch | ❌ 结构性；路线 1 = 所有权等价替代 |
| 档 1 UB_NOT_FULL 分段重扫 | 路线 1 的 bins 分 program | ⚠️ 思想同构（读放大换容量），机制不同 |
| SetAtomicAdd+DataCopyPad 写回 | 路线 2 atomic 替代 / 路线 1 连续 store | ❌ 不需要（机制替代） |
| 清零+SyncAll | wrapper torch.zeros + 流内序 | ❌→不需要（等价保证） |
| SIMT 512 线程 + asc_ldcg + UNROLL=6 | 无对应 | ❌ 线程模型，不翻译（无语义影响） |
| DoBin 双域式+特殊值链 | §5 constexpr+where 链 | ✅ 逐式同构（golden 比对闭环） |

## 7. 精度策略与验收

- 计数 int32 **精确相等**（无容差概念）；验收标准 = y 全等，golden 同式比对。
- 特殊值 case（NaN/±inf/越界/min=−inf/max=+inf/全 NaN）必须单列验收。
- ⚠️ **源内矛盾（登记）**：kernel DoBin:72-74 的 skip 链在 max==+inf 分支**之前**执行——场景（x 含 −inf、min 有限、max=+inf）kernel 丢 −inf（计 500），golden 保留进 bin 0（计 600，与 torch clamp 语义一致）。**验收以 golden 为准**，疑似官方 kernel 边界 bug，待 upstream 确认。
- numel > 2³¹ 时 y 溢出 int32——超规格，登记。

## 8. P2.5 实测摘要（①+②门，2026-09-18，910_9372+CANN 9.1.0）

- ①门抓获 3 条后端事实并全部回流契约：**C-024**（向量 scatter atomic 重复地址丢更新）、**C-025**（0-d 标量张量算术编译失败）、矩阵路线 UB 硬上限 128×64（C-005 实例）。
- ②门 **36/36 PASS**：4 dtype × 2 有效 impl × 常规 12 组 + 特殊值 4 组 + bins=1 + bins=500 多 program，对 numpy 移植 golden 精确相等。
- BLOCK 选型与瓶颈定型保持两态（降级纪律，confidence 不升 measured）。
