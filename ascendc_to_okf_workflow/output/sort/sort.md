---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Sort（沿轴全排列稳定排序，排序家族主卡）Ascend C → Triton 语义转化卡
description: values,indices=sort(x,axis,descending,stable,y2_dtype)——沿 axis 全排列排序，values dtpe=x、indices dtpe={i32,i64}。官方 12 条 SORT_SCHID 分派（axis-one-copy / small-axis insertion/two-stage / radix one/more-core / merge sort 四态 / non-last small-axis 两态）+ host tiling 三文件 + 姊妹 argsort aclnn，全在仓无半壳（唯一外部依赖 AscendC::Sort 库精排=golden 语义补偿）。Triton 主路线=topk v3 同构的发射级迭代 max+argmax 选择：**N 次 kernel 发射（每次 grid=M 并行取所有行的第 j 列）**+ torch.scatter 掩码沉底；赠出序由方向切换——升序反键~key+沉 type-max 取最小先、降序正键+沉 type-min。稳定契约=硬约束（golden 只覆盖 stable=True）→ tie 左most 由 argmax 键 BLOCK-offs 天然给。NaN=最大键（CANN CANONICAL_NAN_KEY 同构）。与 topk 卡共享 `_sortable_key` 全序键/`_mask_out` 防坑封装（u64 走 int64 位视图、非尾轴 movedim 后须 contiguous、发射级非 kernel 内循环）。
tags: [sort, argsort, stable_sort, full_sort, iterative_max_argmax, radix_sort, merge_sort, insertion_sort, small_axis_two_stage, non_last_small_axis, axis_one_copy, descending, nan_max, tie_leftmost, launch_iteration, vsort_limit, scatter_limit, regbase, ascend950, ascend910b]
confidence: inferred
status: draft
resource: src/ops-math/math/sort/（CANN ops-math 本地快照，arch35 四族 kernel + tiling 三文件 + aclnn 两算子 + golden.py）
created_at: "2026-09-20T00:00:00Z"
updated_at: "2026-09-21T00:00:00Z"
---

sources:
  - url: src/ops-math/math/sort/op_kernel/sort_apt.cpp
    role: primary
  - url: src/ops-math/math/sort/op_kernel/arch35/sort_tiling_key.h
    role: primary
  - url: src/ops-math/math/sort/op_host/arch35/sort_tiling_arch35.cpp
    role: primary
  - url: src/ops-math/math/sort/op_host/arch35/sort_tiling_common.h
    role: primary
  - url: src/ops-math/math/sort/op_api/aclnn_sort.cpp
    role: primary
  - url: src/ops-math/math/sort/tests/assets/golden.py
    role: primary
  - url: src/ops-math/math/sort/op_api/aclnn_argsort.cpp
    role: secondary

# Sort（沿轴全排列稳定排序）

沿 axis（负轴归一到尾轴）对 x 每行全排列排序。def 表：x 11 dtype + BOOL（次），1-8 维 ND；Attr **axis=-1/descending=false/stable=false/y2_dtype=DT_INT32**；Output **y1=values(x dtype)、y2=indices(y2_dtype)**。对应 `torch.sort`。**regbase(910_9372) 恒 AiCore**（SortAiCoreForDavid：任意轴[2,2048]免转置两段+i64 indices），非 regbase 退化仅尾轴浮点（还整形走 AICPU cast 通路）；**aclnnAiCore 实际 dtype 仅浮点 {f32,f16,bf16}**。**稳定排序铁律：golden ST 只覆盖 stable=True（`_require_stable('stable=True 契约')` 硬抛错）**——骨架必须稳定（左most tie）。aclnnSort/aclnnArgsort 均默认**确定性**实现。姊妹 aclnnArgsort（无 stable 参数，内部 stable=false，指标 i64 输出）。

## 1. 功能与全序键（golden `torch.sort(stable=True)` 同构）

- `values[r,j]`=排序后值（dtype 同 x，沿 axis 每行有序）；`indices[r,j]`=**轴内原始下标**。descending 双向；axis 任意（非尾轴 [2,2048] 免转置，Triton 统一 movedim 归尾）。
- **全序键**（比较域，值本体不改动，选择骨架无精度变换）：
  - 浮点：位模式 sign-flip——`u = bits ^ (sign ? 0x7FFFFFFF : 0x80000000)`（16bit 0x7FFF/0x8000、64bit 各自宽）；**NaN → 最大键**（CANN `CANONICAL_NAN_KEY_B32=0xFFFFFFFF` 完全同构）；**-0.0/+0.0 折叠为单一 0 键**（CPU 探针实证 torch.sort 视 ±0 相等→稳定保输入序，若给相邻不同键会破坏相等，②门 signed_zero/infinity 曾 FAIL；负值键<0键<正值键保序）。
  - uint64：`u = bits ^ INT64_MIN`；int8~int64 零/符号扩展即有符号序；uint8~32 零扩展+^SIGN 转有符号序。
  - 统一 int64 无符号容器，`~key` 即序反转（**仅用于升序反选的最小先出**，见 §4）。
- **稳定 tie**：`torch.sort(stable=True)` 严格稳定 → tie 保输入序（左most 先出）。实现：argmax 键 `BLOCK-offs` 夺左most。
- **NaN/序向（②门已验）**：NaN=最大键。**升序**反键后 NaN=最小 selection → **排末**；**降序**正键 NaN=最大 → **排首位**——与 torch.sort 逐位一致（nan 模式升/降序②门全 PASS，125999366/降序 corner 已闭环）。

## 2. 原型与约束

| 项 | 值 |
|---|---|
| x | 11 dtype + BOOL（regbase），ND 1-8 |
| y1=values | dtype=x，shape=x |
| y2=indices | dtype=y2_dtype（i32/i64，默认 i32；aclnn 由调用方 indicesOut 提供），shape=x |
| attrs | axis=-1、descending=false、stable=false、y2_dtype=3 |
| 特殊 | axis 长==1 → values=self、indices=全 0（aclnn 恒等 shortcut 同构）；0-dim → values=self cast、indices=全 0；空 tensor → no-op |

**aclnn dtype 面（登记）**：regbase 12 dtype（FLOAT16/FLOAT/BF16/UI8/II8/UI16/II16/UI32/II32/UI64/II64/BOOL）；非 regbase 无 uint16/32/64。AiCore 实际仅浮点（整型 AICPU cast 通路）——Triton 骨架全 dtype 自然覆盖。非浮点 size 上限（aicpu timeout）：INT64≤15e7、int8-32+uint8≤7.25e8。bool→cast uint8、bf16 非 regbase→cast float。

## 3. 官方算法族 → Triton 判定

| 官方 | 存在理由 | Triton 判定 |
|---|---|---|
| radix 多趟（one/more core，byte=8bit NUM_PASS：bin 计数+前缀和+重排） | 大/中尾轴主路径 | ❌ 无 Triton 对应，语义由迭代选择覆盖 |
| small-axis insertion（lastAxis≤512） | 缓存行内插排 | ⚠️ 机制不译，选择环同构 |
| small-axis two-stage（rankInverse 位级精确索引） | 小轴稳定+精确索引 | ⚠️ 位级索引语义由 output-lane 精确下标同构 |
| merge sort 多态（merge/big-size/intra-core/more-core） | 大轴归并 / 非尾轴 | ❌ 机制不逐式译；大 N 二段归并登记 deferred |
| axis-one-copy（axis==1） | 轴长 1 恒等 | ✅ wrapper 恒等特判（values=self、indices=全0） |
| non-last small-axis two-stage（[2,2048] 免转置批次化 inner） | 非尾轴少映一次转置 | ❌ 性能机制不译（wrapper movedim+contiguous 语义等价） |
| AscendC::Sort 库精排（radix 末段） | 库不在仓 | ⚠️ 半壳，全排列直接输出无需精排，golden 补偿 |
| 12 SCHID tilingkey 分派 | host 满核/分派 | ❌ 分派不译（shape 自适应覆盖） |

**主路线判定树**（与 topk 卡 §1 同构）：packed-key `tl.sort` ❌（C-027 vsort 仅 f16/f32，int64 编译失败）；`tl.sort` f32 直排+索引回查 ⚠️ 候选 B（tie 稳定索引回查仍难，①门附带验证）；→ **迭代 max+argmax 选择 ✅**。

## 4. kernel 结构（v-1 发射级迭代，topk v3 同构）

```python
grid = (M,)                       # 一行一 program
@triton.jit
def _sort_col_kernel(x_ptr, vis_ptr, v_ptr, i_ptr, N, j, sv, si,
                     DESCENDING: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64)*N + offs, mask=msk, other=0)
    vis = tl.load(vis_ptr + pid.to(tl.int64)*N + offs, mask=msk, other=1)  # 0=未选
    cur = sortable_key(x, DT)         # 全序键 int64（±0 折叠 + NaN=最大，见 §1）
    if not DESCENDING: cur = ~cur     # 升序：反键 → 取最小先出
    cur = tl.where((vis == 0) & msk, cur, -(2**63))   # 已选/越界 → 键域绝对最小，永不重选
    mx = tl.max(cur, axis=0)          # int64 max reduce（标量化）
    m = tl.argmax(tl.where(cur == mx, BLOCK_N-offs, 0), axis=0)  # 左most 稳定tie；块内 lane=轴内下标
    tl.store(v_ptr + pid*s_v + j, tl.sum(tl.where(offs==m, x, 0)))  # 取 m 处原值（x 不改，含 NaN）
    tl.store(i_ptr + pid*s_i + j, m)
rv, vis: # wrapper：vis=zeros(M,N)；for j: _sort_col_kernel[(M,)](x_work, vis, ...); vis.scatter_(1, idx[:,j], 1)
```

- **N 次发射**（每次 grid=M 并行取所有行第 j 列），**vis 缓冲标记已选位** → 下次取次优，N 次全行有序（**x 值永不改写**）。
- **序向切换（sort 独有）**：升序 `~key` 取最小先出；降序正键取最大先出。**已选位在键域排除**（置 INT64_MIN，恒低于实键含 ±inf/NaN）→ 与值沉底解耦、绝无碰撞。
- **无 kernel 内动态循环**（topk v2 `scf.for` 挂死类规避；发射级）。
- 单 segment N≤BLOCK_N=min(2048,next_pow2(N))（覆盖 small-axis 主导区）；**大 N>2048 二段归并=边界受限，kernel 当前未实现**（mapping 登记，安全断言）。

## 5. 计算语义 → Triton（核心映射）

- 值无计算域提升：v 原值直存（dtype=x）。
- NaN=最大键（CANN 同构）；升序反键后 NaN 先出（①门/②门实测 torch 尾排行为，corner 登记）。
- tie 左most 由 argmax 键 `BLOCK-offs`（稳定排序=硬约束）。
- indices=轴内原始下标（块内 lane=尾轴下标），y2_dtype 在 store 端 cast。
- **非尾轴 dim≠-1**：wrapper `movedim(axis,-1)` + **必须 `.contiguous()`**（topk 探针14 实证：reshape 保留 transpose 步幅(1,32)→错读行；`.clone()` 也不改正布局）。
- 0-dim / axis 长==1 / 空：wrapper 早退特判（indices 全 0）。

## 6. BLOCK / 预算

- BLOCK_N=min(2048,next_pow2(N))；每 program 一行 cur/sel int64 2×8B + x + 键中间 ≈ 24B×BN。
- 发射次数=N（×行并行 M）；对 N≤2048 现实，大 N 需二段归并（边界受限）。
- 无 dot/scan/atomic——preflight 高危项天然不命中；`%` 不涉及；stride 运行传参、DESCENDING/DT constexpr。
- 性能两态：N 次发射+N 次 host 同步=launch 开销随 N 线性（正确性优先降级）；BLOCK/瓶颈未定型。

## 7. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| radix 多趟（bin+前缀和重排） | 迭代 max+argmax 全排列选择 | ⚠️ 算法不同、输出语义同构（顺序+稳定tie+NaN） |
| merge sort 多态 | 单 segment 选择；大 N 二段归并登记 | ⚠️ 结构同构 deferred |
| small-axis insertion / two-stage(rankInverse) | 选择环 + output-lane 精确下标 | ⚠️ 机制不同、位级索引语义同构 |
| axis-one-copy（axis==1） | wrapper 恒等特判 | ✅ wrapper 复刻 |
| non-last small-axis 免转置 | wrapper movedim+contiguous | ❌ 性能机制不译（语义等价） |
| AscendC::Sort 库精排 | 全排列直接输出无需精排 | ⚠️ 半壳，golden 补偿 |
| 12 SCHID 分派 | 单 kernel 自适应 | ⚠️ 分派不译 |
| y2{i32,i64}/0-dim/axis1/stable/descending/任意轴/argsort 接口 | wrapper 全复刻 | ✅ |

## 8. 精度策略与验收

- values：无精度变换，浮点位级相等（同输入直拷）、整型 exact。
- indices：**对 CPU `torch.sort(stable=True)` 逐位比对**（golden `_sort`）：values isnan-aware 全等 + indices（稳定序）全等。
- 注入模式（golden `sort_input`）：duplicate 保左most、signed_zero 相邻、nan→尾排（依①/②门实测）、infinity 边界。
- 覆盖（②门）：f32/f16/bf16 + 整型族（语义面）× {升,降序} × N{8,32,64,700非2幂} × 注入4模式 × axis{尾,中,首} × y2_dtype{i32,i64} × 0-dim/axis1 特例（单 segment N≤2048）。
- **边界受限**：大 N>2048 二段归并（未实现，断言登记；官方 merge/radix more-core 域，同 topk 卡两行段边界面）。单 segment 内 NaN 升/降序 tail 位与 ±0 稳定序均已 ② 门闭环（无此受限）。

## 9. P2.5 实测摘要（①+②门，2026-09-20/21，910_9372+CANN 9.1.0）

- **沿用 topk 卡 C-027/C-028**（vsort 仅 f16/f32→packed-key 废弃；argmax int64→int32 两步）。
- **关键防坑（topk 卡 §9 继承 + sort 特有，②门实抓两类键映射 bug 并修复）**：
  1. 迭代在发射级（N 次发射×grid=M 并行取列），**不得 kernel 内动态循环**（507034 挂死类）；
  2. 排序序向切换：**升序=反键取最小先出、降序=正键取最大先出**（与 topk LARGEST 反向）；
  3. 非尾轴 movedim 必须 `.contiguous()`（transpose 步幅 trap）；
  4. u64 scatter 禁 → int64 位视图（topk `_mask_out`）；
  5. 稳定 tie 左most 由 argmax 键 `BLOCK-offs` 强制（ST 只覆盖 stable=True 契约）；
  6. **已选位必须键域排除（vis 缓冲置 INT64_MIN），不能值域沉底 ±inf**——±inf/NaN 极值键会与实数据碰撞被左most 误选（nan升+infinity 双向曾 FAIL，探针 06 定位）；
  7. **±0 必须折叠为单一 0 键**——sign-flip 给 ±0 相邻不同键会破坏 torch「±0 相等→稳定保序」（signed_zero/infinity 曾 FAIL，CPU 探针 test4/6 实证）。
- **②门单 segment（N≤2048）20 PASS / 0 FAIL 全绿**：全 dtype{f32,f16,bf16,i8,i16,i32,i64,u64}×{升,降序}×N{8,32,64,700非2幂}×四注入模式(duplicate/signed_zero/nan/infinity 每项双向)×axis{尾,中,首}×y2{i32,i64} 逐位对 CPU torch.sort(stable=True) 全等。**本次无环境/机器受限——修复纯走 CPU 探针+②门，机器空闲一次性跑通**。大 N 二段归并=唯一边界受限（登记，未实现）。