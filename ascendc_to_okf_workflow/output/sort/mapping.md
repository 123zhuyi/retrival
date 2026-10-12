# P2 语义转化 — sort（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-20｜依据 `extraction.md`｜**非低价值 → P2.5 降级执行①编译门+②正确性门**（③③.5④ 跳过，BLOCK 选型/瓶颈定型两态）。
> **与 topk 卡姊妹承接**：判定树先决条件、`_sortable_key` 全序键、stable tie 技巧、非尾轴 contiguous 防坑全部复用。

## 0. 检索记录（backend_facts）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.max`/`tl.argmax` lowering | `linalg.reduce` T2 | tl_op_lowering.md:39-40 |
| `tl.sort`/`tl.topk` → vsort | 仅 f16/f32，int64 键编译失败（C-027） | topk ①门回流 |
| `tl.argmax` int64 | 编译>7min+错值→int32 两步（C-028） | topk ①门回流 |
| `%` 禁用 / C-001 | 任务解码用 `//` + `a-(a//b)*b` | C-001 |
| stride/shape 传参 | N/axisLen 运行传参；LARGEST/DT/STABLE constexpr（C-023） | C-017/C-023 |
| persistent-core | 行主语小块默认 persistent-core | C-011 |
| i64 vreduce | 标量化（i64 max-reduce topk 已证正确） | dtype_support.md:58 |

## 1. 算法路线判定树（P2 核心决策）

| 候选 | 判定 | 理由 |
|---|---|---|
| ① packed-key `tl.sort/topk`（值+索引打包单趟） | ❌ **废弃**（C-027） | vsort 仅 f16/f32；int64 键编译失败；int32 键装不下 f32 值键+索引 |
| ② `tl.sort` f32/f16 直排 + 索引回查 | ⚠️ **候选 B** | ①门需实测 f32 tl.sort 值域正确性；**tie 稳定索引回查仍难**（重复值破歧）+ 后端无索引返回 guarantee |
| ③ **迭代 max+argmax 全排列选择（topk v3 同构）** | ✅ **主路线** | 稳定左most 由 `BLOCK-offs` 键天然给、NaN→max 键天然正确、无排序网络编译爆炸；**N 次发射×grid=M 并行取列**，发射次数=N 非 M×N |
| ④ radix 多趟 / merge 多态 / insertion / two-stage 直译 | ❌ 不逐式译 | bin 计数+前缀和+按位重排、归并并发协作、rankInverse 位级精确为硬件向量化特化；语义由 ③ 覆盖，golden 补偿稳定 sort |

**⛔ 与 topk 同构的落地铁律**：迭代在 **wrapper 发射级**（N 次 kernel 发射 + N 次 host 同步 + torch scatter 掩码），**不能放 kernel 内动态循环**（topk v2 `scf.for` 动态界证实挂死，vector core timeout 507034）。sort 无法比 topk 更省——稳定全序必须逐位选出。

**主路线结构（关键优化）**：sort 输出有 N 列，但每发射一次 kernel 同时处理 **所有 M 行取一列**（grid=M，一 program 一行），选中后 scatter 改掩码值。故 **N 次发射、每次 M 行并行**——发射开销≈O(N)，对 small-axis 主导区（N≤512/SORT32 区）与 radix-one/more 中轴现实可测；大 N 二段合并推迟（边界受限，登记）。

## 2. 主 kernel 结构（复用 topk `_pick_kernel` 选择环）

```python
grid = (M,)                       # 一行一 program，N 次发射取全列
@triton.jit
def _sort_col_kernel(x_ptr, v_ptr, i_ptr, N, j, sv, si,
                     DESCENDING: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N); msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64)*N + offs, mask=msk, other=0)
    cur = sortable_key(x, DT)     # §3 全序键 int64 无符号保序；已加工 x_work 掩码值
    if DESCENDING: cur = ~cur     # 位序反转（与 topk largest False 同构，NaN 两向均正确）
    sel = tl.where(msk, cur, tl.full((BLOCK_N,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)                      # int64 max reduce（标量化）
    m = tl.argmax(tl.where(sel == mx, BLOCK_N-offs, 0), axis=0)  # 左most 稳定tie；块内 lane
    v = tl.sum(tl.where(offs == m, x, 0))         # 取 m 处原值（非掩码值? 见 §5 取值域）
    tl.store(v_ptr + pid*s_v + j, v); tl.store(i_ptr + pid*s_i + j, m)
# wrapper：for j in range(N):  _sort_col_kernel[(M,)](x_work, v, idx, N, j,...)
#          x_work.scatter_(1, idx[:,j], type_min_bits)   # 掩码沉底，下一发射取次大
```

- **输出列 j 逐位生成**：第 j 次发射从「当前仍未选中的 x_work」取全局最大（+左most tie）填 `values[r,j]/indices[r,j]`，scatter 沉底后第 j+1 次取次大。N 次后全行有序。
- **升序**：`DESCENDING=False`，掩码沉底用 **type-min**（最大先出），最后 v 列自然升序 → 这正是 torch.sort 升序。**降序**：`DESCENDING=True` 用 `~key`，但掩码沉底值要随序反转（升序沉 type-min 会与降序 ~key 冲突）——**沉底值在 x_work 域（值位模式），不管 DESCENDING**，统一沉「值域最小位模式」：浮点 -inf 位 / 整数 type-min。见 §3 值域设计。
- **两行段/大 N**：N>BLOCK_N 段选+归并（官方 MULT_CORE_OPTIM / merge 同构），**骨架登记为边界受限**，正确性门先锁单行段（N≤BLOCK_N，覆盖 small-axis 主导区）。

## 3. 全序键 + 掩码值（复用 topk `_sortable_key`）

| dtype 族 | 键构造（int64 无符号保序，NaN=最大） | 掩码沉底位模式 |
|---|---|---|
| f32/f16/bf16 | 位模式 sign-flip（16bit 0x7fff/0x8000）；NaN→全1 | **-inf（0xff800000…）** |
| i8/u8/i16/u16/i32/u32 | 符号/零扩展 int64 | type-min 位 |
| i64 | 恒等 int64 | INT64_MIN |
| u64 | `bits ^ INT64_MIN` | -inf 位视图（int64 view） |

- **浮点 NaN→最大键**（与官方 radix CANONICAL_NAN_KEY 一致：0xFFFFFFFF）。
- **沉底走 x_work 值域**：浮点 -inf、整型 type-min（u64 走 int64 位视图再 -inf/2^63 轮廓），因下一发射 `sortable_key(x_work)` 把已选元素 key 压到最低、永不重选。**与 DESCENDING 解耦**（DESCENDING 只反转比较键，不反掩码值）。
- **tie 左most**：argmax 键 `BLOCK-offs`，升/降序均保输入序（官方 golden=stable=True **铁律**）。

## 4. wrapper 归一化

```python
sort(x, axis=-1, descending=False, stable=True, y2_dtype=torch.int32)
```
- **stable：跳过/忽略**（骨架恒稳定输出=超集合法）；ST 只覆盖 stable=True，stable=False 不期望。
- axis：负轴归一 + movedim 到尾轴 + **必须 `.contiguous()`**（topk 卡二阶陷阱实证：reshape 保留 transpose 步幅→错读行）；折叠外层 M。
- DESCENDING/DT constexpr 传参；N=axisLen 运行传参。
- y2_dtype：i32/i64 输出分配，store 端 cast。indices=轴内原始下标（块内 lane 恰=尾轴下标）。
- 0-dim 特例：values=self、indices=全 0（wrapper 早退，匹配 aclnn HandleDimZeroTensor）。
- axis 长==1：values=self、indices=全 0（匹配 aclnn axis1 恒等 shortcut）。
- M=0 / 空：早退。

## 5. 取值域与 dtype 纪律

- **`lsortable_key` 作用于 x_work（掩码后的工作副本）**：`v=tl.sum(where(offs==m, x_work, 0))` 取的是当前工作域值 `=x[r, m]`（未掩码时即原值）。已掩码位置 key 最低、永不被 argmax 选中。
- 值输出无计算域提升：v 原值直存（值与 topk 卡一致，无需 fp32 中间态）。
- **scatter_ 沉底限制**：NPU scatter_ 不支持 **u64**（161002）→ 走 `x_work.view(torch.int64)` + 位模式沉底（topk 卡已封 `_mask_out` helper，直接复用）。

## 6. BLOCK / 预算

- BLOCK_N = min(2048, next_pow2(N))（int64 容器 UB 预算，C-005；①门实测上限）。
- 每 program 一行：cur/sel int64 2×8B + x 4B + 键中间 ≈ 24B×BN ≈ 48KB，普通预算。
- **发射次数=N**（每行 M 并行）；对 N≤2048 单行段现实；N>2048 大尾轴需二段归并（登记边界受限）。
- 无 dot/scan/atomic——preflight 高危项天然不命中；`%` 不涉及（无跨 tile 解码）。

## 7. 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| radix 多趟（bin+前缀和+按位重排） | 迭代 max+argmax 全排列选择 | ⚠️ 算法不同、输出语义同构（顺序+稳定tie+NaN，②门验） |
| merge sort 多态（2/4路/ intra/more-core） | 单行段选择；大 N 二段归并登记 | ⚠️ 结构同构 deferred |
| small-axis insertion/two-stage(rankInverse) | 选择环 + output-lane 索引 | ⚠️ 机制不同、位级精确索引语义同构 |
| axis-one-copy（axis==1） | wrapper 恒等特判 | ✅ wrapper 复刻 |
| non-last small-axis two-stage 免转置 | wrapper movedim+contiguous | ❌ 性能机制不译（语义等价） |
| AscendC::Sort 库精排 | 全排列直接输出，无需精排调用 | ⚠️ 半壳，golden 补偿 |
| 12 SCHID tilingkey 分派 | 单 kernel 自适应 | ⚠️ 分派不译（shape 自适应覆盖） |
| y2 indices{i32,i64} / 0-dim / axis1 / stable / descending / 任意轴 | wrapper 全复刻 | ✅ |

## 8. 精度策略与验收（②门）

- values：无精度变换，浮点位级相等（同输入直拷）、整型 exact。
- indices：**exact 比对**（含 golden 四注入模式 tie/NaN 期望：duplicate 保左most、signed_zero 相邻、nan→末位、infinity 边界）。
- 基准：CPU `torch.sort(stable=True)`（golden `_sort`）。
- 覆盖：f32/f16/bf16 + 整型族(语义补偿,因 AiCore 仅浮点) × {升,降序} × N{small 8/32/非2幂 31/中 700} × 注入4模式 × axis{尾,中,首 movedim} × y2_dtype{i32,i64} × 0-dim/axis1 特例。
- 检测：行序相等 + 稳定 tie 断言（重复值输出序=输入序）+ 索引 gather 反查验证（values gather indices == 原值按序）。

## 9. P2 质量门自查

- [x] tl.max/argmax T2 [SRC]；vsort/argmax 限制沿用 C-027/C-028
- [x] ❌ 项（radix/merge/insertion/免转置/分派）均有降级论证；主路线与 topk ②门已验证选择环 exact
- [x] 无 %；N 运行传参；DESCENDING/DT constexpr；无 atomic；无 kernel 内动态循环（发射级）
- [x] 稳定契约=硬约束 → 左most tie 由 `BLOCK-offs` 参数化给
- [x] **P2.5 降级①+②**：①编译门=迭代选择全序键×DESCENDING×N 真编译 + **f32 tl.sort 附带验证（B 候选分歧点：若 vsort f32 直排可用则登记为 sort 优化岔路）**；②正确性门=`__main__` smoke（对 CPU torch.sort(stable=True)），单行段全绿即交付，大 N/两行段登记边界受限