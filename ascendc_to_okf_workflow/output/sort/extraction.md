# P1 语义抽取 — sort（Sort）

> 生成时间：2026-09-20｜依据 `manifest.md`，全量锚点本轮详读（tiling_arch35 全文、aclnn_sort 全文、aclnn_argsort 全文、golden.py 全文、docs×2、tiling_key/常量头）
> **抽取刻度：AIV 排序算子、regbase 主路径、P2.5 降级（①+②），语义面以 golden 为准补偿。**

## 1. 接口面与语义契约

`values, indices = aclnnSort(self, stable, dim, descending, valuesOut, indicesOut)`。def 表（sort_def.cpp）：Input x REQUIRED；Output y1=values(x dtype)、y2=indices(y2_dtype attr，i32×11 / i64×11)；Attr **axis=-1/descending=false/stable=false/y2_dtype=DT_INT32**；ascend950 AIC DynamicCompile。

**golden 语义契约（golden.py，铁律）**：
- `_require_stable(stable)`：**stable≠True 硬抛 ValueError**——ST/UT 只覆盖 `stable=True` 契约，骨架必须实现**稳定排序**（不能 unstable）。
- golden = `torch.sort(x, dim, descending, stable=True)` 逐位；`values` 有序输出、`indices`=轴内原始下标。
- `_index_dtype`：y2_dtype==9→int64 else int32（output_dtypes[1] 亦驱动）。
- 注入模式（`sort_input` 沿轴填充）：duplicate `[2,1,1,3,2,1,3,1]`、signed_zero `[-0,0,0,-0,1,-1,0,-0]`、nan `[3,nan,1,nan,2,2,-1,0]`、infinity `[inf,3,-inf,0,3,-0,1,-1]`。
- tolerance：float16/32/bfloat16→stat_rel_err；整数→binary_equal。
- 双 spec：`sort_golden(axis=-1)`（kernel 路径）与 `aclnn_sort_golden(dim=0)`（aclnn 路径）、`SortTorchSpec=SortAclnnSpec`（torch 交付路径同 torch.sort 语义）。

**aclnn 层约束（aclnn_sort.cpp / aclnnSort.md）**：
- `CheckParams`：shape x==values==indices、dim∈[-N,N-1]、dtype 列表。
- **regbase(ARCH3510) 12 dtype**：FLOAT16/FLOAT/BF16/UINT8/INT8/INT16/INT32/INT64/UINT16/UINT32/UINT64/BOOL；非 regbase 无 uint16/32/64 且含 BOOL。
- **indices 输出仅 INT64（DTYPE_SUPPORT_LIST_INT={INT64}）**；regbase 走 `reshapeCastRes` 保留 caller indicesType，非 regbase 强制 Cast 到 INT64。**def 的 y2_dtype attr 实际被 aclnn 的 indicesOut 提供的 dtype 覆盖**。
- **aclnnSort 内部**：bf16 非 regbase→Cast 浮点；BOOL→Cast UINT8；`dim 轴长==1` → **恒等特判**：values=self contiguous ViewCopy、indices=全 0（`GetTensorWithValueZero`）——对应 tiling IsAxisOneCopy(SCHID_7) + aclnn 级 shortcut；0-dim → values=self cast、indices=zeros（HandleDimZeroTensor）；空 tensor → workspace 0 no-op；ASCEND310 stable=True 不支持（非目标）。
- **deterministic**：aclnnSort/aclnnArgsort 均默认确定性实现（docs 明确）。
- 非浮点 size 上限（aicpu timeout）：INT64≤150,000,000、int8/16/32+uint8≤725,000,000。
- data limit：DATA_LIMIT=100000、AXIS_LIMIT=8。

## 2. aclnn 分派图（sort.cpp + argsort）

```
aclnnSort(self, stable, dim, descending, values, indices)
 └─ IsRegBase(910_9372)? → 恒 AiCore（SortForDavid：任意轴[2,2048]+i64 indices）✔主路径
 └─ 非 regbase → SortAiCore 仅尾轴 + AICORE_DTYPE_SUPPORT_LIST={f32,f16,bf16}（整型走 AICPU cast 通路）
     axis=1 且 bf16 非 regbase 不支持；SortAiCpu 兜底
 BuildSortGraph:
   UseNoTranspose(selfContiguous, dim)= regbase && dim≠last && [2,2048] && 收益判定
       → l0op::Sort(dim, 无转置)           (non-last small-axis two-stage)
   else → transpose 到尾轴 → l0op::Sort(-1) → transpose 回
aclnnArgsort(self, dim, descending, out):
   内部 l0op::Sort(-1, descending, stable=false, DT_INT32) → Cast DT_INT64 → out
```

**regbase 任意轴[2,2048] 走免转置 two-stage（no-transpose）**，与整轴 transpose+尾轴排序语义等价——**Triton 非尾轴统一 movedim+contiguous 覆盖，免转置=性能机制不译**。

## 3. 排序键语义（full-order key，radix_sort_constants.h + signed_zero_sort_utils.h）

与 topk 卡 §1 **完全同构**（值位模式 sign-flip 保序、NaN→max）：
- FP32：`u = bits ^ (b<0 ? 0x7fffffff : 0x80000000)`，**NaN→0xFFFFFFFF（最大键）**、-0.0→0x7fffffff / +0.0→0x80000000（相邻但不同键）；16-bit 用 0x7fff/0x8000/NaN 0xFFFF。
- 整数：符号/零扩展即有符号/无符号序；uint64 `bits ^ 0x8000...`（XOR_OP_VALUE_B64）。
- **descending**：不是键反转（tiling key 里 isDescend 是独立 tpl 参数，radix/merge 在值比较处反转序更优）；**Triton 骨架仍用 `~key` 位序反转（拓扑双射，topk 卡已证正确）**。
- **稳定排序硬约束 决定索引语义**：tie 必须保输入序（左most first）；Triton 用 argmax 键 `BLOCK-offs` 显式夺左most，或稳定 sort+回查。

## 4. Tiling 分派链（SelectSortSchedule，sort_tiling_arch35.cpp）

```
SortCheckParams → isInt32=IsRadixUint32CounterRange(lastAxis)；isDescend=attr[1]
SelectSortSchedule:
 ① TrySmallAxis (lastAxis ≤ SMALL_AXIS_THRESHOLD=512):
       lastAxis==1 → SCHID_7 axis-one-copy
       两段(two-stage) → isNonLastAxis&&innerSize!=1? SCHID_11 : SCHID_6
       insertion → SCHID_5
 ② TryNonLastSmallAxis (非尾轴 && axisLen∈[2,2048] && 非尾): no-transpose 两段,
       useMergeSort? SCHID_9(merge) : SCHID_10(radix);  innerChunk 批次化 inner 位置
 ③ 若 isNonLastAxis 且①②未收 → GRAPH_FAILED（aclnnSort 只放行 [2,2048] 免转置轴）
 ④ 尾轴: TryMerge (merge: lastAxis≤32? SCHID_8 : SCHID_0; 或 merge-more-core SCHID_3)
       || TryRadixOneCore SCHID_1 || TryMergeIntraCore SCHID_4 || 兜底 SCHID_2 radix-more-core
tilingKey = GET_TPL_TILING_KEY(schId, isInt32, isDescend)
```

**12 SCHID 语义**：0=merge sort(>32)、1=radix one core、2=radix more core、3=merge more core(big size)、4=merge intra core、5=small-axis insertion、6=small-axis two-stage(尾轴/物理连续)、7=axis-one-copy(axis==1)、8=merge sort(≤32 SORT32)、9=non-last merge、10=non-last radix、11=grouped non-last two-stage。

**关键常量**（sort_tiling_common.h）：SMALL_AXIS_THRESHOLD=512；MERGE_SORT_MAX_AXIS fp16/bf16=1024/fp32=4096；NON_LAST_SMALL_AXIS_THRESHOLD=2048；NON_LAST_SMALL_AXIS_MIN_AXIS_LEN=2；ONE_CORE_DATA_SIZE=2048；MULTI_CORE_MERGE_SORT_MAX_AXIS=32768；SORT32_SMALL_AXIS_THRESHOLD=32；TWO_STAGE_RANK_INVERSE_MAX_N=64；RADIX_UINT32_VALUE_MAX=0x3fffffff；MERGE_INTRA_CORE_MAX_BLOCKS=256。isNonLastAxis 物***理连续判据**=`innerSize==1`；innerSize=dim 之后各维乘积。

## 5. 算法族机制（overview，P2 判定对象）

| 机制 | 语义要点 | 稳定实现 |
|---|---|---|
| radix 多趟（one/more core） | 按位（byte=8bit NUM_PASS）计数+前缀和+重排；Simd 核；key 用 uint32/int64 counter 保序 | sortedSuffix/SORT32 槽位 |
| merge sort 多态 | 2/4 路归并、intra-core/more-core、值+索引对搬运、bank 旋转 padding | 归并 tie 取左 |
| small-axis insertion | 缓存行内插排（lastAxis≤512） | 插入稳定 |
| small-axis two-stage（rankInverse） | 首段排序+rankInverse 反查索引=位级精确 | TWO_STAGE_RANK_INVERSE_MAX_N |
| axis-one-copy | axis==1 恒等复制 | — |
| non-last small-axis two-stage | 非尾轴批次化 inner 位置免转置（[2,2048]） | 批次化归并 |

无半壳（全在仓）；唯一外部依赖 AscendC::Sort 库精排（radix 末段）——**由 golden 语义补偿（Triton 全排列直接输出，无需精排调用）**。

## 6. 后端约束预取证回流（C-027/C-028 联动）

- **C-027** `hivm.hir.vsort` hw 排序指令仅 f16/f32：`tl.sort`/`tl.topk` int64 键编译失败、int32 键装不下值+索引打包 → **packed-key 路线废弃（与 topk 卡同构）**。
- **C-028** `tl.argmax` int64 编译>7min CPU + 错值；int32 正常 → 用 int32 flag-argmax 或 int64 max + int32 argmax 两步。
- **sort ①门核心分歧点**（topk 卡留的伏笔）：验证 `tl.sort` **f32/f16 值域**真实正确性（若可直排则 sort 可免迭代选择），但稳定 tie 索引仍须 argmax 技巧或二次回查。

## 7. 判定树先决条件（与 topk 卡 §1 同构，可直接沿用）

| Triton 候选 | 判定 |
|---|---|
| ① packed-key `tl.sort/topk`（值+索引打包） | ❌ 废弃（C-027 vsort 仅 f16/f32；int64 编译失败；int32 装不下 f32值键+索引） |
| ② `tl.sort` f32 值域 + 索引回查 | ⚠️ 候选 B（①门实测正确性）；tie 稳定索引仍难（拖回步骤） |
| ③ 迭代 max+argmax 选择（topk v3） | ✅ 候选 A 主默认：稳定左most 由 `BLOCK-offs` 键天然给，NaN 键映射天然正确，无排序网络编译爆炸 |
| ④ radix/merge/insertion 多机制直译 | ❌ 不译（bin 计数+前缀和+按位重排=硬件向量化特化；归并多态=并发协作）；语义由 ③ 覆盖 |

sort 卡主路线=**与 topk 同构的迭代 max+argmax 全排列选择**（每行 N 次选择），叠加稳定 tie、descending 位序反转、非尾轴 movedim+contiguous、multi-row 批次化（grid 行并行）。

## 8. P1 质量门自查

- [x] 接口/def/aclnn/golden 语义全锚（stable=True 铁律、indicesType 驱动、deterministic、0-dim/axis1/空 tensor 特例）
- [x] 12 SCHID 精确路由 + 阈值 + 免转置边界（[2,2048]、innerSize==1 判据）
- [x] 排序键与 topk 同构（NaN→max、sign-flip）可复用 `_sortable_key`
- [x] 后端限制已预估（C-027/C-028）并定 ①门分歧点
- [x] 无半壳缺省项；AscendC::Sort 库调用=唯一外部依赖，golden 补偿
- [ ] 稳定排序 Triton 表达（迭代选择夺左most vs 直排+回查）①门实测定案——**遗留决策，见面 2**