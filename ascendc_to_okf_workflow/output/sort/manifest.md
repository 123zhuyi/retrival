# P0 材料清单 — sort（Sort）

> 生成时间：2026-09-20｜来源目录：`src/ops-math/math/sort/`（aclnn 层在 `op_api/sort.cpp` + 姊妹 argsort `op_api/aclnn_argsort.*`）
> **kernel 类型判定：AIV-only 排序算子（arch35 全量 12 schId 路线在仓，无半壳）**——radix more/one core、merge sort/big size/intra core、small axis insertion/two-stage、non-last small axis（两/单 stage）、axis one copy 全在仓；host tiling 三文件在仓（common 340+ 行 + arch35 + tiling key）。语义面比 topk 更丰富：**stable 稳定键 + descending 双向 + 任意轴（非尾轴 [2,2048]）+ y2 indices{i32,i64} + 0-dim 特例 + argsort 姊妹 aclnn**。
>
> **🔺 低价值预判：非低价值——P2.5 降级执行①编译门+②正确性门，P4 不豁免。**
> - **排序家族主卡**（sort/topk/kthvalue 同族，与 topk 卡姊妹承接）；
> - **❌ 候选 ≥5**：① 12 条 schId 分派体系（axis-one-copy/small-axis insertion/two-stage/radix 一/多核/merge 四态/非尾轴两态）策略面不可直译；② radix 多趟基数排序（bin 计数+前缀和+按位重排按 dtype 位宽 NUM_PASS）无 Triton 直接对应；③ merge sort 多态（merge/big-size/intra-core/multi-core 分层归并段）硬件特化不逐式译；④ 稳定排序的「sortedSuffix/rankInverse/SORT32 槽位」位级精确索引语义网络不可逐式译；⑤ y2=indices 双 dtype 输出 + 0-dim 全 0 索引 + argsort 接口面属性语义面需 wrapper 全复刻；
> - **后端敏感点（P2 前预取证，①门将正式回流）**：topk 卡已证 **C-027 `hivm.hir.vsort` hw 排序指令仅 f16/f32 元素类型，`tl.sort`/`tl.topk` int64 键编译失败**、**C-028 `tl.argmax` int64 编译爆炸+错值 → int32 两步**——sort 卡 ①门核心任务 = **验证 `tl.sort` f32 值域真实正确性 + 探索稳定左most 索引的 Triton 表达**（tl.argmax 迭代选择 vs `tl.sort` 直排二路线投票，见面 2）；
> - golden.py 金矿：torch.sort(stable=True) 稳定语义、**ST 只覆盖 stable=True 契约**（硬抛错）、y2_dtype={int32,int64}、四注入模式（duplicate/signed_zero/nan/infinity）、source 兼有 sort 与 aclnn_sort 两条 spec 路径；
> - 联动结论：**走 P2.5 降级路径（①+②）**，BLOCK 选型/瓶颈定型两态，confidence 不升 measured(skeleton)。与 topk 卡共享 radix/select 架构与 golden 键映射——**判定树先决条件与 topk 卡 §1 同构（packed-key 废弃已由 C-027 证），可直接沿用。**

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/sort_def.cpp`(69) | ✅ | Input x REQUIRED；Output **y1=values(x dtype)、y2=indices({i32×11 | i64×11}，y2_dtype attr 选)**；Attr axis=-1/descending=false/stable=false(**注释 ST 覆盖 stable=True**)/y2_dtype=DT_INT32；ascend950 AIC DynamicCompile/Shape/Rank | §2、§3 |
| infershape | `op_host/sort_infershape.cpp`(37) | ✅ | y1=y2=x 同 shape（:30-31） | §2 |
| kernel 入口 | `op_kernel/sort_apt.cpp`(143) | ✅ | **12 schId KERNEL_TASK_TYPE 分派**：SORT_SCHID_0/8=merge sort、1=radix one core、2=radix more core（MIX_AIV_1_0）、3=merge big size(仅 f32)、4=merge intra core(仅 f32)、5=small axis insertion、6=small axis two stage、7=axis one copy、9=non-last small axis(merge)、10=non-last small axis(radix)、11=grouped small axis two stage；radix more core 六 key(2/65538/258/65794 等)；maxDealingNum/radix mmerge 平台分派 | §4.1、§5 |
| tiling key | `op_kernel/arch35/sort_tiling_key.h` | ✅ | SORT_SCHID_0~11 全定义；SORT_TPL_MERGE_SORT/ RADIX_MORE/ONE_CORE/ MERGE_BIG_SIZE/ INTRA_CORE/ SMALL_AXIS_INSERTION/ SMALL_AXIS_TWO_STAGE/ NON_LAST_* / AXIS_ONE_COPY 各 KEY_SEL | §4.1 |
| tiling common | `op_host/arch35/sort_tiling_common.h`(340) | ✅ | 常量：SMALL_AXIS_THRESHOLD=512、MERGE_SORT_MAX_AXIS fp16/bf16=**1024**/fp32=**4096**、NON_LAST_SMALL_AXIS_THRESHOLD=2048、ONE_CORE_DATA_SIZE=2048、MULTI_CORE_MERGE_SORT_MAX_AXIS=32768、MERGE_INTRA_CORE_MAX_BLOCKS=256、SORT32_SMALL_AXIS_THRESHOLD=32、TWO_STAGE_RANK_INVERSE_MAX_N=64、RADIX_UINT32_VALUE_MAX、SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT=4095 | §4.2 |
| tiling arch35 | `op_host/arch35/sort_tiling_arch35.cpp/.h` | ✅ | TrySmallAxis（lastAxis≤512→axis-one-copy/insertion/two-stage/non-last）、TryRadix/*MoreCore、TryMergeSort/TryMergeSortBigSize/TryMergeSortIntraCore、TryNonLastSmallAxis 等 Try 链 + IsAxisOneCopy(lastAxis==1) + bank 旋转 padding 对齐 + rankInverse 二段 | §4.2 |
| kernel 架构 | `op_kernel/arch35/*.h`（多族） | ✅ | radix_more/core(b88 位宽 radix)、merge_sort/big_size/intra_core/more_core_base、small_axis_insertion/two_stage_base、non_last_small_axis(_two_stage)、axis_one_copy；common/ 下 signed_zero_sort_utils/radix_sort_constants/merge_sort_constants/util_type_simd | §5 |
| aclnn 层 | `op_api/sort.cpp`(234) | ✅ | **IsRegBase→恒 AiCore(SortForDavid，任意轴[2,2048]+i64)**；非 regbase→SortAiCore 仅尾轴(AICORE_DTYPE_SUPPORT_LIST={f32,f16,bf16})；**axis=1 且 bf16 非 regbase 不支持**；SortAiCpu 兜底；**0-dim→values=self cast、indices=全 0**；AICORE 实际 dtype 仅浮点（整型走 AICPU cast 通路） | §2.2、§6 |
| argsort aclnn | `op_api/aclnn_argsort.cpp`(245)+`.h` | ✅ | 姊妹算子接口面（Sort → 只回 indices），登记不映射为主 | §2.2 |
| docs | `docs/aclnnSort.md`(14131)、`aclnnArgsort.md`(12435) | ✅ | aclnn 接口契约（param type attr、支持列表、边界） | §2 |
| golden | `tests/assets/golden.py`(200+) | ✅ | **`_require_stable` 硬抛错=ST 只覆盖 stable=True**；torch.sort(dim,descending,stable=True)；**y2_dtype 3→int32/9→int64**；four 注入模式；sort_golden(axis=-1) vs aclnn/dim=0 两轴默认 | §测试设计 |
| ST/UT | `tests/` | ✅ | 登记 | §测试设计 |

## 语义速览（P1 详抽对象）

`values,indices = sort(x, axis, descending, stable, y2_dtype)`：沿 axis 排序，values=x dtype 有序输出，indices=同 shape 轴内原位置（y2_dtype 选 i32/i64）。**golden 语义细面**：① 排序语义=torch.sort(**stable=True**)——**稳定排序，tie 保输入序（左most 先出）**，与 topk golden 同 family 双射；② **ST/UT 只覆盖 stable=True 契约**（`_require_stable` 对 stable=False 硬抛 ValueError）——骨架必须实现稳定排序语义（不能 unstable）；③ descending 双向、axis 任意（非尾轴支持 [2,2048]）、y2_dtype controls indices dtype；④ 注入模式 duplicate/signed_zero/nan/infinity（与 topk 同构，nan 处理见 sort 半序键）；⑤ 0-dim 特例 indices=全 0；⑥ regbase(910_9372) 恒 AiCore 主路径、支持任意轴+i64 indices，非 regbase 退化仅尾轴浮点。算法族：radix 多趟为主、small axis 走 insertion/two-stage、非尾轴 small(non-last) 走批次化、axis=1 走 axis-one-copy、大尾轴走 merge。

## 缺省项与注意点（P1 必须处理）

1. **判定树与 topk 卡共享**：packed-key 已证废弃（C-027 vsort 仅 f16/f32 + int64 编译失败）、argmax int64 已证次优（C-028）→ sort 主路线二投票：**a) 迭代 max+argmax 选择（topk v3 同构，稳定左most由 argmax 键 BLOCK-offs 得） vs b) `tl.sort` f32/16 直排 + 索引回查（但要解决 tie 破歧+稳定索引）**。①门实测 f32 tl.sort 正确性为关键分歧点。
2. **稳定排序是硬约束**（ST 契约，不能 unstable）——这决定不能用「值→位置」单向直排而要保证 tie 左most，与 topk 的稳定 tie 完全同构，可复用其位序键+argmax 技巧。
3. **y2 indices i32/i64** store 端 cast；索引=轴内原始下标。
4. **任意轴**：非尾轴[2,2048] 走批次化（movedim+contiguous，topk 卡 α trap 防坑已证须 `.contiguous()`）；尾轴主路径。
5. **descending**：constexpr 序反转（位序键 ~key，topk 卡同构，NaN 两向均正确）。
6. **0-dim 特例**：values=self、indices=全 0（wrapper 早退或特判）。
7. **regbase 主路径 dtype 仅浮点**（f32/f16/bf16）；整型 def 支持但 aclnn 实际走 cast/AICPU——Triton 骨架覆盖浮点主路径，整型作语义面登记。
8. **argsort 姊妹**登记接口面（sort 取 indices，不单独映射）。
9. 大轴 merge 分派/稳定 sort 库调用机制推断，golden 语义补偿。

## P0 质量门结论

材料齐（kernel/tiling/aclnn/golden 全在仓，无半壳缺省项；dtype/阈值/分派/常量全部可锚）→ **通过，进入 P1**。判**非低价值**（排序家族主卡、❌≥5、复属性语义面 6 项、稳定契约硬约束、后端排序指令限制已由 topk 卡 C-027/C-028 预证）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。