# P0 材料清单 — topk（top_k_v2）

> 生成时间：2026-09-18｜来源目录：`src/ops-math/math/top_k_v2/`（aclnn 层在姊妹目录 `math/topk/op_api/topk.cpp`）
> **kernel 类型判定：AIV-only 选择算子（全量 radix/bitonic/merge 三族在仓，无半壳）**——20+ kernel 文件全在仓（arch35），host tiling 全在仓（base 645 + arch35 1669 行），aclnn 层在 `topk/`。语义面比 layer_norm  richer：任意 dim、largest 双向、sorted 开关、indices_dtype 可配、k 为 tensor 输入。
>
> **🔺 低价值预判：非低价值——P2.5 降级执行①编译门+②正确性门，P4 不豁免。**
> - **选择/排序家族代表卡**（sort/topk/kthvalue 同族，sort 卡紧接）；
> - **❌ 候选 ≥4**：① 多趟 radix 基数排序（按 dtype 宽度 b8/16/32/64，NUM_PASS 分趟，bin 计数+前缀和+重排）整体无 Triton 对应；② 五模式分核体系（SINGLE_CORE/MULT_CORE/MULT_CORE_OPTIM/SORT_AND_TOP_K/NON_LAST_SMALL_AXIS）+ 满核约束 + sortLoopTimes 串行趟数不可直译；③ sort_and_top_k 组合模式（尾轴 ≥1e7 时 sort_with_index 全排+截断）策略面不可译；④ SmallBitonicSort 32 槽位网（k≤32 精确 CUDA 索引语义）位级网络不可逐式译；⑤ sort_policy/indices_dtype/k-tensor 属性语义面需 wrapper 全复刻；
> - **后端敏感点（P2 前已探针预取证，①门将正式回流）**：`tl.sort`/`tl.topk` 的 bitonic 网被后端模式匹配到 `hivm.hir.vsort` 硬件指令——**仅 f16/f32 元素类型，int64 键编译失败**；`tl.argmax` int64 键编译 >7min CPU 且结果错，int32 正常；**迭代 max+argmax 选择环（int32/int64 双路线）端到端 exact**——定为 Triton 主路线；
> - golden.py 金矿：CUDA radix 键映射（sign-flip、NaN=max）、k≤32&&sorted 精确 bitonic 索引（严格大于换 → 稳定 → tie 左most）、大 k 语义比较器（valid+gather+dup）、uint 走 view+stable sort、特殊输入注入（duplicate/all_equal/signed_zero/nan/infinity）；
> - 联动结论：**走 P2.5 降级路径（①+②）**，BLOCK 选型/瓶颈定型两态，confidence 不升 measured(skeleton)。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/top_k_v2_def.cpp` | ✅ | x 11 dtype×4 组（i8~u64/f16/f32/bf16）；k 输入 REQUIRED（i32/i64）；values=x dtype；indices={i32,i64}；Attr sorted=true/dim=-1/largest=true/indices_dtype=DT_INT32(3)/sort_policy=0；ascend950 AIC DynamicRank/Shape | §2、§3 |
| infershape | `op_host/top_k_v2_infershape.cpp` | ✅ | 输出 shape=x 轴维换 k（:56-68）；k ValueDepend（InputsDataDependency{1}）；k 读 i32/i64 标量 | §2 |
| kernel 入口 | `op_kernel/top_k_v2_apt.cpp` | ✅ | tiling key 全表：1001-1004(int)/2001-4(uint)/3002-3(fp)/4002(bf16) radix 族、13003/13002/14002 merge sort、23003 more core、33003 intra core、51001-4/52001-4/53002-3/54002 bitonic、63003/63002/64002 bitonic+merge；MODE：SINGLE_CORE=1/MULT_CORE_OPTIM=4/SORT_AND_TOP_K=5/NON_LAST_SMALL=8；分派因子 isLargest/isSort/isSingleBlock/isSingleCore/isInInt32Range/keyParams0 | §4.1 |
| radix 主族 | `op_kernel/arch35/radix_sort_top_k.h`(1121) + b8/16/32/64(86-109) + base(92) + util(84) | ✅ | ProcessTopK 主流程：多趟 radix（NUM_PASS 按位宽）bin 计数→前缀和→按位重排；末段 `AscendC::Sort<T,T_INDEX_TO,false,sortConfig>` 库调用精排（tiling 取 tmp 尺寸）；isInInt32Range 决定索引用 i32/i64 | §5 |
| radix 变体 | single_core(857)/single_block(210)/multi_block/inter_core_template_optimization(378) | ✅ | 单行/单块/多块/跨核模板优化四变体；globalWorkGm 跨核工作区 | §4.2、§5 |
| bitonic 小 k | `top_k_small_size_bitonic_sort.h`(1490) | ✅ | k≤32 32 槽位 bitonic 网（SmallBitonicSort 同构），排序 k 个候选 | §5 |
| merge sort 族 | top_k_merge_sort(255)/more_core(629)/intra_core(356)/simd(170) | ✅ | 归并路径变体（非尾轴/大 k） | §5（裁剪细节） |
| sort_with_index 族 | sort_with_index_entry(167)/multi_block(174)/single_block(210)/merge_sort | ✅ | (value,index) 联合排序基座，SORT_AND_TOP_K 模式复用 | §5 |
| non_last_small_axis | `top_k_non_last_small_non_transpose.h`(337) | ✅ | 非尾轴小轴免转置直排（keyParams0=1 走 merge sort） | §5 |
| tiling base | `op_host/arch35/top_k_v2_tiling_base.cpp/h`(645/292) | ✅ | 常量：SORT_AND_TOP_K_THRESHOLD=10M、SINGLE_CORE_DATA_NUM=15360(B64=10240)、BITONIC_SMALL_TOPK_MIN_K=2、SUPPORT_SORT_MAX_BYTE_SIZE=8000、NON_LAST_SMALL_AXIS_THRESHOLD=2048、MAX_K_FOR_INT64=2000；needSortWithIndex/isBitonicSmallTopkMode 判定 | §4.2 |
| tiling arch35 | `op_host/arch35/top_k_v2_tiling_arch35.cpp`(1669) | ✅ | 模式选择：TileModeSingleCore/MediumSize(尾轴几十万~百万 radix)/BigSize/TileTopkMoreCoreMode/TileMultiCoreOptimSize(k·(N/tile)≤tile 二段归并)/sort_and_top_k(≥1e7)；lastDimNeedCore 分核数 | §4.2 |
| **aclnn 层** | `math/topk/op_api/topk.cpp` | ✅ | IsAiCoreSupport（尺寸超 MAX_AICORE_CALC_INPUTSIZE 且 k<MAX_AICORE_CALC_DIM 走 AICPU）；**IsRadixTopKSupport：f16/bf16 + 910b/910_93 + sortLen>ratio·k + k>MIN_K → 姊妹算子 radix_top_k**；else AICPU 兜底 | §2.2、§6 |
| README | `README.md` | ✅ | k∈[0, x.size(dim)]；1-8 维；sorted 语义说明；平台表 950PR/950DT/A3/A2/200I/500 | §2 |
| golden | `tests/assets/golden.py` | ✅ | 见「语义速览」——本卡语义真值主锚 | §测试设计 |
| ST/UT | `tests/` | ✅ | 登记 | §测试设计 |

## 语义速览（P1 详抽对象）

`values,indices = topk(x, k, dim, largest, sorted, indices_dtype, sort_policy)`：沿 dim 轴选 k 个极值。**golden 语义细面**：① 浮点键映射（sign-flip 保序、NaN=最大——CUDA TopKTypeConfig 同构）；② k≤32&&sorted=True → 精确 CUDA SmallBitonicSort 索引（严格大于才换 → tie 保输入序 = 左most 先出）；③ k>32 或 sorted=False → 语义比较（索引合法+gather 回值相等+无重复）；④ uint16/32/64 torch 原生不支持 → view 有符号 + stable sort 参照；⑤ 注入模式：duplicate/all_equal/signed_zero/nan/infinity；⑥ k 越界 [0,size] 抛错。算法族：radix 多趟为主，k 小走 bitonic，非尾轴小轴(≤2048)免转置，尾轴 ≥1e7 全排截断，f16/bf16 且 N≫k 走姊妹 radix_top_k。

## 缺省项与注意点（P1 必须处理）

1. **radix/bitonic/merge 网络不可逐式译**：P2 以「迭代 max+argmax 选择」重写主路径（probe7 已验证 int32/int64 端到端 exact）；tl.sort/topk 因 vsort f16/f32 限制仅作 f32 值域备选（sort 卡详查）。
2. **k 为 tensor 输入**（ValueDepend）→ wrapper 读标量，kernel 运行传参；k=0 空输出早退；k∈[1,N]。
3. **任意 dim** → wrapper movedim 到尾轴 + contiguous（官方 non_last_small_axis 免转置为性能机制，不译）。
4. **sorted=False** → 骨架恒输出有序（超集，golden 比较器对 sorted=False 先排序再比，语义合法）；登记。
5. **largest 双向** → constexpr LARGEST + 键序反转（~key），NaN 位置两向均正确（largest=True NaN 最先；False NaN 最后）。
6. **indices_dtype i32/i64** → store 端 cast；索引值=轴内原始位置。
7. **i64/u64 值域**：int64 max-reduce（sum 卡已证可用）+ int32 flag-argmax 两步（probe7 证）；u64 先 ^sign 映射 int64 保序。
8. **大 N 两行段**：N>BLOCK_N 时 tile 局选 k' 候选→workspace→按行归并（官方 MULT_CORE 二段同构）；k 接近 N 时应走全排（sort 卡），登记。
9. 特殊值：NaN 唯一映射最大键；±0 按位映射不同键但比较相等域（golden signed_zero 模式登记）； Inf 正常。
10. aclnn 层三岔（AiCore/radix_top_k/AICPU）登记为接口面，Triton 不映射姊妹算子策略。

## P0 质量门结论

材料齐（kernel/tiling/aclnn/golden 全在仓，无半壳缺省项；唯一外部依赖 `AscendC::Sort` 库调用有 golden 语义补偿）→ **通过，进入 P1**。判**非低价值**（选择家族代表卡、❌≥4、属性语义面 5 项、后端排序指令限制已预取证）→ **P2.5 降级执行①编译门+②正确性门**，P4 不豁免。
