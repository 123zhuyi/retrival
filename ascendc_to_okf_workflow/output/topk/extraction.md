# P1 语义抽取 — topk（top_k_v2）

> 生成时间：2026-09-18｜依据：manifest.md｜判定：非低价值（P2.5 降级①+②）

## 1. 算子语义（数学定义）

```
沿 dim 轴（负轴归一）对 x 每"行"（其余维折叠为 M）选择 k 个元素：
  largest=True ：值最大的 k 个，按值降序输出（sorted=True）
  largest=False：值最小的 k 个，按值升序输出（sorted=True）
  sorted=False ：无序（比较器语义合法即可）
values[r,j] = 选中值（dtype 同 x）；indices[r,j] = 轴内原始位置（dtype=indices_dtype，默认 i32）
k ∈ [0, N]（N=x.size(dim)），k 为 0D tensor 输入；输出 shape = x 轴维替换为 k。
```

**全序键定义（golden `_radix_keys` / `_value_better`，CUDA TopKTypeConfig 同构）**：
- 浮点：位模式 sign-flip——`u = bits ^ (sign ? 0x7FFFFFFF : 0x80000000)`（无符号保序键），**NaN → u_max（全 1）即"最大"**；-0.0/+0.0 键不同但比较相邻（golden signed_zero 模式登记）。
- uint64：`u = bits ^ INT64_MIN`（有符号容器保序）；int8~int64：零/符号扩展即有符号序。
- **tie 语义**：`_value_better` 用严格 `>`（largest）/ `<`（False）——等值不换 → bitonic 网对等值子集稳定 → **输出序 tie = 输入序（左most 先出）**。

## 2. 原型与约束

| 项 | 值（def + infershape + README） |
|---|---|
| x | i8/u8/i16/u16/i32/u32/i64/u64/f16/f32/bf16（11 种），1-8 维，ND |
| k | 0D tensor，i32/i64，ValueDepend，[0, N] |
| values | dtype=x，shape=x 轴维→k |
| indices | dtype=indices_dtype（i32/i64，默认 i32），shape 同 values |
| attrs | sorted=true、dim=-1、largest=true、indices_dtype=3(DT_INT32)、sort_policy=0 |
| 平台 | ascend950（AIC），DynamicRank/Shape |
| 特殊 | k 越界抛错（golden: ValueError）；M=0 → 空输出 |

## 3. aclnn 接口层（math/topk/op_api/topk.cpp）

三岔分派：① `IsAiCoreSupport`：输入超 MAX_AICORE_CALC_INPUTSIZE 且 k<MAX_AICORE_CALC_DIM → AICPU；② `IsRadixTopKSupport`：**f16/bf16 + 910b/910_93 + sortLen > ratio·k + k > MIN_K → 姊妹算子 radix_top_k**（N≫k 专用基数排序）；③ else AICPU。Triton 侧不映射姊妹算子策略（登记接口面）。

## 4. 分核与 tiling（host 侧，全在仓）

### 4.1 模式体系（apt.cpp + tiling_arch35）

```
MODE: SINGLE_CORE=1（行均分核，单行单核全处理）
      MULT_CORE_OPTIM=4（k·(N/tile)≤tile → 每核局选 k 候选 → 单核二段归并）
      SORT_AND_TOP_K=5（N≥1e7：sort_with_index 全排 + narrow 截断 k）
      NON_LAST_SMALL_AXIS=8（非尾轴且轴≤2048：免转置直排；keyParams0=1 走 merge sort）
isSingleBlock=lastDimNeedCore==1（尾轴单块）/ isInInt32Range（索引用 i32 域）
tiling key：radix 族 1001-1004/2001-2004/3002-3003/4002；bitonic 51001+ 族（k≥2 小 k）；
           merge sort 13003/23003/33003 族；末位 +BITONIC_SORT_TILING_OFFSET
```

### 4.2 关键常量（tiling_base.h）

SORT_AND_TOP_K_THRESHOLD=**10,000,000**；SINGLE_CORE_DATA_NUM=**15360**（B64=10240）；BITONIC_SMALL_TOPK_MIN_K=2；SUPPORT_SORT_MAX_BYTE_SIZE=8000（sorted&&k·elem≤8000B 走 sort 路径）；NON_LAST_SMALL_AXIS_THRESHOLD=2048；MAX_K_FOR_INT64=2000。TileModeMediumSize 注释：lastAxis 几十万~百万走中规模 radix；BigSize：lastDimNeedCore=min(核数, tileNum) 尾轴跨核。

## 5. kernel 计算结构（arch35 三族 + 组合）

### 5.1 radix 主族（radix_sort_top_k.h 1121 行 + 位宽 helper）

ProcessTopK：按 dtype 位宽 NUM_PASS 分趟（b8=4? 按 2bit/趟 radix）——bin 计数（hist）→ 前缀和 → 按位重排；跨核时 globalWorkGm 工作区；末段 `AscendC::Sort<T,T_INDEX_TO,false,sortConfig>` 库调用对候选精排（tmp 尺寸 host 经 GetTopkApiTmpBufferSize 预算——库本体不在仓，**半壳成分登记**：排序网络内部不可见，语义由 golden 补偿）。变体：single_core（单行）、single_block（单块）、multi_block、inter_core_template_optimization（跨核模板优化）——按 lastDimNeedCore/单核标志分派。

### 5.2 bitonic 小 k（top_k_small_size_bitonic_sort.h 1490 行）

k≤32 的 32 槽位 bitonic 网：候选载入（无效槽位 valid 掩码）→ 对数级 compare-and-swap（`_value_better` 严格比较 + invalid 槽换入）→ 输出前 k。与 golden `_small_bitonic_sort` 逐式同构（golden 即 CUDA 版复刻）。

### 5.3 merge sort 族 / sort_with_index 族 / non_last_small

merge_sort(_more_core/_intra_core/_simd)：归并路径，非尾轴或大 k。sort_with_index(entry/multi/single/merge)：(value,index) 联合排序基座——radix 按值分桶同时搬运索引。non_last_small_axis_non_transpose：轴≤2048 的非尾轴免转置方案（每行沿轴方向直读直排）。

### 5.4 库 API 缺省登记

`AscendC::Sort` / `GetTopkApiTmpBufferSize` / `GetTopkNonLastSortTmpSize` 实现不在仓——卡片标「机制推断」，语义以 golden + bitonic 文件（在仓）补偿。

## 6. 计算语义 → 精度域事实（P2 直接消费）

1. **比较域**：选择/排序全程在「全序键」域比较（§1），值本体不改动；输出值=原值（无精度变换，与 sum/layer_norm 的 fp32 域纪律不同——本算子是纯选择，**无计算域提升**）。
2. **NaN 优先级**：largest=True NaN 视为最大（先选）；False 视为最小（后选）——键映射天然给出，无需特判分支。
3. **tie**：左most 先出（严格比较稳定性，golden/CUDA 同）。
4. **indices 语义**：轴内原始下标（非排序后位置），与 sorted 无关。
5. **uint 值域**：按位映射到有符号容器保序（golden `_radix_keys`），输出值回原 dtype。

## 7. 特殊/边界语义

- k=0 → 空 values/indices（shape 轴维=0）；M=0 → 空；k=N → 全排等价（应走 sort 路线，登记）。
- 输入含 Inf/NaN/±0/全等值/重复值：golden 五注入模式（duplicate/all_equal/signed_zero/nan/infinity）为验收标配。
- sorted=False：骨架恒输出有序（超集合法）。
- 非连续输入：aclnn Contiguous 化；wrapper 同。

## 8. 测试资产

golden.py（含比较器/输入注入/third_party= torch.topk）+ tests/ ST/UT。②门基准：CPU torch.topk（uint 走 view+stable sort 参照路径）+ golden 五注入模式语义核对；k≤32&&sorted 用 exact 索引比对（左most tie 语义）；大 k 用语义比较器。

## 9. 风险清单（P2/P2.5 关注点）

1. **vsort f16/f32 限制**（已预取证）：tl.sort/topk 的 64-bit 键直接编译失败；f32 键可用性①门正式验证并回流——**packed-key 排序路线废弃**，主路线=迭代选择。
2. **argmax int64 键编译爆炸+错值**（已预取证）：i64/u64 走 max(int64)+argmax(int32 flags) 两步（probe7 证）。
3. k 运行时循环（scf.for 动态界）+ 每趟两次归约：编译/性能两态。
4. 大 N 两行段归并的 workspace 布局与 i64 索引 cast。
5. NaN 在 largest=False 时排最后（键=最小），与 torch 一致需 ②门验证。
6. BLOCK_N 上限受 UB 约束（cur/x/key 三向量），①门探 4096。
