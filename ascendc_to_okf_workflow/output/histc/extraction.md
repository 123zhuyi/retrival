# P1 语义抽取 — histc（histogram_fixed_width）

> 生成时间：2026-09-18｜依据：`output/histc/manifest.md`（**非低价值** → P2.5 降级①+②，P4 不豁免）
> 根目录 `src/ops-math/math/histogram_fixed_width/`；kernel 文件均指 `op_kernel/arch35/`。

## 1. 功能与公式

- `y[b] = #{i : bin(x_i) = b}`：以 range=[min,max] 为统计上下限划 nbins 个等宽区间，统计 x 各元素落区计数（README 功能说明）。
- **越界归边**：<min 计入首 bin，>max 计入末 bin（README）。
- bin 计算式（kernel DoBin，simt_full_load.h:81-96，三档共用）：
  - 浮点域（x=f32/f16，COMPUTE_TYPE=float）：`idx = int32( (clamp(v,min,max) − min) · invRange · bins )`，invRange=1/(max−min) 预计算（full_load.h:167-168），截断取整后 clamp[0, bins−1]；
  - 整型域（x=i32/i64，COMPUTE_TYPE=**int64**，histogram_fixed_width.cpp:35-41）：`idx = int32( (clamp(v,min,max) − min) · bins / range )`，int64 乘除截断，clamp 同。
- **特殊值**（DoBin :72-79）：v==+inf → 丢弃；NaN → 丢弃；v==−inf 且 min≠−inf → 丢弃；**min==−inf → 全部有效元素进末 bin**；**max==+inf → 全部进 bin 0**（−inf 元素在 min=−inf 时保留并随之进末 bin）。
  - ⚠️ **kernel/golden 源内矛盾（P2.5 ②门实测抓获）**：场景（x 含 −inf、min 有限、max=+inf）——DoBin :72-74 的 skip 链在 max==+inf 分支**之前**执行，−inf 元素被丢弃（计数 500）；golden(spec.py) ±inf 分支只排除 +inf/NaN，−inf 保留进 bin 0（计数 600）。golden 与 torch.histc clamp 语义一致，**验收以 golden 为准**；骨架 valid 条件改为 `~(ninf & ~min_ninf & ~max_pinf)`。疑似官方 kernel 边界 bug，登记待 upstream 确认。
- golden（spec.py:26-105）与 DoBin **逐式同构**：fp32 域 `np.float32(diff·inv_range·bins)` 同序、int64 域同式、±inf 三分支一致；fp16 升 fp32（:39-42）。

## 2. 接口与原型

### 2.1 GEIR op（histogram_fixed_width_def.cpp）
- x：{F32, F16, I64, I32}，ND（:22-26）。
- **range：tensor [2]，与 x 同 dtype**（tiling 校验，tiling:86-91），ValueDepend OPTIONAL（:27-32）；**min<max 强制**（tiling:108-115，否则报错）。
- **nbins：I32 标量 tensor**，ValueDepend（:33-37）；host tiling 读其值（tiling:136-140）。
- y：I32 [nbins]，**InitValue(0)**（:38-42）；y 长度==nbins 校验（tiling:144-150）。
- Attr dtype：预留，默认 3（:43）——无语义。
- DynamicRank/Shape（:44-47）；opFile=histogram_fixed_width；**AICore 配置仅 ascend950**（:48-49）。

### 2.2 接口层
- **无 aclnn 层**（GEIR-only，仅 examples/test_geir_histogram_fixed_width.cpp）——与 fused_bias_leaky_relu 同形态。
- torch 对应：`torch.histc(input, bins, min, max)`——语义对齐（等宽、越界归边），但 torch 返回**输入同 dtype**（浮点直方图），本算子 y 固定 int32；dtype 差异登记。
- 平台矛盾登记：README 平台表含 A2/A3/Atlas 推理 ✓，def AICore 仅 950——非 950 平台走仓外实现（本卡只管 950 AICore）。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype | x∈{f32,f16,i32,i64}，range 同 x，nbins=i32，y=i32 | def.cpp:22-42 |
| range | [2] tensor，min<max，dtype==x dtype | tiling:86-115 |
| 属性 | dtype 预留（默认 3），无语义 | def.cpp:43 |
| 平台 | ascend950 AICore（README 称多平台=仓外实现） | def.cpp:49 |
| 特殊值 | +inf/NaN 丢弃；−inf 视 min；min=−inf→末 bin；max=+inf→首 bin | DoBin:72-79 |

## 4. 分核策略与 tiling（✅ 全部在仓）

### 4.1 host 决策（tiling_arch35.cpp:155-207）
- **三档 LoadMode**（:161-169）：
  - `bins < ubNumCanUse`（=(ubSize−**32KB SIMT dcache**)/4B，:157-158 + .h:36）→ **UB_FULL(0)**；
  - 否则 `totalLength > bins / 100`（**GM_ATOMIC_ADD_FACTOR=100**，.h:39——「GM 逐元素原子加 ≈ 100 倍 UB 路径代价」的官方启发式）→ **UB_NOT_FULL(1)**；
  - 否则 → **GM_ATOMIC(2)**（bins 大且 x 短：重扫 ubLoop 遍比逐元素 GM 原子更亏）。
- x 均分：`formerLength=ceil(total/coreNum)`、needXCoreNum、尾核 tailLength（:171-173）——**每核一段连续 x**。
- y 清零均分：clearYFactor=ceil(bins/coreNum)（:175-181）；needCoreNum=max(needXCoreNum, clearYCoreNum)；blockDim=needCoreNum，ScheduleMode=1（:198-199）。
- **TilingInputsDataDependency({range, nbins})**（:230）——range/nbins 值变 → host 重 tiling。

### 4.2 tilingdata（histogram_fixed_width_tilingdata.h）
bins / ubNumCanUse / ubLoopNum / needXCoreNum / formerLength / tailLength / clearYCoreNum / clearYFactor / clearYTail / needCoreNum。tilingkey=LoadMode 单参（tilingkey.h:23-27）。

## 5. 片上内存

- UB_FULL：yQue 1 份 bins×4B（full_load.h:158）+ SIMT dcache 32KB 预留。
- UB_NOT_FULL：yQue 1 份 ubNumCanUse×4B（not_full_load.h:79），ubLoop 段复用。
- GM_ATOMIC：**零 UB buffer**（无 TPipe 使用，simt_not_full_load_simt.h:26-47）。
- x 不搬 UB——SIMT 线程 `asc_ldcg`（L1 cache load）直读 GM（full_load.h:120-125），UNROLL=6。

## 6. 计算语义（核心资产）

### 6.1 DoBin（三档共用，full_load.h:67-99）
见 §1 公式与特殊值。累加原语 = `asc_atomic_add(y+idx, 1)`（UB 或 GM 视档）。

### 6.2 UB_FULL（loadMode=0，full_load.h:162-210）
1. SimtCleanWsReduce 清 GM y 段（:178-179）→ **SyncAll**（:182，等全核清零完成）；
2. 核私有 UB 直方图 Duplicate(0)（:186-187）→ SIMT 512 线程扫本核 x 段，UB 原子加（UbSimtComputeWsReduce :102-137）；
3. `SetAtomicAdd<int32>` + DataCopyPad 整段 GM 原子累加 → SetAtomicNone（:201-205）——**每核一次 GM 原子写回**。

### 6.3 UB_NOT_FULL（loadMode=1，not_full_load.h:193-226）
ubLoop 段循环：段内 UB 私有（ubNumCanUse 个 bin）→ **全量重扫本核 x 段**，只统计 bin∈[ubBase,ubEnd)（DoBinNotFull :115-117）→ 段末 GM 原子写回（AtomicAddToGm :215-226）。**x 被读 ubLoopNum 遍**——用读放大换 UB 容量。

### 6.4 GM_ATOMIC（loadMode=2，not_full_load_simt.h:140-166）
清零+SyncAll 后，SIMT 线程对本核 x 段**逐元素直接 `asc_atomic_add` GM**（DoBinAtomic :99）——无 UB、无重扫。Triton 唯一可同构路径。

### 6.5 精度策略
- f16 输入：load 后 static_cast<float>（DoBin 参数 COMPUTE_TYPE=float）——**fp32 域计算 bin**，idx 截断 int32；计数本身是整数精确。
- i32/i64：int64 域 `(c−min)·bins/range`——`(c−min)·bins` 可超 int32（大 range×bins），**int64 必须保留**，否则溢出错 bin。
- 累加 int32 原子加，精确。

## 7. 数值精度

- bin 归属：浮点域 fp32 乘-截断 vs 整型域 int64 除-截断——**两域边界归属可能不同**（同一真实值在两式下可落相邻 bin；golden 与 kernel 同式，验收以 golden 为准）。
- y 计数 int32 精确（atomic add 无舍入）。
- numel > 2³¹ 时 y 溢出 int32——超规格，登记。

## 8. 测试与标杆

- golden：spec.py golden（§1 同构）；输入生成 customize_inputs 把 float x 预缩放到 [0,nbins]（测试构造，非算子语义）。
- UT：host tiling（test_..._tiling.cpp）+ infershape——**无 ST atk json、无仓内 benchmark**。
- 验收建议：计数精确相等（int32 直方图无容差概念）；特殊值 case（NaN/±inf/越界）必须单列。

## 9. 偏差与风险清单

1. ~~`tl.atomic_add` 向量索引 scatter 后端未取证~~ → **①门实测结案（2026-09-18，910_9372+CANN 9.1.0）**：编译通过但**错值**——向量内重复地址的 atomic 更新被静默丢弃（全同地址 1024→128=恰好 1/8；两值交替 1280/4096；异地址 63/63 正确）。向量 scatter atomic 在本后端**不可用于直方图类语义**，已回流 backend_facts 契约（C-018 同族）。替代：路线 1 矩阵计数（主）/ 路线 2 逐元素 1 宽向量 atomic（= 官方档 2 真同构，正确性兜底）。
2. UB 私有直方图（档 0/1）Triton 结构性 ❌：无可寻址 UB scratch；SetAtomicAdd+DataCopyPad ❌——唯一同构路径是档 2（GM 原子，且须 1 宽向量逐元素形式）。
3. ~~向量化替代路线~~ → **路线 1 实测为主路径**：program 独占 BLOCK_BINS 个 bin 等值比较累加，无原子；实测约束 **BLOCK_X×BLOCK_BINS ≤ 128×64**（128×128 触发 ub overflow 1606400>1572864 bits，BiShengIR 自动关 code-motion/multi-buffer 后仍失败）。bins 大时比较代价 numel×BLOCK_BINS 退化 → 切路线 2。
4. int64 bin 计算的 4a 标量化预警——但官方 SIMT 本就逐元素标量，**非回归**（卡片写明，勿当缺陷警）。
5. nbins/range 值依赖 host tiling——运行时变 nbins 场景行为（重 tiling）登记。
6. 浮点/整型 bin 式不同（§7）——wrapper 必须按 x dtype 走 IS_INT constexpr 两式，勿统一成浮点式（大 int 值 fp32 精度不够）。
7. SyncAll 清零屏障 Triton 不需要（wrapper torch.zeros + 流内序）。
8. **0-d 标量张量算术编译不过**（①门抓获）：`tl.load(x_ptr+off)` 得 0-d 张量，后续算术触发 `failed to legalize unresolved materialization from () to 'tensor<1xf32>'`（triton-adapter-opt ConvertTritonIRToLinalgIR 阶段）——标量语义须用 1 宽向量（`off + tl.arange(0,1)`）承载。已回流 backend_facts 契约。

## P1 质量门

- [x] 三层语义齐备（kernel 三档 / GEIR 层（无 aclnn）/ golden+UT 层）
- [x] 关键论断带 file:line 锚点；全部实现仓内一手
- [x] ❌ 候选逐项识别（UB scratch 原子/SetAtomicAdd 写回/SyncAll/SIMT 模型）+ 降级方向（GM 原子同构 / 向量化矩阵计数）
- [x] **非低价值确认 → P2.5 降级①+②**，P4 不豁免
