# P1 语义抽取 — sum（reduce_sum）

> 生成时间：2026-09-18｜依据：`output/sum/manifest.md`（**非低价值** → P2.5 降级①+②，P4 不豁免）
> 根目录 `src/ops-math/math/reduce_sum/`；base 库 tiling 指 `model/opbase-master-src-op_common/src/op_common/atvoss/reduce/reduce_tiling.cpp`。**半壳算子**：DAG+tiling 在仓，调度器本体（ReduceSch/Vec::ReduceSumOp/GetPromoteType 定义）不在仓——不可见部分标「机制推断」。

## 1. 功能与公式

- `y = Σ x over axes`：对 axes 指定的任意组合维度求和（README「返回给定维度中输入张量每行的和」）。
- **keep_dims**（Attr，默认 false，def.cpp:40）：输出是否保留被规约维（长度 1）。
- **noop_with_empty_axes**（Attr，默认 true，def.cpp:41）：axes 为空时直返 x（identity）；False 时 = 全轴规约（golden.py:30-35 同构）。
- **dtype 输出类型独立指定**（aclnn 层参数）：self≠dtype 先 Cast 到 promoteType 再规约（aclnn:245-249）；GEIR def 层 y dtype 查表：bool→int64，其余与 x 同型（def.cpp:23-25）。
- **空 dims（aclnn 层）→ 全轴规约**（aclnn:252-260）；**all-reduce 快路径**：输出标量且输入连续 → tiling 不读 axes 常量直接全轴（tiling GetInputParam :213-219）。
- golden（golden.py:26-49）：**f16/bf16 升 fp32 求和后回原型**；其余 dtype np.sum 直算（i32→np.sum 默认升 i64？——np.sum(int32) 返回 int64，但 golden 未显式 cast 回 i32，需验收时以 np 默认行为对齐，登记）。

## 2. 接口与原型

### 2.1 GEIR op（reduce_sum_def.cpp）
- x：{F16, F32, BF16, I32, I64, BOOL}×2 组，ND，支持非连续（:19-21, :27-33）。
- axes：{I32, I64} tensor，ValueDepend OPTIONAL（:36-38）——host tiling 读常量值。
- y：BOOL→I64，其余同 x（:23-25）。
- Attr：keep_dims（默认 false）、noop_with_empty_axes（默认 true）。
- DynamicCompileStatic/Rank/Shape 全开；**ascend950 + ascend350 双 AICore 配置**（:53-54），opFile=reduce_sum_apt。

### 2.2 aclnn 层（aclnn_reduce_sum.cpp，= torch.sum 语义）
- `aclnnReduceSum(self, dims, keepDims, dtype, out)`：dtype 参数即输出类型；self≠dtype → Cast(self, promoteType) 后规约（:245-249）。
- **bool+int64 特判**（:261-268）：reduce 维元素数 < 2²⁴(16777216) → 经 **FLOAT** 中间域（fp32 精确表示 2²⁴ 内整数），否则经 INT64——精度驱动的中间域切换。
- **非连续双路径**（:177-189, 269-301）：满足 IsNonContiguousSupport（含 self dtype==promoteType 等）→ view 直算 + Cast + ViewCopy；否则 Contiguous 化后规约。
- torch 对应：`torch.sum(input, dim, keepdim, dtype=)`——语义完全对齐（torch.sum 的 dtype 参数同义）。

### 2.3 平台
def：950+350 AICore；AICPU 兜底存在（tests/ut/op_kernel_aicpu）。README 平台表 dtype 含 DOUBLE/COMPLEX64/128（Atlas 推理/训练系列）= 仓外实现，本卡只管 def 表 6 dtype。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype | x∈{f16,f32,bf16,i32,i64,bool}，y=bool→i64 其余同型 | def.cpp:19-25 |
| axes | i32/i64 tensor 常量，负轴归一（MakeWrapDim :271-279：+ndim 后排序） | def.cpp:36-38; tiling |
| shape | 0-8 维，非连续 ND | README; def |
| 属性 | keep_dims=false 默认；noop_with_empty_axes=true 默认 | def.cpp:40-41 |
| 平台 | 950+350 AICore；AICPU 兜底 | def.cpp:53-54 |
| 确定性 | **默认确定性实现**（README 约束说明） | README |

## 4. 分核策略与 tiling（✅ 计算器在仓，调度执行细节半壳）

### 4.1 形状预处理 → Pattern（tiling:613-620, 1262-1290）
TransformShape：EliminateOne（去长度 1 维）→ PadDimForNonContiguous → MergeAxis（相邻同 A/R 属性且地址连续维合并）。预处理后维数 1~9 分派 **A / AR / ARA / ARAR / …（A=规约轴、R=保留轴交错）**，5/7 维 pad 到 8/9；A/R 由奇偶位判定（IsAxisA :284-291）。**任何 axes 组合最终归约为交错 AR pattern**——这是官方对任意规约的统一抽象。

### 4.2 5 步 ComputeTiling（:1061-1091）
1. **basicBlock**（CalcBasicBlock :966-1044）：UB 可用 = ubSize − CACHE_BUF_SIZE − reserved；reduce 前计算图开 DB（buffer×2）；按 cast 前后字节比 ratio 折算；FloorAlign 到 vRegSize。**resultBlock**（reduce 输出驻留块）cacheLine 对齐并 clamp [MAX_INNER_A, CACHE_BUF_SIZE]——「输出大小不超 cache 缓存，否则额外 A 轴搬入越界」（:1014-1017 注释）。
2. **cacheLine 切分轴**（ComputeCacheLineBlock :649+）：按硬件 cacheLine（如 256B）定位切分轴/step/outer，保证搬运性能；尾轴为 A 时按 UBBlock(32B) 元素数对齐（:655-657）。
3. UB 内 A 轴切分：最大 512B，或 A 轴分核利用率低于 85% 时调整（步骤注释 :1064-1067）。
4. UB 内 R 轴切分（ComputeUnitR）。
5. R 轴过小（UB 全载但 basicBlock 不满载）回调 A 轴，上限 UB 二分缓存。

### 4.3 分核（SetTilingData :1101-1145）
- 总任务块 = unitA.outer × unitR.outer，按 vectorCoreNum 均分（perCoreNum=ceil(总数/核数)）；
- **A 轴优先分核**：factorACntPerCore=ceil(outerA/numBlocks)，numBlocks 向 outerA 倍数对齐（ceil/floor 双向，capped by 核数）——每核拿整数个 A 块时避免 R 维部分和；
- R 轴再分：factorRCntPerCore=ceil(outerR / ceil(numBlocks/outerA))；
- realCoreNum = ceil(outerA/factorA)×ceil(outerR/factorR) 重算。

### 4.4 跨核归约（关键 ❌ 项）
`groupR = ceil(outerR / factorRCntPerCore)`；**groupR>1 → ① SetScheduleMode(1)（核间同步调度，:1123-1126）② GM workspace = vectorCoreNum × align(outSize × maxInputBytes, cacheLine)（CalcUserWorkSpace :1249-1259）**——各核部分和落 GM workspace 后二阶段归约（二阶段执行在缺失的 ReduceSch 本体，机制由 workspace 形状反推=每核一份 outSize 部分和）。apt.cpp:34-40 workspace 必需校验与此呼应。

### 4.5 特殊分支
- **空 tensor**（ComputeEmptyTiling :1047-1058）：去 R 轴按全 A pattern 切；outSize=0 直返。
- **batchInvariant**（:1074-1079 + reduce_tiling_batch_invariant.h）：R 优先切分（ComputeRFirst），尾轴 A=1 保留标识——batch 一致性/确定性场景。
- **useNddma**（SetTilingData :1137 + IsUseNddma）：非连续 NDDMA 搬运判定。

### 4.6 kernel 壳（reduce_sum_apt.cpp）
AIC 早退（:31-33）；`PromoteType = bool ? int64_t : GetPromoteType<DTYPE_X>::T`（:45——GetPromoteType 定义在缺失头文件，golden 佐证 f16/bf16→fp32，i32/i64 映射登记缺省）；bool/非 bool DAG constexpr 分派（:47-54）。

## 5. 片上内存

- UB 预算：ubSize − CACHE_BUF_SIZE − reserved（:969-975；常量值在缺失头文件）；basicBlock FloorAlign vRegSize；resultBlock ≤ CACHE_BUF_SIZE。
- reduce 前计算图开 DB（×2 buffer，:980-984）；reduceOpPos==1（reduce 前无计算）时输入 buffer 再 ×2 保流水（:985-987）。
- MemCfg LEVEL_2（dag.h:36/50）——内存优化级别 2（语义在缺失头文件，登记）。
- 跨核部分和走 **GM workspace**（§4.4），不占 UB。

## 6. 计算语义（核心资产）

### 6.1 非 bool DAG（reduce_sum_dag.h:28-38）
```
CopyIn<T> → Cast<PromoteT, T> → ReduceSumOp<PromoteT> → Cast<T, PromoteT> → CopyOut<T>
```
**先升域、规约、再降回**——f16/bf16 在 fp32 域累加（golden 佐证），输出回原型。ReduceSumOp 的 Reg 级展开（树形/相邻对加）在缺失本体，机制推断：linalg.reduce 同族的向量归约原语。

### 6.2 BoolDag（:40-52）
```
CopyIn<int8> → Cast<half> → Cast<float> → Cast<int64> → ReduceSumOp<int64> → CopyOut<int64>
```
bool 以 i8 读入，经 half→float→int64 三级 cast，int64 域计数累加，y=int64。（aclnn 层另有 2²⁴ 限流：小规约维走 FLOAT 中间域——aclnn:261-268，与 kernel 层 DAG 是两套中间域策略，登记差异：kernel 层 BoolDag 固定 int64 域。）

### 6.3 精度策略
- f16/bf16：fp32 中间域（DAG Cast + golden 双重佐证）——**Triton 侧必须保留升域**，直用 f16 累加会丢精度（长规约轴尤甚）。
- i32：y=i32，PromoteType 未知（缺省）——若官方升 i64 域累加则溢出行为与 np 不同；验收以 golden 为准（golden np.sum(i32) 默认升 i64 后未显式回 cast——ST 容差 1e-4 级，整数精确场景无差异，登记）。
- i64：i64 域直算（def y 同型）——**vreduce i64 在 A2/A3 标量化**（dtype_support.md:58/68 [SRC]），950 不走标量兜底（模板库路径），性能两态、正确性不受限。
- bool：§6.2。
- **确定性**：README 声明默认确定性——groupR>1 两阶段归约的合并顺序由 ScheduleMode=1 保证固定（机制推断：同步调度 → 固定合并序）。

## 7. 数值精度

- 浮点累加顺序 = tiling/分核决定（UB 内树形 + 跨核两阶段）——**不同 shape/axes 下累加序不同**，与 torch/np 的位级一致不保证；ST 容差 1e-4（csv absolute_precision 0.05/0.0001 列）。
- f16/bf16 升 fp32 是官方精度兜底，Triton 必须同构。
- 空 tensor：outSize=0 直返（tiling 特判）；全轴规约空 tensor → 0（语义登记）。

## 8. 测试与标杆

- golden：golden.py（§1）；f16/bf16 升域同构。
- ST：kernel 层 csv 仅 5 case（f32/i64，axes 单轴，keep_dims 两态）+ aclnn atk json（torch.sum 入口，bool 起始 dtype 随机化）——**覆盖偏薄**，②正确性门须自补：多轴组合/负轴/keep_dims/全轴/空 axes noop/6 dtype/非尾轴规约/大 R 轴（触发跨核）。
- 无仓内 benchmark。

## 9. 偏差与风险清单

1. **半壳**：ReduceSch/Vec::ReduceSumOp/GetPromoteType/常量表不在仓——Reg 级展开、DB 细节、promote 映射为机制推断+golden 互证；卡片相关段落标「机制推断」。
2. **跨核归约 ❌**：groupR>1 的 GM workspace 两阶段 + ScheduleMode=1 核间同步——Triton 单 kernel 无核间同步；降级：① wrapper 两段式（partial+final 两 kernel，GM 中转=workspace 同构）；② A 轴全载（每输出元素单 program 扫整条规约轴，groupR=1 形态，官方 pattern A/尾 A 场景本就如此）。P2 判定树。
3. **i64 vreduce 标量化** [SRC]：A2/A3 标量化、950 模板库路径——本栈 910_9372 ①门附取证（编译通过性+正确性），性能不外推。
4. **useNddma/非连续**：Triton wrapper 按 contiguous 接（torch.sum 语义层处理），官方 NDDMA 优化不可译（性能机制，无语义影响）。
5. **bool 中间域两套策略**（kernel BoolDag int64 域 vs aclnn 2²⁴ 限流 FLOAT 域）——Triton 对齐 GEIR kernel 层（int64 域）即可，差异登记。
6. **i32 累加域不明**（GetPromoteType 缺省）——验收以 golden 为准；大规约轴 i32 溢出风险登记（np 默认升 i64，官方 kernel 未知）。
7. **确定性声明**：官方默认确定性（固定合并序）；Triton 两段式/全载均天然确定（无原子），对齐无负担——但勿引入 atomic 归约写法（histc C-024 教训：scatter atomic 本来就错）。
8. axes ValueDepend：运行时变 axes → host 重 tiling；wrapper 按 Python int/tuple 接。

## P1 质量门

- [x] 三层语义齐备（DAG/golden/aclnn+ST 层；kernel 调度层半壳已标机制推断）
- [x] 关键论断带 file:line 锚点；tiling 全链一手
- [x] ❌ 候选逐项识别（跨核归约两阶段/NDDMA/ScheduleMode）+ 降级方向（两段式/A 轴全载）
- [x] **非低价值确认 → P2.5 降级①+②**，P4 不豁免
