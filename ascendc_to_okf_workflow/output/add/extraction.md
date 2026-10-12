# P1 语义抽取 — add

> 生成时间：2026-09-03｜来源：`Triton-Ascend-KernelWiki/kernelwiki/artifacts/ops-math/math/add/`（以下锚点均为相对该目录路径）
> 标杆对照：`ascendc-kernelgen-data/npu_benchmark/level1/3_Add.{py,json}`（NPUKernelBench level1）

## 1. 功能与公式

- 功能：逐元素二目加，支持 numpy 式广播（README L20-28）
- GEIR 层公式：**y = x1 + x2**（README L24-28；无 alpha）
- aclnn 层公式：**out = self + alpha × other**（`op_api/aclnn_add.cpp:318` `aclnnAddGetWorkspaceSize(self, other, alpha, out, ...)`；alpha 为必选标参，`CheckNotNull` 校验 L77-81）
- infershape：输出形状 = 两输入广播形状，委托 base 库 `Ops::Base::InferShape4Broadcast`（`op_host/add_infershape.cpp:17-23`）

## 2. 接口与原型

### 2.1 GEIR op（算子原型，add_def.cpp）

| 参数 | 方向 | dtype | format | 备注 |
|---|---|---|---|---|
| x1 | 输入 | BF16/FP16/FP32/INT32/UINT8/INT8/INT64/BOOL/COMPLEX32/COMPLEX64（L24-27） | ND + UnknownShapeFormat（L32-35） | 动态 rank/动态 shape（L67-68） |
| x2 | 输入 | 同 x1（L38-41；跨 dtype 组合见 2.3） | ND | — |
| y | 输出 | 同 x1（L52-55；mix-dtype 时为 FP32） | ND | — |

属性：**无**（GEIR Add 无任何属性，alpha 不存在于算子层）

### 2.2 aclnn 对外接口（op_api/aclnn_add.cpp）

- `aclnnAddGetWorkspaceSize(self, other, alpha, out, workspaceSize, executor)` L318；`aclnnAdd(workspace, ...)` L683；inplace 版 L664-672
- 另有 `aclnnAddsGetWorkspaceSize` L558（alpha 为 scalar 的便利封装）、`aclnn_add_v3.cpp`（247 行，V3 变体）
- alpha 类型校验：必须能 cast 到 promoteType（`CheckPromoteType` L201-228）
- **contiguous 策略**：`l0op::IsAddSupportNonContiguous` 为真时直接 view 透传 stride（L354、373-374），否则先 `Contiguous`（L358-364）

### 2.3 原型约束（tiling 层 dtype 规则，add_tiling_arch35.cpp）

- `IsMixedDtype` L37-43：仅允许 (FP16,FP32)、(FP32,FP16)、(BF16,FP32)、(FP32,BF16) 四种跨 dtype 组合
- `CheckDtype` L45-69：
  - 非 mix：x1/x2/y 三 dtype 必须全同，否则报错（L49-58）
  - mix：y 必须是 FP32，否则报错（L60-67）
- 同型 FP16/BF16/FP32 输出 dtype = 输入 dtype

## 3. 规格约束

| 类别 | 取值 | 锚点 |
|---|---|---|
| 平台 | GEIR 仅注册 ascend950（L72）；README 产品表另列 A3/A2（README L6-15，老平台走其他 kernel 仓） | §3 |
| dtype | 见 2.1；13 个 binary 实例见 add_binary.json（BF16、BF16_F32、BOOL、CPX32、CPX64、FP16、FP16_F32、F32_BF16、F32_FP16、F32、INT32、INT64、INT8、UINT8） | §3 |
| format | ND；FormatAgnostic（binary json shape=[-2] rank 无关） | §3 |
| 确定性 | 天然确定（纯 elementwise，无 atomic/无跨核共享） | — |
| 交付 | aclnn（Add/Adds/AddV3/Inplace*）+ GEIR 图模式 | §3 |
| workspace | GEIR kernel 无 workspace（`GetWorkspaceSize` 空实现 L143-146）；aclnn 层 workspace 由 executor 按需（Contiguous/Cast 中转，L479） | §5 |

## 4. 分核策略与 tiling（✅ 源码锚点已补齐，来源 op_common base 库）

> 2026-09-03 补：`model/opbase-master-src-op_common/src/op_common/atvoss/broadcast/` 提供了 BroadcastBaseTiling 的完整实现（原标注"base 库不可得"解除）。以下锚点 `bt` = broadcast_tiling.cpp，`btb` = broadcast_tiling_base.cpp，均相对 op_common 目录。

### 4.1 分核（源码事实）

两级切分：**UB 内分块**（GetBlockSplitFactor，bt:494-536）→ **核间分块**（DoBrodcastTiling，bt:538-607）。

1. **maxElemNum（= Triton BLOCK 的 Ascend C 对应物）**：`BroadcastGetMaxElemNum`（btb:36-45）= `(ubSize - extraSize) × BROADCAST_BITS_NUM / bufferDivisor`，再**按 256B 重复粒度向下对齐**（`alignFactor = 256B × BITS / minDtypeBits`，btb:42-43）——BLOCK 由 UB 容量反推且对齐到 256B
2. **UB 内切分**（bt:494-536）：从最内轴累乘至超 maxElemNum 处定为 ubSplitAxis；`ubFormer = maxElemNum / 内轴外乘积`（内块宽）、`ubOuter = ceil(dim[split]/ubFormer)`、`ubTail`；`fusedProduct = ubOuter × split 轴之前各轴乘积` = 核间总块数
3. **核间分块**（bt:565-566）：`blockFormer = ceil(fusedProduct / coreNum)`（每核块数），`blockNum = ceil(fusedProduct / blockFormer)`（实际用核数）
4. **preferMultiCore 回调**（bt:569-586）：blockNum < coreNum 时按 CACHE_LINE 步长缩小 maxElemNum、重算 split factor，直到块数够铺满核（或 maxElemNum ≤ CACHE_LINE_512 保底）——**官方"多核优先"策略**
5. 尾核：`blockTail = fusedProduct - (blockNum-1)×blockFormer`（bt:588）
6. 连续性与维度：连续输入走 `DimensionCollapse` 合轴（bt:151-269），非连续走 `NonContiguousDimensionCollapse` 的 KEEP/MERGE/ABSORB 计划（bt:271-333, 344-492）；合轴后维度 ≤ 8（BROADCAST_MAX_DIMS，bt:632-634）

### 4.2 tilingkey 语义（部分恢复）

`innerKey = computeKey × BROADCAST_COMPUTE_KEY_OFFSET + scheduleKey`（bt:604-605）：computeKey 恒 1（btb:20,26）；scheduleKey = `BroadcastGetScheduleKey(axisInsideUB)`（btb:28-34）——UB 内轴数 ≤ 5（MAX_NDDMA_DIM，NDDMA 搬运上限）→ 100，否则 1100。即 **tilingkey 只区分"UB 内轴数是否超 ND DMA 5 维上限"两态**；DEFAULT_SCHEMODE=1（btb:24）。

### 4.3 BLOCK 候选（Triton 转化用）

Ascend C 的 maxElemNum 公式（§4.1-1）给出 BLOCK 上限的官方推导式；Triton 侧 {1024, 2048, 4096, 8192} autotune，key=["NUMEL", "SAME_SHAPE"]。preferMultiCore 策略提示：BLOCK 过大导致核数占用不足时，应优先缩小 BLOCK 而非让核空闲——autotune 空间设计已覆盖该语义。

### 4.4 tilingdata 字段（由 bt:593-602 补全）

`BroadcastTilingData`：ubSplitAxis / ubFormer / ubOuter / ubTail（UB 内切分四元组）+ blockFormer / blockNum / blockTail（核间切分三元组）+ dimProductBeforeUbInner / elemNum(=maxElemNum) / innerKey / dims / strides / shapeLen。

## 5. 片上内存

- GEIR kernel：输入/输出全经 UB 队列（MemCfg=MemLevel::LEVEL_2，add_dag.h:39/56/72/87），无 GM workspace
- 每元素生命周期：CopyInBrc（按广播展开搬入）→（cast）→ Add →（cast）→ CopyOut；中间零驻留
- UB 预算不可算（BLOCK 不可得），elementwise 场景 BLOCK×(2+1) tile 极小，无风险

## 6. 流水编排（DAG 模板语义，add_dag.h —— 本算子核心语义资产）

四类计算模板（kernel `add.cpp:24-51` 按 dtype constexpr 分派）：

| 模板 | dtype 范围 | DAG 链（行号） | 精度语义 |
|---|---|---|---|
| `AddWithCastCompute<T>` | T=half/bf16/**float**（add.cpp:43-46） | CopyInBrc×2 → Cast<float,T,NONE>×2 → **Add<float>** → Cast<T,float,**RINT**> → CopyOut（add_dag.h:43-58） | 输入升 fp32，fp32 相加，**RTNE 舍入**回 T |
| `AddMixDtypeCompute<T1,T2>` | fp16/bf16 × fp32（add.cpp:39-42） | CopyInBrc×2 → Cast→float×2 → Add<float> → CopyOut<**float**>（L60-74） | fp32 直出，不回 cast |
| `AddWithoutCastCompute<T>` | int8/uint8/int32/int64/complex（add.cpp:47-50） | CopyInBrc×2 → Add<T> → CopyOut（L30-41） | 原生精度直加 |
| `AddBoolCompute<int8>` | bool（add.cpp:35-38） | CopyInBrc×2 → Add<int8> → **Mins(const 1)** → CopyOut（L76-90） | min(x1+x2,1)，布尔或 |

- cast 模式常量：CAST_NONE_MODE=0 / CAST_RINT_MODE=1（L27-28；RINT=round-to-nearest-even）
- ⚠️ 注意：T=float 时也走 WithCast 分支，`Cast<float,float>` 为恒等，实质 = 纯 fp32 相加

### aclnn 层 alpha 消化策略（op_api/aclnn_add.cpp，alpha≠1 时的 4 条路径）

1. alpha==1 → 直调 `l0op::Add`（L371-403；mix-dtype 且 alpha==1 走 L0 混合 kernel，L352-365）
2. `IsSupportAxpy(promoteType)` → `l0op::Axpy(self, other, alpha)` **单 kernel 融合**（L404-421；dtype 清单为外部常量 AXPY_DTYPE_SUPPORT_LIST L294-300）
3. `IsSupportAxpyV2`（RegBase 平台）→ `AxpyV2` alpha 张量化（L422-442）
4. 兜底 → `Mul(other, alpha)` + `Add` 两 kernel（L443-466）
5. 输出统一经 Cast 到 out dtype + ViewCopy（L470-476，支持非连续 out）

**结论**：alpha 是 aclnn 层语义，GEIR kernel 无 alpha；aclnn 优先单 kernel Axpy 融合。

## 7. 数值精度

- fp16/bf16：fp32 中间精度 + RTNE 回写（add_dag.h:48-51）——与 torch CPU/CUDA half add 的 float 累加行为一致
- fp32：直加（无精度变换）
- int：原生 wrap（无饱和）
- bool：min(x1+x2,1)
- 性能 flags：`PrecisionReduceFlag(true)`（def.cpp:70）——允许精度换性能（未观察到实际影响点）

## 8. 测试与标杆

- ST：tests/st/arch35/ttk_kernel_add_st.csv（acl nn 层，广播形态丰富：异 rank、(1,) 标量、8 维、625 万 1D、mix-dtype、bool、complex；abs 精度 1e-8）
- UT：infershape/tiling/aclnn/aicpu 四层
- benchmark 标杆：`3_Add.py` L68 `torch.add(x, y, alpha=alpha)`；输入生成 L90-97：**偶数 case x~N(μ∈[-100,100], σ∈[1,25])（大数值），奇数 case x~U(-5,5)，y~randn**，alpha 取自 json
- benchmark case 域（3_Add.json，41 条）：1D/2D/3D 等形、[M,1]/[1,N]/[N] 单向广播、4D×2D 尾维广播、[1,C,1,1] 逐通道广播、大 2D [4096,8192]/[8192,4096]；dtype fp32/fp16/bf16 同型；alpha∈{0.5,2,-1,1.5,3,0.25,-0.5,1}

## 9. 偏差与风险清单

| # | 偏差 | 影响 | 处理 |
|---|---|---|---|
| 1 | README 参数表列了 INT16/FLOAT64/COMPLEX128/STRING，def.cpp 与 add_binary.json 均无 | 文档过期 | 以 def.cpp+binary json 为准 |
| 2 | GEIR 算子无 alpha，aclnn 有 alpha 且 4 策略分解 | benchmark alpha≠1 case 的语义锚点在 aclnn 层（Axpy 融合路径），不在 kernel | 卡片公式以 `out = x + alpha·y` 为准，注明 kernel 本体只有 x1+x2 |
| 3 | ~~调度器/tiling 计算器/infershape 均在 base 库~~ **已解决（2026-09-03）**：opbase-master-src-op_common 提供了 tiling 计算器与 infershape 实现（broadcast_tiling*.cpp 全量、infershape_broadcast_util.cpp）；**仍缺**：kernel 侧 BroadcastSch 调度器本体、broadcast_tiling.h 头文件（struct 字段/常量如 CACHE_LINE 具体值从 .cpp 使用处反推） | §4 已从 inferred 升级为源码锚点；§6 调度器仍 inferred | 见 §4 新锚点 |
| 4 | AXPY_DTYPE_SUPPORT_LIST 等 dtype 清单为外部常量 | alpha≠1 时某 dtype 走 Axpy 还是 Mul+Add 不可知 | 该常量在 op_common 中亦未见（可能在 ops-math 仓）；benchmark 只含 fp32/fp16/bf16，生成侧单 kernel 融合 alpha 优于 aclnn 分解策略，无需复刻 |
| 5 | benchmark 偶数 case 大数值（μ±100、σ≤25、alpha 至 ±3） | fp16/bf16 输出溢出/舍入风险；fp32 大数+小数对消 | §9 验收卡记录值域；fp32 中间精度必选 |
| 6 | ST 含 benchmark 未覆盖形态（高维异 rank、(1,) 标量、int/complex/bool） | 验收域差异 | 以 benchmark 41 条 case 为验收域，ST 作扩展参考 |

## P1 质量门

- 公式/接口/dtype 规则/DAG 语义/alpha 策略均有仓内锚点 ✅
- §4 分核/BLOCK 已从 inferred 升级为 op_common 源码锚点（2026-09-03 补，broadcast_tiling*.cpp）；§6 kernel 侧调度器仍 inferred（BroadcastSch 本体不在 op_common 拷贝内）✅
- 6 条偏差全部登记（#3 已解决）✅
- 进入 P2
