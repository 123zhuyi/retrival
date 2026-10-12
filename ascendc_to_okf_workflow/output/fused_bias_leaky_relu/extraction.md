# P1 语义抽取 — fused_bias_leaky_relu

> 只做事实搬运，不做 Triton 转化。每个论断带 `file:line` 锚点（相对 `src/ops-nn/activation/fused_bias_leaky_relu/`）。

## 1.1 功能分析（算什么）

- **算子语义一句话**：对输入 x 加等形偏置 bias 后施加带负半轴斜率的 LeakyReLU，再整体乘缩放因子 scale（README.md:16）。
- **数学公式**（README.md:19-25）：

$$
y = \begin{cases} (x+bias)\cdot scale, & x+bias \ge 0 \\ (x+bias)\cdot negative\_slope \cdot scale, & x+bias < 0 \end{cases}
$$

  变量：x/bias/y 同 shape（≤8 维 ND，打平后总元素数 totalNum）；negative_slope 默认 0.2（def.cpp:42）；scale 默认 1.414213562373 = √2（def.cpp:43）。
- **Compute 实现核对**（op_kernel/arch35/fused_bias_leaky_relu.h:161-169）：`Add → LeakyRelu(negativeSlope_) → Muls(scale_)`，三步全在 fp32 域（fp16 路径前置 Cast 提升 L158-159、后置 Cast RTNE 回写 L165）。公式与实现一致；分段边界 t=0 处两分支同值（0），文档 `>`/`≥` 表述差异无影响。
- **语义分层**：
  - GEIR kernel 本体：上式，属性经 tilingdata 以 runtime 标量透传（tiling_data.h:25-26 → kernel h:93-94）。
  - aclnn 接口层：**不存在**（无 op_api/ 目录），无额外参数消化。
  - benchmark 层：**未收录**（ascendc-kernelgen-data 全库检索无条目），torch golden 需自建，天然无超集偏差。

## 2 原型设计（来源：op_host/fused_bias_leaky_relu_def.cpp）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 | 锚点 |
|---|---|---|---|---|---|---|---|
| x | 输入 | REQUIRED | FLOAT16 / FLOAT | ND | 0-8 维，与 bias 完全相同 | 主输入 | L24-29 |
| bias | 输入 | REQUIRED | FLOAT16 / FLOAT | ND | 与 x 完全相同（无广播） | 偏置 | L30-35 |
| y | 输出 | REQUIRED | FLOAT16 / FLOAT | ND | 与 x 相同 | 输出 | L36-41 |

- 三参数均 `.AutoContiguous()`（L29/35/41）——非连续输入由框架自动转连续，kernel 只需处理连续 ND。
- OpAICoreConfig：DynamicRankSupportFlag(true)、DynamicShapeSupportFlag(true)、PrecisionReduceFlag(true)、仅 `ascend950` config（L46-53）。

### 属性

| 属性 | 类型 | 默认值 | 语义 | 锚点 |
|---|---|---|---|---|
| negative_slope | float，OPTIONAL | 0.2 | LeakyReLU 负半轴斜率 | def.cpp:42；host 兜底 tiling_arch35.cpp:117-118 |
| scale | float，OPTIONAL | 1.414213562373 | 整体缩放因子 | def.cpp:43；host 兜底 tiling_arch35.cpp:119-120 |

## 2.1 原型约束

- shape 边界：x/bias dim ≤ 8（tiling_arch35.cpp:60/66-80）；x/bias/y 三者 **ShapeSize 相等**校验（L86-90）；infershape 直接 `*outputShape = *inputShape`（infershape.cpp:32）。
- 值域边界：属性为任意 float，无范围校验（L117-120 仅判空指针取默认）。
- 输入间依赖：x 与 bias dtype 必须一致（L110-113）；dtype 全集 {FLOAT16, FLOAT}（L94-109）。
- 拦截校验：全部在 tiling host 侧 OP_CHECK（上引）；空 tensor（totalNum=0）合法，走短路分支（L155-164）。
- **偏差记录**：`op_graph/fused_bias_leaky_relu_proto.h:42-46` 声明 dtype 含 DT_DOUBLE，与 def.cpp 不一致——以 def.cpp 为准（kernel 无 double 实例化路径）。

## 3 规格约束

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | 仅 ascend950（arch35）；README 产品表（A2/A3 等）为文档表述，以 def.cpp 为准 | def.cpp:53 |
| dtype 全集 | FLOAT16、FLOAT（x/bias/y 同型） | def.cpp:26-38 |
| format | ND（含 UnknownShapeFormat=ND） | def.cpp:27-40 |
| 确定性计算 | 是（纯 elementwise，无规约/无 atomic/无 workspace） | tiling_arch35.cpp:34（WS_SYS_SIZE=0） |
| 交付形式 | GEIR 图模式（REG_OP/OP_ADD）；无 aclnn | README.md:116；examples/test_geir_*.cpp |
| 架构差异 | 仅 arch35 一套 kernel，无 arch22 对照——不涉及 | 目录结构 |

## 4.1.1 分核策略（tiling_arch35.cpp）

- 整体打平为一维，按元素数均分到 AIV 核：
  - `ubBlockSize = 32B / typeSize`（fp16→16 元素，fp32→8 元素）（L37-39, L166-167）
  - **`blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize)`**（L169）——每核处理量按 32B 对齐上取整
  - **`usedCoreNum = CeilDiv(totalNum, blockFactor)`**（L170），`SetBlockDim(usedCoreNum)`（L199）——尾核自然少载，无显式 normal/tail core 二分
- 空 tensor：`blockDim=1`、tilingkey=0、tilingdata 全 0（L155-164）。
- kernel 侧核对：`blockOffset = GetBlockIdx()*blockFactor`；`blockLength_ = min(blockFactor, totalNum - blockOffset)`（h:79-81）；`blockLength_<=0` 直接返回（h:114-116）——blockFactor 上对齐导致末核可能空载，由该判断兜底。

## 4.1.2 单核内切分（UB 因子）

- `loopCount_ = blockLength_ / ubFactor`；`tailNum_ = blockLength_ % ubFactor`（h:89-90）；主循环全量块 + 一次尾块（h:117-126）。
- **ubFactor 计算（host，L172-191）**：
  - buffer 常数：TENSOR_NUM=3（x/bias/y 三条队列）、FP16_CAST_BUF_NUM=3（fp16 路径 xFp32/biasFp32/yFp32 三个 fp32 TBuf）、DOUBLE_BUFFER_MULTIPLIER=2（L35-41）
  - 双缓冲判定：`loopCountIfDouble = CeilDiv(blockFactor, ubSize/typeSize/(totalBufNum*2))`；`useDoubleBuffer = (loopCountIfDouble >= 2)`（L177-179）——即「单缓冲放不下整个 blockFactor 且能至少循环 2 次」才开双缓冲
  - `queueBufNum = 3 * (useDoubleBuffer ? 2 : 1)`（L182）
  - fp32：`ubFactor = FloorAlign(ubSize/4 / queueBufNum, 8)`（L190）
  - fp16：`ubFactor = FloorAlign(ubSize / (queueBufNum*2 + 3*4), 16)`（L187-188）——每元素占 queueBufNum×2B（队列）+ 12B（cast 缓冲）
- **等价每元素 UB 开销**：fp16 双缓冲 = 6×2B+3×4B = 24B；fp16 单缓冲 = 3×2B+12B = 18B；fp32 双缓冲 = 24B；fp32 单缓冲 = 12B。

## 4.1.3 tilingdata / tilingkey

| 字段 | 类型 | 语义 | host 计算式 | 锚点 |
|---|---|---|---|---|
| totalNum | int64 | 打平总元素数 | x 的 ShapeSize | tiling_data.h:22；tiling L92 |
| blockFactor | int64 | 每核处理元素数（32B 对齐） | CeilAlign(CeilDiv(totalNum,coreNum), 32B/typeSize) | tiling_data.h:23；tiling L169 |
| ubFactor | int64 | 单循环 UB 处理元素数 | 见 §4.1.2 | tiling_data.h:24；tiling L184-191 |
| negativeSlope | float | 属性透传 | attrs->GetFloat(0)，缺省 0.2 | tiling_data.h:25；tiling L117-118 |
| scale | float | 属性透传 | attrs->GetFloat(1)，缺省 1.414213562373 | tiling_data.h:26；tiling L119-120 |

- **tilingkey 只编码 BUFFER_MODE**（uint8，{0,1}：0=单缓冲、1=双缓冲）（tiling_key.h:26-28；tiling L201 `ASCENDC_TPL_SEL_PARAM(context, useDoubleBuffer)`）。dtype 不进 key——由 def DataType 列表驱动 DTYPE_X 宏（kernel 入口注释 fused_bias_leaky_relu.cpp:17-19）。
- kernel 模板实例化：`FusedBiasLeakyRelu<DTYPE_X, BUFFER_MODE>`，`BUFFER_NUM = BUFFER_MODE ? 2 : 1`（h:39）。

## 5 片上内存（h:55-61, 100-108）

| buffer | 位置 | 容量表达式 | buffer_num | 锚点 |
|---|---|---|---|---|
| inputQueueX | VECIN | ubFactor × sizeof(T) | BUFFER_NUM(1/2) | h:56, 101 |
| inputQueueBias | VECIN | ubFactor × sizeof(T) | BUFFER_NUM | h:57, 102 |
| outputQueueY | VECOUT | ubFactor × sizeof(T) | BUFFER_NUM | h:58, 103 |
| xFp32Buf_ | VECCALC(TBuf) | ubFactor × 4B | 1（仅 fp16） | h:59, 105 |
| biasFp32Buf_ | VECCALC(TBuf) | ubFactor × 4B | 1（仅 fp16） | h:60, 106 |
| yFp32Buf_ | VECCALC(TBuf) | ubFactor × 4B | 1（仅 fp16） | h:61, 107 |

- UB 总量（双缓冲最坏）：fp16 = 24B×ubFactor，fp32 = 24B×ubFactor——host ubFactor 公式即按「打满 ubSize」反解（tiling L187-190），kernel 分配与 host 预算严格一致。
- 双缓冲：仅三条队列（TQue）享受；cast TBuf 恒单份（tiling L186 注释）。

## 6 流水编排（h:111-188）

- 结构：经典三段式 `for i in [0,loopCount_): CopyIn(i,ubFactor) → Compute(ubFactor) → CopyOut(i,ubFactor)`，尾块一次（h:117-126）。
- CopyIn：两路 `DataCopyPad`（blockCount=1、blockLen=currentNum*sizeof(T)、stride=0；padParams 不补零）（h:134-143）——非对齐尾块由 DataCopyPad 硬件处理。
- Compute：见 §1.1 三步链；fp16 路径首尾各多一次 Cast（h:153-170）。
- CopyOut：单路 `DataCopyPad` UB→GM（h:180-187）。
- 同步：TQue EnQue/DeQue 自动事件同步，无手工 SetFlag/WaitFlag；双缓冲掩盖 MTE2/MTE3 与 VEC 的间隙。
- 特殊指令：无 Reg gather/scatter、无 NZ、无 CV 交互——纯向量直通。

## 精度策略

- **fp16 路径唯一提升点**：输入 Cast(fp16→fp32, CAST_NONE 精确) → fp32 三步计算 → Cast(fp32→fp16, CAST_ROUND=RTNE) 回写（h:158-165）。全链路单次舍入（末段 RTNE）。
- fp32 路径原生精度，无 cast。
- 属性标量以 fp32 参与 Muls/LeakyRelu，无精度损失。

## 测试设计

- kernel UT（test_fused_bias_leaky_relu.cpp）：totalNum ∈ {8（极小/非对齐）, 1024, 8192（tilingKey=1 双缓冲）, 65536（大 shape 双缓冲）}；属性变体 {ns=0.0 & scale=1.0（L101-108）, scale=1.0（L110-117）, ns=0.5（L119-126）}；UB 模拟 ubFactor=min(totalNum, 4096)、blockDim=1（L45-54）。
- host tiling UT：x/bias 同 shape 注入 + 期望状态断言（test_..._tiling.cpp:41-118）。
- GEIR 示例：shape (2,4)，默认属性（examples L147-158）。
- **标杆**：源码未附标杆脚本；benchmark 库无本算子 → P3 卡片 §9 自建 torch golden。

## P1 质量门自查

全部章节有锚点；公式与 Compute 已逐行核对一致；两处文档/实现偏差（proto dtype、README 产品表）已记录并以 def.cpp 为准。
