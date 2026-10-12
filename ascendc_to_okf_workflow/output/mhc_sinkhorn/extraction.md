# P1 语义抽取 — mhc_sinkhorn（事实表，不含转化）

> 规则：每论断带 `file:line` 锚点；文档与代码冲突以代码为准并标注偏差。
> 路径前缀：`src/mhc/mhc_sinkhorn/` 记为 `$OP`，`src/mhc/mhc_sinkhorn_common/` 记为 `$COMMON`。

## 1.1 功能分析（算什么）

- **功能**：mHC（超连接）架构核心算子，用 Sinkhorn-Knopp 迭代把初始混合矩阵投影到双随机矩阵流形（行和=列和=1），输出供 MhcPost 使用。锚点：`$OP/README.md:16`
- **文档公式**（README L20-28）：
  $$M^{(k+1)} = M^{(k)} \oslash (1_n (M^{(k)})^T 1_n),\quad M^{(k+2)} = M^{(k+1)} \oslash ((M^{(k+1)} 1_n) 1_n^T)$$
  即交替做列归一、行归一。
- **⚠️ 文档与实现偏差**：kernel 实际**第 0 次迭代不是直接归一，而是先做逐行 softmax**（max 减去 → exp → 除行和），之后才进入「行归一/列归一」交替循环。锚点：`$OP/op_kernel/arch35/mhc_sinkhorn_arch35.h:213-235`（CalcSoftmax 内 `ReduceMaxWithDataBlock`→`Sub`→`Exp`→`ReduceSumWithDataBlock`→`Div`）。
- **实现公式**（以代码为准，每个 n×n 矩阵独立处理）：
  - 第 0 轮：`M⁰ = rowsoftmax(X)`；列和 `S_c = Σ_row(M⁰ + eps)`；`M¹ = M⁰ / (S_c + eps)`（CalcColNorm 中先 `Adds(sum,eps)` 再 `Div`，arch35.h:288-299；softmax 阶段已把 `M⁰+eps` 累进 sumoutReg，arch35.h:233-234）
  - 第 k 轮（k=1..num_iters-1）：行归一 `M = M / (rowsum(M)+eps)`（CalcRowNorm，arch35.h:358-373），再列归一 `M = M / (colsum(M)+eps)`（CalcColNorm<ADDS_EPS=false>，arch35.h:296-299）
  - 每步分母都加 `eps` 防除零。
- **执行序**（Process，arch35.h:472-551）：`CalcSoftmax → CalcColNorm<true> → for iter in 1..num_iters-1 { CalcRowNorm → CalcColNorm<false> } → ScatterOutFromNorm → CopyOut`。共 num_iters 次列归一 + (num_iters-1) 次行归一 + 1 次 softmax 初始化。
- kernel 类型：**AIV-only**，全程 Vector/Reg，无 matmul。锚点：`$OP/op_kernel/mhc_sinkhorn_apt.cpp:26`

## 2 原型设计（接口）

来源：`$OP/op_host/mhc_sinkhorn_def.cpp:23-49`（⚠️ 与 README 参数名不一致，以 def.cpp 为准）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| h_res | 输入 | 必选（REQUIRED） | FLOAT32 | ND | (T,n,n) 或 (B,S,n,n) | 初始混合矩阵 X（每 T 项一个 n×n 矩阵） |
| y | 输出 | 必选 | FLOAT32 | ND | 同输入 | 双随机化结果 |
| norm_out | 输出 | 必选（IR 层）/ 可空（aclnn 层） | FLOAT32 | ND | out_flag=1: 1维，2·num_iters·T·n·8；out_flag=0: {1} | 每轮归一后的中间矩阵（调试输出） |
| sum_out | 输出 | 必选（IR 层）/ 可空（aclnn 层） | FLOAT32 | ND | out_flag=1: 1维，2·num_iters·T·8；out_flag=0: {1} | 每轮行/列和（调试输出） |
| eps | 属性 | 可选，默认 1e-6 | float | — | 标量 | 分母保护项 |
| num_iters | 属性 | 可选，默认 20 | int | — | [1,100] | Sinkhorn 迭代轮数 |
| out_flag | 属性 | 可选，默认 0 | int | — | {0,1} | 是否输出 norm_out/sum_out 调试信息 |

aclnn 两段式：`aclnnMhcSinkhornGetWorkspaceSize(x, eps, numIters, output, normOut, sumOut, ...)`（`$OP/op_host/op_api/aclnn_mhc_sinkhorn.h:30-32`）；normOut/sumOut 可传空指针（docs L151）。out_flag 不进 aclnn 签名，由 L0 层 `MhcSinkhorn(x, outFlag, eps, numIters, ...)` 传递（`op_api/mhc_sinkhorn.cpp:45-48`）。

## 2.1 原型约束

- x 维度数仅 3 或 4（tiling.cpp:162-166；infershape.cpp:73-78）
- 最后两维必须相等（方阵，tiling.cpp:176-180）且 **n ∈ {4, 6, 8}**（tiling.cpp:181-185）
- 全 dtype 仅 FLOAT32，y/norm_out/sum_out dtype 必须等于 x（tiling.cpp:107-147）
- out_flag=1 时 norm_out 元素数 = `2*num_iters*T*n*8`、sum_out = `2*num_iters*T*8`，其中 T=dim0（3 维）或 dim0*dim1（4 维），8 = n 对齐值（tiling.cpp:216-238；docs/aclnnMhcSinkhorn.md L151）
- num_iters ∈ [1,100]（tiling.cpp:85-89）；out_flag ∈ {0,1}（tiling.cpp:93-97）
- 输入非空（tiling.cpp:157-159）；y shape 必须等于 x shape（tiling.cpp:202-214）
- README 补充（L81-82）：init_matrix 需二维非负；建议 max_iter 50~200（与 tiling 上限 100 冲突，**以 tiling 校验 [1,100] 为准**）、eps 1e-6~1e-4
- infershape 支持 unknownRank(-2)/unknownDim(-1)（infershape.cpp:66-71, 86-113）

## 3 规格约束

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | 仅 Ascend 950PR/950DT | README L7；def.cpp L59 仅注册 `ascend950` |
| dtype | FLOAT32 唯一 | def.cpp L25；tiling.cpp L53 |
| format | ND | def.cpp L26 |
| 确定性计算 | 无 atomic/跨核规约，天然确定（每核独立处理若干完整矩阵） | Process 结构 arch35.h:465-552 |
| 交付形式 | aclnn 调用（examples/）+ GEIR 图模式（op_graph/mhc_sinkhorn_proto.h） | |
| tilingkey | `default=0`（simplified_key.ini）；实际 key = out_flag（tiling.cpp:308-315） | |

## 4.1.1 分核策略

来源：`$COMMON/op_host/arch35/mhc_sinkhorn_base_tiling.cpp:122-145`（ForwardSplitCore）

- 切分轴：**仅 T 轴**（T 个 n×n 矩阵互不依赖，天然并行）。n 轴不切（n≤8，单寄存器可容）
- `tNormCore = CeilDiv(T, totalCoreNumAIV)`；`usedCoreNum = CeilDiv(T, tNormCore)`；尾核 `tTailCore = T - tNormCore*(usedCoreNum-1)`（L125-127）
- `BaseSplitCores`（L33-46）同时计算前向/后向两套切分，**取 tUbFactor 更小者**（前后向共用 tiling 结构，mhc_sinkhorn 用 forward 结果）
- host 侧 `SetBlockDim(usedCoreNum_)`（tiling.cpp:327）

## 4.1.2 单核内切分

- 每核处理 `tNormCore` 个矩阵，按 `tUbFactor` 个一组循环（`tNormCoreLoop = CeilDiv(tNormCore, tUbFactor)`，尾循环 `tUbFactorTail`；尾核同理 `tTailCoreLoop`/`tUbTailTail`）（base_tiling.cpp:140-144）
- **tUbFactor 决策**（base_tiling.cpp:131-138）：先试全载 `tUbFactor = tNormCore`，若 UB 占用超 `(ubSize - 64B - 768B)/2`（预留 maskBuffer 64B + mid/sumBuffer 各 256B，双缓冲减半），则 `tUbFactor = FloorAlign(availableUb / onePieceSize, 8)`，保底 8
- 单片占用（ForwardCalOccupySize L100-120，fp32）：`inQue + outQue + normQue + sumColQue (+ sumRowQue 仅 out_flag=1)`，其中 inQue=outQue=align(t·n·n)，normQue=align(t·n·nAlign)，sumQue=align(t·nAlign)，nAlign=CeilAlign(n, 8)

## 4.1.3 tilingdata & tilingkey

tilingdata（struct.h:18-31，host 侧填充 tiling.cpp:287-301）：

| 字段 | 语义 | host 计算式 |
|---|---|---|
| eps | 分母保护 | 属性透传 |
| num_iters | 迭代轮数 | 属性透传，[1,100] |
| out_flag | 调试输出开关 | 属性透传，{0,1} |
| n | 矩阵维度 | shape 推得，{4,6,8} |
| usedCoreNum | 实际用核数 | CeilDiv(T, tNormCore) |
| tNormCore | 正常核矩阵数 | CeilDiv(T, AIV核数) |
| tNormCoreLoop | 正常核循环次数 | CeilDiv(tNormCore, tUbFactor) |
| tUbFactor | 单循环矩阵数 | UB 预算反推，8 对齐 |
| tUbFactorTail | 正常核尾循环矩阵数 | tNormCore - (loop-1)*tUbFactor |
| tTailCoreLoop | 尾核循环次数 | CeilDiv(tTailCore, tUbFactor) |
| tUbTailTail | 尾核尾循环矩阵数 | tTailCore - (loop-1)*tUbFactor |

tilingkey：`GET_TPL_TILING_KEY(out_flag)`（tiling.cpp:308-315），`TEMPLATE_MODE ∈ {0,1}`（tiling_key.h:20-25）→ apt.cpp 两个模板实例 `MhcSinkhornSimd<float, false/true>`（apt.cpp:30-40）

## 5 片上内存资源（Init，arch35.h:102-115）

| buffer | 位置 | 容量 | double buffer |
|---|---|---|---|
| inputQue_ | VECIN | align(tUbFactor·n·n, 8)·4B | 是（DOUBLE_BUFFER=2） |
| outputQue_ | VECOUT | align(tUbFactor·n·n, 8)·4B | 是 |
| normOutQue_ | VECOUT | tUbFactor·n·nAlign·4B | 是 |
| sumColQue_ | VECOUT | tUbFactor·nAlign·4B | 是 |
| sumRowQue_ | VECOUT | out_flag=1: tUbFactor·nAlign·4B；=0: 仅 8·4B | 是 |
| maskBuffer_ | VECCALC | 64B（存放 n∈{4,6,8} 对应的 lane mask 常量） | — |
| midBuffer_ | VECCALC | 256B（规约结果中转） | — |
| sumBuffer_ | VECCALC | 256B（规约结果中转） | — |

mask 常量（arch35.h:29-31, 449-456）：n=4/6/8 → MASK_4/MASK_6/MASK_8，32bit mask 每 8 lane 一组选前 n 个。

## 6 流水编排与计算流程（kernel 方案）

### 循环结构（Process，arch35.h:465-552）

```
for loopIdx in 0..loop_-1:                    # tUbFactor 个矩阵一批
    handleNum = 尾循环 ? tUbFactorTail : tUbFactor
    CopyIn(loopIdx, handleNum)                # GM→UB 连续搬 tUbFactor 个 n×n（DataCopyPad）
    CalcSoftmax(handleNum)                    # 行 softmax → normOutQue（转置布局 [n, tUbFactor·nAlign]）
    [out_flag] 搬出 norm_out[2*0]
    SetFlag/WaitFlag(MTE3_V)                  # 等 CopyIn 的 MTE3 完成
    CalcColNorm<true>(handleNum)              # 列归一（原地更新 normOutQue）
    [out_flag] 搬出 norm_out[1], sum_out[1]
    for iter in 1..num_iters-1:
        CalcRowNorm(handleNum)                # 行归一（原地）→ sumRowQue
        [out_flag] 搬出 norm_out[2*iter], sum_out[2*iter]
        CalcColNorm<false>(handleNorm)        # 列归一（原地）
        [out_flag] 搬出 norm_out[2*iter+1], sum_out[2*iter+1]
    ScatterOutFromNorm(handleNum)             # 转置布局 scatter 回行主序 → outputQue
    CopyOut(loopIdx, handleNum)               # UB→GM
```

### 关键实现技巧（Reg 级）

1. **布局转换 gather/scatter**：GM 行主序 [t,n,n] ↔ UB 转置布局 [n, t·nAlign]（列主序视角）。rowIndexReg 用 Arange+Div+Mul 合成 gather 下标（arch35.h:195-202）；入口 `DataCopyGather`（L215）、出口 `DataCopyScatter`（L434）。目的：让同一列的 n 个元素在寄存器中连续，行/列规约都变连续访存。
2. **规约经 UB 中转**：`ReduceMaxWithDataBlock/ReduceSumWithDataBlock` 结果先 `Reg::DataCopy` 写 midBuffer/sumBuffer，加 `LocalMemBar` 后以 `DIST_E2B_B32` 广播读回，实现"规约值广播到全 lane"（arch35.h:216-229, 358-372）
3. **mask 三段式**：`UpdateMask(handleSize)`（尾块有效 lane）∧ `LoadAlign(maskAddr)`（n 有效位）→ 计算 mask；`UpdateMask(copyLen)` → 搬运 mask（arch35.h:206-210）
4. **双缓冲**：所有 Que buffer_num=2，CopyIn 与上一批 Compute 重叠；循环内 MTE3_V 事件同步（arch35.h:482-484, 502-507, 525-526）
5. **非对齐写回**：out_flag=1 时 sum_out 用 `StoreUnAlign` 按 handleNum 偏移写（arch35.h:362-367）

## 精度策略

- 全程 fp32（输入即 fp32，无精度提升/降级点）。eps 加分母防除零（4 处：softmax 累和、colnorm 分母、rownorm 分母×2）
- 无竞品对齐说明（源码未含）

## 测试设计

- UT（`test/ut/op_host/op_api/test_aclnn_mhc_sinkhorn.cpp`）：
  - 正常：{1,128,4,4}（L38-42）、{1024,6,6}（L58-62）
  - 异常：空 batch {1,0,6,6}（L78-82）、错误 dtype fp16（L100-104）、5 维（L121-124）
- 示例（`examples/test_aclnn_mhc_sinkhorn.cpp:311-324`）：x={1024,4,4}，num_iters=20，norm_out={40·4·8·1024}，sum_out={40·8·1024}
- **源码未提供标杆脚本**（torch 参考实现需在 P4 自建：rowsoftmax 初始化 + 交替行列归一，fp64 真值）

## P1 质量门

- 全部章节带锚点 ✅
- 偏差已标注 2 处：README 参数名 vs def.cpp；README max_iter 建议 50~200 vs tiling 校验上限 100 ✅
- 公式已与 Compute 逐步核对（softmax 初始化偏差为核心发现）✅
