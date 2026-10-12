# P1 语义抽取 — mhc_pre（事实表，不含转化）


> **2026-09-14 新流程复验**：事实表复用，锚点抽验 5/5 有效：
> ① 段布局 [pre|post|res]（vector_compute.h:100-128，pre=mm+0/post=mm+HEAD_NUM/res=mm+2·HEAD_NUM）✅
> ② invRms 序列 Muls(scaleMean)→Adds(normEps)→Sqrt→Div（vector_compute.h:429-433）✅
> ③ sigmoid Neg→Exp→Adds(1)→Div 且 hPre 恒加 hcEps（vector_compute.h:596-603）✅
> ④ hPost Muls(2) + 仅 NO_RESI 加 hcEps（vector_compute.h:640-643）✅
> ⑤ KERNEL_TYPE_MIX_AIC_1_2（apt.cpp:33）✅

> 规则：每论断带 `file:line` 锚点；文档与代码冲突以代码为准并标注偏差。
> 路径前缀：`src/mhc/mhc_pre/` 记为 `$OP`。

## 1.1 功能分析（算什么）

- **功能**：mHC 架构前置算子。一次调用完成 RMSNorm + 混合矩阵投影（matmul）+ 三路段后处理（sigmoid 门控）+ 输入重加权求和，产出 Attention/MLP 输入 hIn 与投影矩阵 hPost/hRes。锚点：`$OP/README.md:16`
- **公式**（README L22-46，与 kernel 逐步核对一致）：
  $$invRms=(mean(xFlat^2)+normEps)^{-1/2},\quad xGamma=xFlat\odot gammaFlat\ (\gamma\text{ 可缺省})$$
  $$hMix=xGamma@\phi^T,\quad w=hMix\odot invRms$$
  $$(pPre,pPost,pRes)=split(w,(n,n,n^2))\ (\alpha=[3]\text{ 时});\ (pPre,pPost)\ (\alpha=[2])$$
  $$hPre=\sigma(pPre\alpha_0+bias_0)+hcEps$$
  $$hPost=2\sigma(pPost\alpha_1+bias_1)\ \{\alpha=[3]\};\quad 2\sigma(\cdot)+hcEps\ \{\alpha=[2]\}$$
  $$hRes=pRes\alpha_2+bias_2\ \{\alpha=[3]\}$$
  $$hIn_d=\sum_{i=0}^{n-1}hPre_i\,x_{i,d}$$
- **段布局事实**：hMix 每行 matN 个元素按 `[pPre(n) | pPost(n) | pRes(n²)]` 顺序排布（vector_compute.h:107-125，pre 取 mm+0、post 取 mm+n、res 取 mm+2n）。alpha/bias 已广播展开为长度 matN 的向量（VFExpandAlpha，vector_compute.h:131-150：alpha[0]→n 份、alpha[1]→n 份、alpha[2]→n² 份）。
- **数值顺序事实**：
  - invRms 计算顺序：`Muls(scaleMean=1/nD) → Adds(normEps) → Sqrt → Div(1, ·)`（vector_compute.h:429-433），即 mean(x²) 先乘后加 eps。
  - hIn 累加在 fp32 寄存器完成，最终 `CAST_RINT` 转回 bf16/fp16（vector_compute.h:36-37, 720-722）。
  - FP32 matmul 模式 Cube 侧固定 `mmKStep=32` 小步累加以保累加顺序（cube_compute.h:246-247）；HF32 模式整 baseK 一次打入。
  - sigmoid 用 `1/(1+exp(-x))` 显式序列实现（Neg→Exp→Adds(1)→Div，vector_compute.h:596-603），hPost 再 `Muls(2)`（L640）。
- **kernel 类型**：CV 融合 MIX_AIC_1_2（apt.cpp:33）。950 走 Basic API（自管 L1/L0/Mmad/Fixpipe）；910b/910_93 走高阶 matmul API 两阶段（mhc_pre.cpp:44-89）。

## 2 原型设计（接口）

来源：`$OP/op_host/mhc_pre_def.cpp:23-97`

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | BF16/FP16 | ND | (B,S,n,D)/(T,n,D) | mHC 层输入 |
| phi | 输入 | 必选 | FP32 | ND | (2n, nD)/(n²+2n, nD)[/(n!+2n, nD)] | 投影矩阵 |
| alpha | 输入 | 必选 | FP32 | ND | (3)/(2) | 缩放参数；[3] 含残差支路 |
| bias | 输入 | 必选 | FP32 | ND | 同 phi 首维 | bias 参数 |
| gamma | 输入 | 可选 | FP32 | ND | (n, D) | RMSNorm 缩放因子 |
| hin | 输出 | 必选 | 同 x | ND | (B,S,D)/(T,D) | Attention/MLP 输入 |
| h_post | 输出 | 必选 | FP32 | ND | (B,S,n)/(T,n) | h_post 矩阵 |
| h_res | 输出 | 必选* | FP32 | ND | (B,S,n,n)/(T,n,n)（或 n! 形态）；alpha=[2] 时 shape=[0] 可空 | 未 sinkhorn 的 h_res |
| inv_rms | 输出 | 可选（out_flag=1 时必填） | FP32 | ND | (B,S)/(T) | 1/r |
| h_mix | 输出 | 可选（同上） | FP32 | ND | (…, matN) | matmul 原始结果 |
| h_pre | 输出 | 可选（同上） | FP32 | ND | (…, n) | sigmoid 后的 h_pre |

属性（def.cpp:94-97）：`out_flag` int 默认 0；`norm_eps` float 默认 1e-6；`hc_eps` float 默认 1e-6；`op_impl_mode` int 默认 0（0=Cube FP32，1=HF32，仅 950/V2）。

aclnn 两段式：`aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gammaOptional, normEps, hcEps, hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional, ...)`（docs/aclnnMhcPre.md L64-81）；V2 增加 opImplMode（docs/aclnnMhcPreV2.md）。

## 2.1 原型约束

- x rank 仅 3/4（infershape.cpp:112-114）；phi 必须 2 维（L109-111）；alpha 必须 1 维且长度 ∈ {2,3}（L116-125）
- **n ∈ {4,6,8}**（tiling.cpp:58, 480-490；README L172）
- **D ∈ [16,16384] 且 16 对齐**（tiling.cpp:63-64, 433-441, 492-496；README L173）
- phi 首维 = matN：alpha=[2] → 2n；alpha=[3] → n²+2n（arch35 强校验 tiling.cpp:500-511）。⚠️ **偏差**：infershape 还接受 n!+2n（infershape.cpp:146-163），但仅 910b/910_93 kernel 有 factorial 分支（mhc_pre.cpp:59-74），950 tiling 会拒绝 n! 形态。
- x dtype 仅 BF16/FP16，phi/alpha/bias/gamma 必须 FP32；hin dtype 必须等于 x（tiling.cpp:256-298）
- out_flag ∈ {0,1}，=1 时 inv_rms/h_mix/h_pre 三个可选输出必须非空（tiling.cpp:586-606）
- op_impl_mode ∈ {0,1}（tiling.cpp:629-636）；**A3/A2 平台只支持 0**（README L175）
- hasResi（alpha=[3]）时 h_res 输出必须非空（tiling.cpp:608-614）
- 硬件拓扑：AIV 核数必须 = 2×AIC 核数（tiling.cpp:575-579）
- 所有维度 > 0（CheckShapePositive，tiling.cpp:223-255）；输出 shape 必须与推导一致（L300-432）

## 3 规格约束

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | Ascend 950PR/950DT、Atlas A3、Atlas A2 | README L5-13；def.cpp L81-92 |
| dtype | x/hin: BF16,FP16；其余 FP32 | def.cpp L23-79 |
| format | ND | def.cpp 全表 |
| 确定性计算 | 支持 deterministic level 3：强制 SPLIT_BS 模式（K=1024 顺序部分和 + atomic 保序） | tiling.cpp:813-816；split_bs.h:198-208 |
| 交付形式 | aclnn V1/V2 + GEIR | examples/、docs/ |
| tilingkey | arch35: `GET_TPL_TILING_KEY(tilingMode, hasResi?0:1)`，tilingMode∈{0=BS,1=ND,2=MK}（tiling.cpp:914-917；tiling_key.h:21-31）；910b: {0,10,11}（membase_tiling.cpp:533-541） | |
| 系统 workspace | 固定 20MB + 用户区（tiling.cpp:65） | |

### 架构差异

- **ascend950（arch35）**：Basic API 手写 Cube（Nd2Nz→L1 双缓冲→L0 乒乓→Mmad→Fixpipe），三种 tiling 模式路由；op_impl_mode 支持 FP32/HF32。
- **910b/910_93**：高阶 matmul API（lib/matmul_intf），Stage1/Stage2 两 kernel 段；支持 n!+2n factorial；n=4、D≤100000 且 128 对齐（README L174-177）。
- 卡片以 arch35 为转化主线（目标平台 950）。

## 4.1.1 分核策略

来源：`$OP/op_host/op_tiling/arch35/mhc_pre_tiling.cpp`

**模式路由**（TilingProcess L811-833，优先级自上而下）：
1. `deterministic level == 3` → **SPLIT_BS**（L815-816）
2. `CanUseMk()`（L641-649：n∈{4,6,8} 且 totalLength ≤ 10240 且 totalLength ≤ activeAic×256 且 matK=nD ≥ 512）→ **SPLIT_M_K**
3. `totalLength ≤ 256` → **SPLIT_ND**（decode 场景）
4. 否则 → **SPLIT_BS**

**各模式分核**：
- **SPLIT_BS**（CalculateBsTiling L660-687）：usedCoreNum=AIC 核数；T 轴均分，`chunkTSize = CeilAlign(CeilDiv(totalLength, blockDim), 32)`，上限 128（L673-675）。K 轴核内循环（每片 256）。AIV0/AIV1 按 subBlock 对半分 M（split_bs.h:148-163）。
- **SPLIT_ND**（CalculateNdTiling L768-789；split_nd.h:117-129）：T 轴 chunkTSize=2 仅用于 AIV0 cast；**K 轴按全部 AIC 均分** `chunkNd = Ceil(nD, coreNum)`，每 AIC 算 [M, chunkNd]×[chunkNd, matN] partial 写 GM，AIV1 归并。
- **SPLIT_M_K**（CalculateMkTiling L689-766）：AIC 二维网格 mDim×kDim：mDim 由 totalLength/mL1Size(≤256) 决定、kDim=activeCore/mDim；单核 K=splitK（256 对齐，≥2048 时启用 1024 顺序部分和保序 L707-713）；**Part2 阶段 AIV 独立分核**：stage2RowsPerCore ≥2 行/AIV，stage2UsedAivNum（L732-737）。

## 4.1.2 单核内切分

- **BS**：K 循环步长 ND_LENGTH=256（vector_compute.h:171）；AIV0 每批 V0_BASE_T=16 行（L172）cast bf16→fp32、乘 gamma、Σx²，写 GM X staging（每 AIC 2 个 ping-pong 槽）；AIC L1 tile mL1Size=32/kL1Size=256；K 每累计满 1024 输出一次部分和，partialIndex>0 用 `SetAtomicAdd<float>` 保序累加进 hMix（split_bs.h:189-208）。AIV1 按 V1_BASE_T=16 行读 hMix 做后处理（split_bs.h:213-232）。
- **ND**：AIV0 ND_LENGTH=2048 一片 cast 全量 X→fp32 workspace + Σx²（split_nd.h:375-435；VFDoV0ProcessXInSingleReduce L309-358）；AIC 单核 K=chunkNd 再按 maxKL1 细分（L244-274）；AIV1 按 V1_BASE_T=8 行把 mmResBlockNum 个 partial 在 UB 内 Add 归并（L473-510）。
- **MK**：L1 tile mL1Size≤256、kL1Size≤1024（128 对齐，且须整除 1024 顺序段，tiling.cpp:715-723）；AIV 半核各处理 ⌈mReal/2⌉ 行（m_k_split.h:665-667）；AIV 预取下一片 X/gamma 与当前片计算重叠（L703-714）；Part2 每次归并行数由 80KB buffer 反推 GetReduceRows（≤32 行、8 对齐，L862-878）。
- **UB 预算硬校验**（CheckUbBufferSize，tiling.cpp:835-871）：BS/MK 固定池 64KB+96KB；ND 固定池 80KB×2+16KB×2+20KB；动态区 matN×4B×2 + invRms + gamma（256/2048 元素）。

## 4.1.3 tilingdata & tilingkey

tilingdata 24 字段（`$OP/op_host/op_tiling/arch35/mhc_pre_tiling.h:31-60`）：

| 字段 | 语义 | host 计算式 |
|---|---|---|
| coreNum | 实际 AIC 核数 | 模式相关（BS/ND=全部 AIC；MK=mDim×kDim 覆盖数） |
| totalLength / nD / fusionSize / N / D | 问题尺寸 | M=T、K=nD、N=matN=2n 或 n²+2n |
| normEps / hcEps / scaleMean | 标量 | scaleMean=1/nD（L663） |
| chunkTSize / v1ChunkDSize | T 块 / D 块 | 模式相关；v1ChunkDSize=5120 |
| outFlag / hasGamma / hasResi / implMode | 开关 | 属性/输入透传 |
| mL1Size / kUbSize / kL1Size / fusionAlign | L1/UB tile | BS: 32/256/256；MK: ≤256/=kL1/≤1024；fusionAlign=RoundUp(matN,8) |
| cubeBlockDimM / cubeBlockDimK / multCoreSplitKSize | MK 网格与 splitK | CalculateMkTiling L697-727 |
| mkWorkspaceMmOffset / RmsOffset / FinalOffset | workspace 分区偏移（512B 对齐） | L747-754 |
| mkUseGmStage | X staging 路径 | SelectMkStage L651-658：FP32 且 M≥512，或 N=8 且 M<1536 → 0(UB→L1 直写)，否则 1(GM ping-pong) |
| stage2UsedAivNum / stage2RowsPerCore | MK Part2 AIV 划分 | L732-737 |

tilingkey：`GET_TPL_TILING_KEY(tilingMode, hasResi?0:1)`（tiling.cpp:914-916）→ 6 个模板实例（tiling_key.h:27-50，TILING_MODE∈{BS,ND,MK} × HAS_RESI∈{0,1}）。

## 5 片上内存资源

**L1（AIC 侧，common.h:80-83）**：总 512KB，A/B 各 2 buffer × 128×256 fp32（=128KB/buffer）。

**L0（cube_compute.h:86-95）**：L0A/L0B 各 256×32 fp32 ×2（乒乓），L0C 256×128 fp32 ×2。

**UB（AIV 侧，vector_compute.h:177-180 + split_nd.h:71-74）**：

| buffer | BS/MK | ND | 双缓冲 |
|---|---|---|---|
| xInQueue_ | 80KB | 80KB | ×2 |
| outQueue_ | 20KB | 16KB | ×2 |
| tmpBuff_ | 40KB（hPre+hPost 段缓冲） | 20KB | — |
| invRmsOutQueue_ | ceil(rows/2)×4B 对齐 | 同 | 单 |
| gammaInQueue_ | 256×4B | 2048×4B | 单 |
| biasInQue_/alphaBuf_ | matN×4B 对齐 | 同 | 单 |

host 侧校验 fixed+dynamic ≤ ubSize（tiling.cpp:835-871）。

## 6 流水编排与计算流程（kernel 方案）

### SPLIT_BS 主循环（split_bs.h:54-115）

```
for curblock in coreIdx..tBlockNum step coreNum:      # T 块
  for offsetNd in 0..nD step 256:                     # K 片
    AIV: [vectorCount≥2 等 X_CONSUMED] → V0Prologue(cast+γ+Σx²→GM staging)
         → SetFlag X_READY                             # 双槽 ping-pong
    AIC: 等 X_READY → Nd2Nz L1 → Mmad(L0乒乓,K=32保序)
         → 每 K=1024 顺序段末 Fixpipe→hMix GM（后续段 SetAtomicAdd）
         → SetFlag X_CONSUMED
  AIC: SetFlag MM_READY
  AIV0: V0PostProcess → invRms=1/sqrt(mean+eps) → 写 invRms GM
  AIV1: 等 MM_READY → 按 16 行读 hMix → AIV1PostProcessTile
```

跨核协议：flag mode=4，X_READY=8/X_CONSUMED=9/MM_READY=10，subblock 偏移 +16（common.h:95-99）。

### SPLIT_ND（split_nd.h:186-225）

单趟：AIV0 cast 全量 X+Σx² →（V0toV0、V0toC 两道 flag）→ 全部 AIC 各算 K 段 partial→GM →（CtoC、CtoV1 barrier，空核也必须到场 L214-217）→ AIV1 UB 内 Add 归并 partial + 后处理。RMS 每行整 D 段和在单寄存器累加、每行只 Reduce 一次（L322-324）。

### SPLIT_M_K（apt.cpp:44-54 + m_k_split.h）

同一 launch 内两阶段：Part1（AIC cube partial [groupK,M,matN]→workspace + AIV cast/γ/Σx²→rms partial [kBlocks,M]→workspace）→ `pipe.Destroy()` 重建 TPipe（注释：两阶段 UB 布局不同，释放 stage1 队列让 stage2 用满 UB，apt.cpp:47-49）→ Part2（纯 AIV：读 partial 4 路展开顺序归并、算 invRms、AIV1PostProcessTile 全量后处理）。

X→L1 两路径（m_k_split.h:736-768）：GM staging（AIV 写 GM 槽 ping-pong，AIC Nd2Nz 读）或 UB→L1 直写（AIV 软件 ND2NZ `MhcPreMkVFTransND2NZ` L34-91 后 DataCopy 进 L1）。

### 后处理链（三模式共用 AIV1PostProcessTile，vector_compute.h:793-840）

`VFPostSegments`（w=hMix·invRms；三段 ×alpha +bias，L58-129）→ hRes copyout（HAS_RESI 时）→ `AIV1ProcessHPost`（2σ+条件 eps）→ `AIV1ProcessHPre`（σ+hcEps，out_flag=1 时写出）→ `AIV1ProcessHInBatched`（D 轴分块 v1ChunkDSize=5120，行批 2/4，fp32 累加 ΣhPre_i·x_i,d，CAST_RINT 回 b16，L728-791）。

## 精度策略

- x bf16/fp16 → 进 kernel 立即 cast fp32（Reg::Cast B16_TO_FP32，vector_compute.h:396）；全程 fp32 计算；仅 hin 输出 cast 回 b16（CAST_RINT，L36-37, 720-722）
- matmul 两种模式：FP32（impl_mode=0，Cube mmKStep=32 保序）/ HF32（impl_mode=1，仅 950，cube_compute.h:105-111, 246-247）
- splitK 多核归并保序：splitK≥2048 时按 1024 顺序段输出部分和（tiling.cpp:707-713；m_k_split.h:27-29）；Part2 归并用 4 路展开但**段内顺序固定**（m_k_split.h:335-384 注释 L392 "without changing sum order"）
- deterministic level 3 → 强制 BS 路径（tiling.cpp:813-816）
- 无竞品对齐说明（源码未含）

## 测试设计

- UT op_api（test_aclnn_mhc_pre.cpp:202-321）：正常 4D fp16 含可选输出；V2 impl_mode 合法/非法；缺失必选 tensor；非法 dtype/shape 十余例（x rank、n 值、phi rank/首维、alpha rank/长度、bias、gamma、hIn rank…）
- UT tiling（test_mhc_pre_tiling.cpp，1928 行）：三模式路由、tiling 字段校验
- UT infershape（test_mhc_pre_infershape.cpp，345 行）
- 示例（examples/test_aclnn_mhc_pre.cpp:277-281）：B=1,S=2048,n=4,D=2560（T=2048,K=10240,matN=24），fp16，out_flag=1 全输出
- **源码未提供标杆脚本**（P4 需自建 torch fp64/fp32 参考：rmsnorm→matmul→三段后处理→加权求和）

## P1 质量门

- 全部章节带锚点 ✅
- 偏差标注 1 处：infershape 接受 n!+2n 但 950 tiling 拒绝（factorial 仅 910b/910_93）✅
- 公式已与 kernel 逐步核对（段布局 [pre|post|res]、invRms 计算顺序、sigmoid+eps 细节、hIn fp32 累加）✅
