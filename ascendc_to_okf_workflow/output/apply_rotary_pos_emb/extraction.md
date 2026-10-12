# P1 语义抽取 — apply_rotary_pos_emb
> **2026-09-14 新流程复验**：本文为 2026-09-08 版抽查复用（新流程 P1 仍为纯源码事实搬运，
> 不掺后端论断，规程未变）。锚点抽验记录见 manifest.md 头注，4/4 有效。


> 只做事实搬运，不做 Triton 转化。每个论断带 `file:line` 锚点（相对 `src/ops-transformer/posembedding/apply_rotary_pos_emb/`）。

## 1.1 功能分析（算什么）

- **算子语义一句话**：对 query、key 两路执行旋转位置编码（RoPE）并原地更新——推理网络为提升性能将两路融合为一次调用（README.md:18）。
- **统一数学公式**（README.md:19-136）：`q_embed = query·cos + rotate(query)·sin`，`k_embed = key·cos + rotate(key)·sin`；rotate 按 rotary_mode 定义：
  - **half**：x₁=x[.., :D/2]，x₂=x[.., D/2:]，rotate(x)=cat(−x₂, x₁)（L23-44）→ 逐元素 out₁=x₁·c₁−x₂·s₁，out₂=x₂·c₂+x₁·s₂（与 HalfAlignVF 的 `Sub(cos·in, sin·halfIn)` / `Add(halfCos·halfIn, halfSin·in)` 逐步一致，common.h:111-119）。
  - **quarter**：四等分 x₁..x₄，rotate(x)=cat(−x₂, x₁, −x₄, x₃)（L57-94）→ out₁=x₁c₁−x₂s₁、out₂=x₂c₂+x₁s₂、out₃=x₃c₃−x₄s₃、out₄=x₄c₄+x₃s₃（与 QuarterAlignVF common.h:193-209 一致）。
  - **interleave**：奇偶对 rotate：(x[2i], x[2i+1])→(−x[2i+1], x[2i])（L107-115）→ out[2i]=xe·ce−xo·se，out[2i+1]=xo·co+xe·so（与 InterleaveModeVF 的 DeInterleave→Muls(odd,−1)→Interleave→cos·in+sin·rot 一致，common.h:270-280）。
- **partial rope**：cos/sin 末维 realDim ≤ D；realDim < D 时仅前 realDim 维旋转、剩余 [realDim, D) 直通（host 判定 `isPartialRope_ = (reald_ != d_)`，tiling_base_arch35.cpp:319；kernel 搬运双段化，common.h:781-877）。
- **语义分层**：
  - GEIR kernel 本体：上式；属性 layout/rotary_mode 是编译期分派维度（决定 tiling 模板与 kernel 分支），不进运行时数据流。
  - aclnn 接口层：`aclnnApplyRotaryPosEmb`（V1）= V2 锁 rotary_mode="half"（op_api/aclnn_apply_rotary_pos_emb.cpp:27-29）；`aclnnApplyRotaryPosEmbV2` 暴露 (queryRef, keyRef, cos, sin, layout, rotaryMode)（v2.cpp:24-32）；inner 本体 extern 不在本目录（两 cpp:17-21 声明）。无额外参数消化（无 alpha 类参数）。
  - benchmark 层：**无精确 case**；近邻 RotaryMul（单输入、half/interleave）/InterleaveRope（单输入 interleave）为本算子语义子集，torch golden 须自建参数化三模式双路版本。
- **在仓未用资产**：common.h 另有 DeepSeekInterleaveModeVF（L331-452）与 BatchDeepSeekInterleaveModeVF（L698-779）、以及 RotaryPosEmbeddingMode 枚举（L54-59，含 DEEPSEEK_INTERLEAVE=3）——本算子 4 个 tilingkey 均不分派到它们（apt.cpp:35-59 仅 HALF/INTERLEAVE/QUARTER），属姊妹算子 RotaryPositionEmbedding 资产，不计入本算子语义。

## 2 原型设计（来源：op_host/apply_rotary_pos_emb_def.cpp）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 | 锚点 |
|---|---|---|---|---|---|---|---|
| query | 输入 | REQUIRED | FLOAT16 / FLOAT / BF16（310p/kirin 仅 FP16/FP32） | ND | 4 维（TND 时 3 维） | 待旋转 query | L27-32 |
| key | 输入 | REQUIRED | 同 query | ND | 与 query 仅 N 维可不同 | 待旋转 key | L33-38 |
| cos | 输入 | REQUIRED | 同 query | ND | N 维=1，B 维∈{qB,1}，D 维=realDim≤qD | 旋转角余弦 | L39-44 |
| sin | 输入 | REQUIRED | 同 query | ND | 与 cos 完全相同 | 旋转角正弦 | L45-50 |
| query | 输出 | REQUIRED | 同输入 | ND | 与输入 query 相同 | 原地语义输出 | L51 |
| key | 输出 | REQUIRED | 同输入 | ND | 与输入 key 相同 | 原地语义输出 | L52 |

- 四输入均 `.AutoContiguous()`（L32/38/44/50）。
- OpAICoreConfig：910b/910_93 → opFile `apply_rotary_pos_emb`（L55-63）；**ascend950 → opFile `apply_rotary_pos_emb_apt`**（L64-69，无 DynamicFormatFlag）；310p/kirinx90/kirin9030 dtype 裁剪为 FP16/FP32（L70-109）。

### 属性

| 属性 | 类型 | 默认值 | 语义 | 锚点 |
|---|---|---|---|---|
| layout | int 可选 | 1 | 1=BSND、2=SBND、3=BNSD、4=TND | def.cpp:53；枚举 tiling.h:139-144 |
| rotary_mode | string 可选 | "half" | "half"/"interleave"/"quarter" | def.cpp:54；校验 tiling_base_arch35.cpp:236-238 |

## 2.1 原型约束

- **shape 关系（arch35，CheckShapeRelation tiling_base_arch35.cpp:158-207）**：维度按 layout 重映射（L160-175）；cos==sin shape；q/k 输出 shape==输入；cos/sin N 维==1；q/k/cos/sin 的 S 维一致；cosD ≤ qD 且 kD ≤ qD；非 TND 时 kB==qB 且 cosB∈{qB,1}；所有维度为正（CheckShapeAllPositive L120-140，q/k 一空一非空时提示改用 rotaryPositionEmbedding 算子 L125-127）。
- **rotary 约束（CheckRotaryModeShapeRelation L142-156）**：D ≤ 1024（D_LIMIT L33）；half/interleave 要 D%2==0；quarter 要 D%4==0。
- **dtype**：q/k/cos/sin/两输出全同型，全集 {F32, BF16, F16}（SUPPORT_DTYPE L36，CheckDtypeAndAttr L233-259）。
- **infershape**：输出 = 输入与 cos/sin 广播对齐（BroadcastShape，infershape.cpp:116-125, 149-176）；BroadcastDim 放行 dim1>dim2 以包容 partial rope（L52-54）。
- **legacy 额外约束（tiling.cpp CheckParams L179-219）**：layout 仅 BSND(1)/TND(4)（L188）；**cosD 与 qD 均必须 ∈{64,128}**（L195-200）；cosN==1（L201）；qkcNum ≤ 255（kernel Mul repeatTimes uint8 限制，L306-308）；rotary_mode 实际仅 half（legacy kernel 无模式分支）。
- **文档/实现偏差**：docs V2 称支持空 tensor，但 arch35 tiling CheckShapeAllPositive 拒绝非正维度——以实现为准。
- **拦截校验位置**：全部在 tiling host 侧（OP_CHECK 系列）；kernel 内无校验。

## 3 规格约束

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | ascend950（arch35 主线）、ascend910b/910_93（legacy）、ascend310p/kirinx90/kirin9030（legacy，dtype 裁剪）；README 产品表另列 A2/A3 训练/推理（950PR/950DT √、Atlas 推理系列 √ 无 BF16、训练系列 ×） | def.cpp:62-109；README.md:5-14,149 |
| dtype 全集 | arch35/910b/910_93：FP16/FP32/BF16；310p/kirin：FP16/FP32 | def.cpp:20-21 vs 76-106 |
| format | ND（AutoContiguous） | def.cpp:27-50 |
| 确定性计算 | 是（纯向量 elementwise，无规约/atomic；workspace 仅 16MB 系统保留，不参与数据流） | tiling_aba_and_ba L26/251-255 等 |
| 交付形式 | aclnn V1/V2 + GEIR（OP_ADD） | def.cpp:113；op_api/；examples/ |
| 空 tensor | legacy 不支持（README L165）；arch35 实现拒绝（tiling_base_arch35.cpp:120-140）；V2 文档声称支持——**以实现为准** | 见 §2.1 |

### 架构差异（arch35 vs legacy，详见卡片 §3 同名节）

- legacy：layout 仅 BSND/TND、D∈{64,128}、rotary 仅 half、BF16 走 AB_CAST+fp32 计算、key{1,3,4}、依赖 workspace 与 16MB 保留。
- arch35：layout 全 4 种、D≤1024、三模式、key{20010,20011,20020,20030}、Regbase tiling 模板体系（优先级 10000→40000 顺次尝试，小者优先）。

## 4.1.1 分核策略（arch35 三模板 + legacy）

**模板选择顺序**（REGISTER_OPS_TILING_TEMPLATE 优先级数字小者先尝试，IsCapable 不满足返回 GRAPH_PARAM_INVALID 落下一个，tiling_templates_registry.h:468-470 + tiling_base.h:99-110）：
1. AB(10000)：BSND cosb=B / SBND / TND；2. ABA/BA(20000)：仅 BNSD；3. BAB(30000)：BSND cosb=1；4. legacy(40000)：兜底（非 regbase 平台实际走这里）。

- **AB（tiling_ab_arch35.cpp:53-88）**：BS 打平成一维（SBND+cosb==1 时 b 折进 n：bs=s_、qn_=b_*qn_、kn_=b_*kn_，L56-60）；`blockFactor=CeilDiv(bs, coreNum)`、`blockNum=CeilDiv(bs, blockFactor)`、`blockTail=bs−(blockNum−1)×blockFactor`（L69-71）；SetBlockDim(blockNum)（L112）。
- **ABA/BA（tiling_aba_and_ba_arch35.cpp:80-135）**：B 整除核数→只切 B（blockNumB=coreNum）；S 整除→只切 S；否则构造「B 优先」与「S 优先」两种混合切分，取 usedCoreNum 大者（L101-133）；SetBlockDim(blockNumB×blockNumS)（L262）。kernel 侧核坐标：`bIdx=blockIdx%blockNumB`、`sIdx=blockIdx/blockNumB`，尾核取余（aba_and_ba.h:151-166）。
- **BAB（tiling_bab_arch35.cpp:105-149）**：同样双方案比选，但判据乘上 ubFactorS（`usedCoreNum×min(ubFactorS, blockFactorS)` 大者胜，L131-147）；SetBlockDim(usedCoreNum)（L233）。
- **legacy（tiling.cpp:279-338）**：ab=B×S（TND 时 B=1）总 batch；`preCoreBatch=ceil(ab/coreNum)`、`useCoreNum=ceil(ab/preCoreBatch)`、尾核 lastCoreBatch（L290-293）；preCoreBatch==1 且 UB 够 → 小 shape 路径 key=1（L326-332）。

## 4.1.2 单核内切分（UB 因子）

- **公共量**：`dAlign = CeilAlign(realDim/dSplitCoef, 32B/sizeof(T)) × dSplitCoef`（tiling_ab L68；aba_and_ba.h:137；bab.h:90）——旋转维按 32B 对齐后再乘回切分系数。
- **AB**：UB 对半再双缓冲（`qkUbSize=ubSize/2/2`，L75）；baseBlockInUb=FloorAlign(qkUbSize,32B)/dAlign字节（L76）；**N 轴（QN+KN 合并）切 ubLoop/ubFactor/ubTail**（L79-82），ubFactor≤4095/dSplitCoef（DataCopy blockCount 上限，L81）；ubFactorBS=FloorDiv(baseBlockInUb, totalN+1)（L83-85）。
- **ABA/BA**：UB 预算注释「4×(b·n+1)×s×dAlign」（L143-145，4=qin/qout/cos/sin 双缓冲）；按 numOfDAvailable 依次解 ubFactorS→ubFactorQN/KN（cosb==1 时少占一份 cos 空间，L164-177）→ubFactorB（L182）；ubFactorS≤4095/coef（L154）。
- **BAB**：canLoadDNum=ubSize/dAlign，÷4（3 入 1 出双缓冲，L156-161）；ubFactorQN/KN 各按 n 切（≤4095/coef，L162-177）；ubFactorS=FloorDiv(canLoadDNum, max(qn,kn)+1)（L178-179）。
- **legacy**：UB 一次装 preCBatchB=times+shengMte 个 batch（times=UB/单 batch 开销，shengMte=剩余空间多塞的 batch 数；shengMte>0 且非 cast → key=3 聚集搬出，否则 key=4 一一对应，tiling.cpp:221-243）。

## 4.1.3 tilingdata / tilingkey

**arch35 RegbaseTilingData**（tiling.h:70-96，key 20010/20011/20020 共用）：

| 字段组 | 字段 | 语义 |
|---|---|---|
| 形状 | B, CosB, S, D, QN, KN | layout 归一化后的六元组（SetBsnd 抽取） |
| 分核 | blockNumB, blockFactorB, blockNumS, blockFactorS | B/S 二维分核（AB 模板不用此结构） |
| UB-S | ubLoopNumS, ubFactorS, ubTailFactorS | S 轴核内切分 |
| UB-B | ubLoopNumB, ubFactorB, ubTailFactorB | B 轴核内切分 |
| UB-N | ubLoopNumQN/KN, ubFactorQN/KN, ubTailFactorQN/KN | Q/K 各自 N 轴切分 |
| 模式 | rotaryMode, realDim, isPartialRope | 1/2/3；旋转实维；realDim≠D |

**arch35 RegbaseABTilingData**（tiling.h:102-121，key 20030 专用）：B/CosB/S/D/QN/KN + dAlign/dSplitCoef + **blockNumBS/blockFactorBS/blockTailBS**（BS 打平分核）+ ubFactorBS + **ubLoopN/ubFactorN/ubTailN**（QN+KN 合并轴）+ rotaryMode/realDim/isPartialRope。

**tilingkey 一览**：

| key | 模板 | kernel 类（apt.cpp 锚点） | 触发条件 |
|---|---|---|---|
| 20010 | ABA | ABAAndBA<T,false>（L48-53） | BNSD 且 cosb=B |
| 20011 | BA | ABAAndBA<T,true>（L54-59） | BNSD 且 cosb==1（cos/sin B 维广播，IsBBoardcast=true → cos/sin 每 S 段只载一次，aba_and_ba.h:185-196） |
| 20020 | BAB | BAB<T>（L42-47） | BSND 且 cosb==1 |
| 20030 | AB | AB<T>（L35-41） | BSND cosb=B / SBND / TND |
| 1 / 3 / 4 | legacy | ARPESmall / ARPEComputeAB / ARPEComputeABCast（apply_rotary_pos_emb.cpp:36-47） | 非 regbase 平台；小 shape→1；能多 batch 聚集搬出→3；否则（含 BF16）→4 |

dtype 不进 tilingkey，由 def DataType 列表驱动 DTYPE_QUERY 宏（apt.cpp 模板形参）。

## 5 片上内存（arch35）

| kernel | buffer（InitBuffer 锚点） | 容量表达式 | 双缓冲 |
|---|---|---|---|
| ABAAndBA | qInQueue_/qOutQueue_（aba_and_ba.h:139-141） | ubFactorB×ubFactorS×ubFactorN×dAlign×sizeof(T) | ×2 |
| ABAAndBA | cosInQueue_/sinInQueue_（L142-148） | BA: ubFactorS×dAlign×T；ABA: ubFactorB×ubFactorS×dAlign×T | ×2 |
| BAB | qkInQue_/qkOutQue_（bab.h:99-102） | ubFactorS×ubFactorN×dAlign×T | ×2 |
| BAB | cosInQue_/sinInQue_（L100-101） | ubFactorS×dAlign×T | ×2 |
| AB | qkInQueue_/outQueue_（ab.h:85-88） | dAlign×T×ubFactorBS×ubFactorN | ×2 |
| AB | cosInQueue_/sinInQueue_（L86-87） | ubFactorBS×dAlign×T | ×2 |

- host 侧 UB 预算公式与 kernel 分配严格互逆（AB: qkUbSize=ubSize/4；BAB: 4 路均分；ABA/BA: 4×(bn+1)×s×dAlign）。
- workspace：arch35 三模板均保留 16MB 系统 workspace 但 kernel 数据流不使用（tiling L26/L24/L21 常量 + apt.cpp 未读取）；legacy 必须有 workspace（apply_rotary_pos_emb.cpp:25-32 空则 return）。

## 6 流水编排

**ABA/BA**（aba_and_ba.h）：三重嵌套——Process 沿 S 轴 ubFactorS 步进（L168-178）→ ProcessInSLoop 沿 B 轴（**BA 广播时 cos/sin 在 S 段载一次、整个 B 循环复用**，L185-196；ABA 每 B 段重载，L197-211）→ ProcessInSBLoop 先 QN 全循环后 KN 全循环（L214-234，cos/sin 复用）→ ProcessInSBNLoop=CopyIn→Compute→CopyOut（L236-244）。CopyInQOrK 用 `LoopModeParams`+`SetLoopModePara` 实现 BNSD 三维跨步 DMA（loop2=B、loop1=N、内层 S×D 连续块，L284-291）；partial rope 按模式走 CopyInHalf/Quarter/InterleaveMode 双段/四段搬运（L301-308）。

**BAB**（bab.h）：Process=B 循环×S 循环（L120-133）→ ProcessNLoop 先载本 S 段 cos/sin（L139-149）→ ProcessQN 全循环再 ProcessKN（L156-251），每 N 块 CopyIn→Compute→CopyOut。

**AB**（ab.h）：Process 沿 BS 打平轴 ubFactorBS 步进，每段先载 cos/sin（L97-112）；currBSNum>1 时 ProcessQKLoop 整段（BS×N 一次性，先 Q 后 K，L113-115）；currBSNum==1 时沿 N（QN+KN 合并轴）ubFactorN 步进 ProcessLoop（L116-120），**ProcessLoop 处理 N 合并轴跨 Q/K 边界的块**（ubStart<QN 但 ubStart+nCount>QN 时先 copy Q 尾再拼 K 头，L142-161）。

**Compute**（三 kernel 同构）：按 rotaryMode 分派 Batch/Half/Quarter/InterleaveAlignVF——全部在 `MicroAPI::RegTensor<float>` 域计算（common.h 各 VF），`ops::LoadOne/TwoTensorForDtypeT` 完成 UB→Reg 的 dtype 转换，`MicroAPI::UpdateMask<float>` 处理非 VL 整数倍尾。Interleave 用 `MicroAPI::DeInterleave/Interleave` 做奇偶解/重交织（common.h:272-274）。VReg 宽度：arch35 取 256B（GetVRegSize L24-31）→ VL_FLOAT32_SIZE=64 元素/寄存器。

**同步**：TQue AllocTensor/EnQue/DeQue/FreeTensor 自动事件同步，无手工 SetFlag/WaitFlag。

## 精度策略

- **arch35 全模式 fp32 计算域**：RegTensor 一律 float（common.h:84-92 等所有 VF 寄存器声明）；fp16/bf16 输入在 Load 辅助函数内转 fp32，Store 时转回原 dtype——单次末段舍入。
- **legacy**：BF16 实例化 `<DTYPE_QUERY, float>`（apply_rotary_pos_emb.cpp:52-62），ComputeBF16 内 Cast(CAST_NONE) 提升、计算、Cast(CAST_ROUND=RTNE) 回写（compute_ab_cast.h:250-284）。
- 无累加、无规约，数值误差仅来自输入输出两次 dtype 转换。

## 测试设计

- examples：test_aclnn_apply_rotary_pos_emb.cpp（V1）/test_aclnn_apply_rotary_pos_emb_v2.cpp（V2）。
- tests/ut/{op_kernel,op_host,op_graph}、tests/st/aclnnApplyRotaryPosEmb 存在（本阶段未逐行抽取，P4 前补锚点）。
- **标杆**：源码无标杆脚本，benchmark 库无精确 case → 自建 torch golden：`out = x*cos + rotate(x)*sin`（三模式参数化，fp32 计算域），q/k 分别套用。

## P1 质量门自查

全部章节有锚点；三模式公式已逐行核对 VF 实现一致；文档/实现偏差 2 处（V2 空 tensor、proto 无）已记录；DeepSeek VF 确认为在仓未用资产并排除出本算子语义。
