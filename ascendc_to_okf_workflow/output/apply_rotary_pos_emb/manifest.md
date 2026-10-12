# P0 材料清单 — apply_rotary_pos_emb
> **2026-09-14 新流程复验**：本清单为 2026-09-08 版抽查复用（新流程 P0 不查外部库，规程未变）。
> 抽验 4 锚点全部有效：common.h:111-119（Half VF 配对式 ✓）、tiling_ab_arch35.cpp:68-71
> （dAlign/分核式 ✓）、aba_and_ba.h:185-196（BA cos/sin 载一次 ✓）、apt.cpp:31（AIC return ✓）。
> 增补 P4 回流锚点：tiling UT 950 期望 key（tests/ut/op_host 期望 20030/20010、UB mock=196608B
> ——注意此为 UT mock 值，非硬件规格；950 UB 规格以 backend_facts/hardware_platforms.md
> 的 248KB 可用为准）、ST 300 case（V1 half-only、D∈{64,128}、fp16_error=2⁻⁸、填充 [-1,1]）、
> 图级原地语义证据（fusion_pass tensormove）。


> 生成时间：2026-09-08｜来源目录：`src/ops-transformer/posembedding/apply_rotary_pos_emb/`（cann/ops-transformer @ d28d8f99a8858b71caa27a535da32bb5ba68ce2a）
> **kernel 类型判定：AIV-only 纯向量**。arch35 入口 `apply_rotary_pos_emb_apt.cpp:31` `if (g_coreType == AIC) return;`；legacy 入口 `apply_rotary_pos_emb.cpp:24` `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`。全部计算为 Vector/MicroAPI 指令（Mul/Add/Sub/Muls + DeInterleave/Interleave），无 Cube/Mmad、无规约、无 atomic。
> **非壳算子**：4 个 arch35 kernel 类 + 3 个 legacy kernel 类 + 5 个 tiling 类全部本体在仓内（op_kernel/ 约 3200 行，op_host/ 约 1900 行），无 base 库依赖。
> **aclnn 双层入口存在**：`aclnnApplyRotaryPosEmb`（V1，rotary_mode 硬编码 "half"，op_api/aclnn_apply_rotary_pos_emb.cpp:27-29）与 `aclnnApplyRotaryPosEmbV2`（暴露 rotaryMode 字符串，v2.cpp:24-32），均转发 `aclnnInnerApplyRotaryPosEmb*`（本体不在本目录，extern 声明）。
> **架构覆盖：arch35（ascend950，Regbase 新 tiling 模板体系）为主线 + legacy（910b/910_93/310p/kirin，老 tiling 类）一套并存**。def.cpp:62-69 分别注册 `apply_rotary_pos_emb`（legacy opFile）与 `apply_rotary_pos_emb_apt`（arch35 opFile）；310p/kirinx90/kirin9030 仅 FP16/FP32（def.cpp:70-109）。
> **复杂度预判：高价值算子，不触发 v1.2 低价值跳过**——3 种旋转模式（half/interleave/quarter，数学结构各异）× 4 种 layout（BSND/SBND/BNSD/TND）× cos/sin 批量广播两态（CosB∈{B,1}）× partial rope（realDim<D 直通尾段）× 4+3 条 tiling 路径；kernel 含 MicroAPI Reg 级 DeInterleave/Interleave 与双半场 strided load 等硬件特化手段（❌/⚠️ 可达性判定是卡片核心增量）；公式虽在基座先验内，但 layout 归一化、模式配对、partial 直通等决策点密集，对照组首轮易错。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/apply_rotary_pos_emb_def.cpp`（114 行） | ✅ | 4 输入 query/key/cos/sin REQUIRED+AutoContiguous L27-50；2 输出 query/key（与输入同名，原地语义）L51-52；Attr layout Int 默认 1（BSND）L53、rotary_mode String 默认 "half" L54；ascend910b/910_93→legacy opFile L55-63；**ascend950→regbase opFile `apply_rotary_pos_emb_apt`** L64-69；310p/kirinx90/kirin9030 仅 FP16/FP32 L70-109 | §2、§3 |
| shape 推导 | `op_host/apply_rotary_pos_emb_infershape.cpp`（186 行） | ✅ | 输出 = 输入与 cos/sin 广播（BroadcastShape L116-125）；**BroadcastDim 放行 dim1>dim2 = partial rope 维度包容** L52-54；unknown rank → [-2] L77-82 | §2.1 |
| tiling 基类（arch35） | `op_host/apply_rotary_pos_emb_tiling_base_arch35.cpp`（346 行） | ✅ | D_LIMIT=1024 L33、half/interleave coef=2 / quarter coef=4 L31-32；GetPlatformInfo（AIV 核数+UB）L40-60；**CheckShapeRelation**（cos==sin shape、cosN==1、S 一致、cosD≤qD、cosB∈{qB,1}）L158-207；CheckRotaryModeShapeRelation（D≤1024、half/interleave D%2==0、quarter D%4==0）L142-156；ConvertRotaryMode（dSplitCoef: half=2/interleave=1/quarter=4）L270-282；SetBsnd（4 种 layout 的 b/s/qn/kn/d/reald 抽取，TND 时 b_=1）L284-320；**isPartialRope_ = (reald_ != d_)** L319 | §2.1、§4.1 |
| tiling 模板 AB | `op_host/apply_rotary_pos_emb_tiling_ab_arch35.cpp`（154 行） | ✅ | 优先级 **10000**（最小=最先尝试）L27；TILING_KEY_AB=20030 L26；IsCapable：BSND cosb==b / SBND cosb∈{b,1} / TND L137-151；**SBND+cosb==1 时 b 折进 n（bs=s_, qn_=b_*qn_）** L56-60；dAlign=CeilAlign(reald/coef, 32B/typeSize)*coef L68；BS 打平分核 blockNum/blockFactor/blockTail L69-71；ubLoop/ubFactor 在 **QN+KN 合并轴**上切 L73-82；ubFactorBS L83-85 | §4.1、§4.1.3 |
| tiling 模板 ABA/BA | `op_host/apply_rotary_pos_emb_tiling_aba_and_ba_arch35.cpp`（271 行） | ✅ | 优先级 20000 L23；key ABA=20010 / BA=20011 L27-28；IsCapable：**仅 BNSD** L71-78；SplitCore（B 整除→只切 B；S 整除→只切 S；否则两种混合切分比 usedCoreNum）L80-135；ComputeUbFactor（UB 预算 4×(b·n+1)×s×dAlign 反解 ubFactorS→QN/KN→B）L137-185；**GetTilingKey：cosb==1→BA(20011)，否则 ABA(20010)** L246-249；SetBlockDim(blockNumB×blockNumS) L262 | §4.1、§4.1.3 |
| tiling 模板 BAB | `op_host/apply_rotary_pos_emb_tiling_bab_arch35.cpp`（246 行） | ✅ | 优先级 30000 L20；key BAB=20020 L25；IsCapable：**BSND 且 cosb==1** L47-57；SplitUb（canLoadDNum=ubSize/dAlign，÷4 因 3 入 1 出双缓冲；ubFactorQN/KN≤4095/coef）L151-181；SplitCore（同 ABA 双方案比选，但乘 ubFactorS 综判）L105-149；SetBlockDim(usedCoreNum_) L233 | §4.1、§4.1.3 |
| tiling 模板 legacy | `op_host/apply_rotary_pos_emb_tiling.cpp`（547 行） | ✅ | 优先级 40000（兜底）L542；CheckParams：**layout 仅 BSND/TND** L188、**D 与 cosD 必须 ∈{64,128}** L195-200、cosN==1 L201；Compute：每核 batch 均分 + 小 shape（preCoreBatch==1）→key=SMALL(1) L326-332；ComputeAB：UB 一次装多 batch，shengMte>0→key=AB(3) 否则 AB_CAST(4) L238-242；workspace=16MB L38 | §4.1、架构差异 |
| tilingdata 结构 | `op_host/apply_rotary_pos_emb_tiling.h`（241 行） | ✅ | legacy 40 字段 L27-66；**RegbaseTilingData**（B/CosB/S/D/QN/KN + blockNumB/FactorB/NumS/FactorS + ubLoop/Factor/Tail×S/B/QN/KN + rotaryMode/realDim/isPartialRope）L70-96；**RegbaseABTilingData**（+dAlign/dSplitCoef/blockNumBS/FactorBS/TailBS/ubFactorBS/ubLoopN/FactorN/TailN）L102-121；枚举 Layout{BSND=1,SBND=2,BNSD=3,TND=4} L139-144、RotaryMode{HALF=1,INTERLEAVE=2,QUARTER=3} L146-150、legacy key{SMALL=1,AB=3,AB_CAST=4} L132-137 | §4.1.3 |
| kernel 入口（arch35） | `op_kernel/apply_rotary_pos_emb_apt.cpp`（60 行） | ✅ | AIC 直接 return（AIV 路径）L31；4 key 分派：AB→`ApplyRotaryPosEmbAB<T>` L35-41、BAB→`ApplyRotaryPosEmbBAB<T>` L42-47、ABA→`ApplyRotaryPosEmbABAAndBA<T,false>` L48-53、BA→`<T,true>`（IsBBoardcast）L54-59 | §4.1.3、§6 |
| kernel 入口（legacy） | `op_kernel/apply_rotary_pos_emb.cpp`（64 行） | ✅ | AIV_ONLY L24；**workspace 空则直接 return** L25-32；key 1→ARPESmall / 3→ARPEComputeAB / 4→ARPEComputeABCast L36-47；**BF16 时计算类型=float**（`<DTYPE_QUERY, float>`）L52-62 | 架构差异 |
| kernel 公共 VF 库（arch35） | `op_kernel/arch35/apply_rotary_pos_emb_common.h`（879 行） | ✅ | **全部计算在 RegTensor<float> 域**（HalfAlignVF L66-124、QuarterAlignVF L133-214、InterleaveModeVF L217-328、DeepSeekInterleaveModeVF L331-452——⚠️ 未被本算子 4 key 分派使用，属姊妹算子资产）；Batch 版本（带 IsBBoardcast 模板参）L454-779；**partial rope 搬运族**：CopyIn/OutHalfMode（双半场 srcStride=D-reald/coef）L781-813、InterleaveMode L815-841、QuarterMode（4 段）L843-877 | §5、§6、精度 |
| kernel AB（arch35） | `op_kernel/arch35/apply_rotary_pos_emb_ab.h`（269 行） | ✅ | BS 打平偏移 L74-81；UB 4 队列 DB_FLAG=2 L83-88；Process：尾核 blockTailBS L94-95、currBSNum>1→ProcessQKLoop（Q 全 N 再 K 全 N）否则 N 轴循环 L113-120；**ProcessLoop 处理 Q/K 跨界块**（ubStart 跨 QN 时拼 Q 尾+K 头）L142-161 | §6 |
| kernel BAB（arch35） | `op_kernel/arch35/apply_rotary_pos_emb_bab.h`（296 行） | ✅ | dAlign 计算 L90；ubFactorN=max(QN,KN) L91-92；Process：B 循环→S 循环（ubFactorS 步进）L120-133；ProcessNLoop：cos/sin 按 S 段载入一次，先 ProcessQN 全循环再 ProcessKN L135-154 | §6 |
| kernel ABA/BA（arch35） | `op_kernel/arch35/apply_rotary_pos_emb_aba_and_ba.h`（386 行） | ✅ | 三重循环 ProcessInSLoop→ProcessInSBLoop→ProcessInSBNLoop L168-244；**BA（IsBBoardcast）时 cos/sin 在 S 段只载一次、B 循环复用** L185-196；CopyInCosAndSin 用 LoopModeParams 做 B 轴 loop 搬运 L246-275；CopyInQOrK 用 LoopModeParams 处理 BNSD 的 B→N→S 三维跨步 L277-320；partial rope 按模式分派 CopyInHalf/Quarter/Interleave L301-308 | §6 |
| kernel legacy 实现 | `op_kernel/apply_rotary_pos_emb_{base,small,compute_ab,compute_ab_cast}.h` | ✅ | ApplyRotaryPosEmbBase（base.h:24）；ARPESmall（小 shape 每核 1 batch）；ARPEComputeAB（多 batch 聚集搬出）；ARPEComputeABCast（1:1 搬入搬出 / BF16 路径，ComputeBF16 内 Cast float 计算、CAST_ROUND 回写 compute_ab_cast.h:250-284） | 架构差异 |
| 平台 config | `op_host/config/{ascend310p,ascend910,ascend910_93,ascend910b,ascend950,kirin9030,kirinx90}/` | ✅ | 7 平台 binary.json + simplified_key.ini；ascend950 binary.json 收 3 dtype（fp16/bf16/fp32）ND | §3 |
| 文档 | `README.md`、`docs/aclnnApplyRotaryPosEmb{,V2}.md` | ✅ | 三模式公式 README L19-136；参数表 L140-147；**推理系列不支持 BF16** L149；legacy 约束（layout 仅 BSND/TND、D∈{64,128}、rotary 仅 half、不支持空 tensor）L153-165；950 约束（4 维、cosB∈{qB,1}、D≤1024、half/interleave D%2、quarter D%4）L167-170；V2 文档：V2 支持空 tensor、V1 不支持（aclnnApplyRotaryPosEmbV2.md L201-254） | §1、§2.1、§3 |
| aclnn 接口 | `op_host/op_api/aclnn_apply_rotary_pos_emb{,_v2}.{h,cpp}` | ✅ | V1=V2 语义子集（rotary_mode 锁 "half"，aclnn_apply_rotary_pos_emb.cpp:27-29）；inner 本体 extern 不在本目录 | §1 语义分层 |
| 调用示例 | `examples/test_aclnn_apply_rotary_pos_emb{,_v2}.cpp` | ✅ | aclnn V1/V2 调用样例 | §测试设计 |
| 测试 | `tests/ut/{op_kernel,op_host,op_graph}`、`tests/st/aclnnApplyRotaryPosEmb` | ✅ | 存在 UT/ST 目录（本阶段未逐行抽取，P4 前补典型 shape 锚点） | §测试设计 |
| 图融合 | `op_graph/apply_rotary_pos_emb_graph_plugin.cpp`、`fusion_pass/` | ✅ | 图注册插件 + 融合 pass（RoPE 图融合场景，不影响单算子语义） | 旁证 |
| benchmark case | ascendc-kernelgen-data | ⚠️ 近邻非精确 | **无 ApplyRotaryPosEmb 精确 case**；近邻：`kernel_generator/level2/RotaryMul`（单输入 x、half/interleave、torch_npu.npu_rotary_mul——本算子语义子集，无 q/k 双路、无 layout、无 partial）、`level2/InterleaveRope`（单输入 interleave only）、`ascendc_migration/level2/RopeWithSinCosCache`（vLLM cache 式，不同融合）、`level2/FusedRopeWithQkNormAndKvCacheUpdate`（含 qk-norm+kv-cache，超集方向不同） | §9 |

## 语义速览（README 公式 + kernel VF 核对）

三模式统一式：`out = x·cos + rotate(x)·sin`（q、k 两路独立同式），rotate 定义：

| rotary_mode | rotate(x) | 配对结构 | dSplitCoef |
|---|---|---|---|
| half（默认） | cat(-x₂, x₁)，x₁=x[.., :D/2]、x₂=x[.., D/2:] | 前后半场交叉配对：out₁=x₁c₁−x₂s₁，out₂=x₂c₂+x₁s₂ | 2 |
| interleave | 奇偶对 (x[2i], x[2i+1]) → (−x[2i+1], x[2i]) | 相邻两元素配对：out[2i]=xe·ce−xo·se，out[2i+1]=xo·co+xe·so | 1 |
| quarter | cat(−x₂, x₁, −x₄, x₃)，四等分 | 1↔2、3↔4 两段内各自 half 式配对 | 4 |

- **partial rope**：cos/sin 末维 realDim 可 < D，仅前 realDim 维参与旋转，[realDim, D) 原样直通（tiling_base_arch35.cpp:319 isPartialRope_；kernel 用 CopyIn/OutHalfMode 等双段搬运实现，common.h:781-877）。
- **cos/sin 广播**：N 维恒为 1；B 维 ∈ {qB, 1}（CosB==1 即全 batch 共享一份 cos/sin → BA/BAB 两个 broadcast 专用 key 的动因）。
- **精度**：arch35 全模式在 MicroAPI `RegTensor<float>` 域计算（common.h 各 VF），Load/Store 辅助函数完成 dtype↔fp32 转换；legacy BF16 路径显式 `<T, float>` 计算 + CAST_ROUND 回写（apply_rotary_pos_emb.cpp:52-62、compute_ab_cast.h:250-284）。

## tilingkey / 模板分派全景（REGISTER_OPS_TILING_TEMPLATE，优先级数字越小越先尝试，tiling_templates_registry.h:468-470）

| 优先级 | tiling 类 | IsCapable 条件 | tilingkey | kernel 类 |
|---|---|---|---|---|
| 10000 | TilingAB | BSND cosb=B / SBND cosb∈{B,1} / TND | 20030 | ApplyRotaryPosEmbAB |
| 20000 | TilingABAAndBA | 仅 BNSD | cosb==1 → 20011(BA)，否则 20010(ABA) | ApplyRotaryPosEmbABAAndBA<T, IsBBoardcast> |
| 30000 | TilingBAB | BSND 且 cosb==1 | 20020 | ApplyRotaryPosEmbBAB |
| 40000 | legacy membase | 无条件（但 CheckParams 限 layout∈{BSND,TND}、D∈{64,128}） | 1(SMALL)/3(AB)/4(AB_CAST) | ARPESmall / ARPEComputeAB / ARPEComputeABCast |

arch35（950）上前三个按序命中其一；非 regbase 平台（910b 等）IsRegbaseSocVersion 为假，落 legacy。key 命名 = cos/sin 的 B/N 维广播形态（A=全量、B=广播）：ABA=b,s,1,d 全量；BA=1,s,1,d（B 广播）；BAB=BSND+1,s,1,d；AB=通用打平。

## 缺省项与注意点（P1 必须处理）

1. **benchmark 精确 case 已补建（2026-09-08，晚于 P0 时点）**：`ascendc-migration/level2/5_ApplyRotaryPosEmb/` 45 case（3 模式×4 layout×3 dtype×GQA/MQA×cosB 广播×partial rope+特殊值）。审核发现其 Model golden 原先把 fp16 留在原生精度计算（与 kernel fp32+RTNE 舍入序列不一致，5 个大量级 fp16 case 会误判正确 kernel 为 FAIL），已修复为全 dtype fp32 计算域并补 partial rope 支持；修复后 45/45 与「fp32+RTNE 完美 kernel 代理」在 evaluator 真实判定逻辑下互验通过（含 partial 直通段 bitwise 校验）。近邻 RotaryMul/InterleaveRope 仍为语义子集，仅作模式级交叉验证。
2. **DeepSeekInterleaveModeVF 在 common.h 存在但本算子 4 key 不分派它**（apt.cpp 仅 HALF/INTERLEAVE/QUARTER 三分支）——P1 标注「在仓未用资产」，勿混入本算子语义。
3. **aclnnInner 本体不在本目录**（extern 声明）——V1/V2 wrapper 已够定语义分层（V1=V2 锁 half）；如需 inner 参数消化细节须跨仓查，本期标缺省。
4. **legacy 与 arch35 约束差异大**（D∈{64,128} vs D≤1024；layout 2 种 vs 4 种；rotary 仅 half vs 三模式）——卡片以 arch35 为主线，legacy 进「架构差异」节，勿混写约束。
5. tests/ut、tests/st 目录存在但未逐行抽取；P3 §9 典型 shape 先以 examples + README 约束构造，P4 前回补 UT 锚点。
6. 空 tensor：legacy 不支持（README L165）；arch35 base CheckShapeAllPositive 拒绝非正维度（tiling_base_arch35.cpp:120-140）；V2 文档称支持空 tensor（docs L206 区域）——以代码为准：950 上 V2 走 arch35 模板，空 tensor 在 tiling 校验被拒，**文档与实现存在偏差，以实现为准**。

## P0 质量门结论

五类材料齐全（def/infershape/tiling×5/kernel×7/文档/示例/config），kernel+tiling 本体全量在仓内，非壳算子；aclnn inner 与 benchmark case 两项缺省已写明原因与替代（自建标杆）。体量约 5100 行，语义丰富但封闭 → 通过，进入 P1。
