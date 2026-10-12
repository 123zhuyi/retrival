# P0 材料清单 — flash_attention_score（服务 level4 算子 SelfAttention）

> 生成时间：2026-09-21｜来源目录：`src/flash_attention_score/`（来自 `src/ops-transformer/attention/flash_attention_score`，gitcode.com/cann/ops-transformer @ d28d8f99a；与 CANN 9.1.0 安装包 `opp/.../ops_transformer/ascendc/flash_attention_score/` 同源，本仓多带 op_host/op_api/docs/tests）
> **kernel 类型判定：CV 融合（Cube+Vector）**——入口 `op_kernel/flash_attention_score.cpp:379` `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（1 Cube : 2 Vector）。**非壳算子**：注意力核全部调度/计算本体在仓（op_kernel 728K arch32 + 100K arch35，op_host 齐全）。
>
> **🔺 低价值预判：非低价值——全量执行 P1~P5。**
> - **❌ 不可达项预计 ≥4**：① Cube-Vector 异构流水（Cube bmm1/bmm2 × Vector softmax，专栏 `FlashAttentionScoreS1s2Bn2gs1`）Triton 无对应编程模型；② level4 的 **4 个 nn.Linear 投影（q/k/v/out）不在本 kernel 内**，对应 CANN **MatMul**，须在卡片中另列（P2 拆出，`mapping.md` 显式标注投影段）；③ varlen/FASV 的 actual_seq 变长 + rope + fp8 量化反量化通路；④ BNSD/BSH 多 layout。
> - 有 pse（位置偏置 real_shift/pse_type）、atten_mask、drop、sparse_mode、prefix、sink、varlen、fp8——工业算子特征全。
> - level4 目标为**统一大 S 非对齐形状**（S∈[512,5182]、head_dim∈[32,256]、d_model≤8192、B≤8，100% S>64），FAS 工业实现按主路径 S1S2 分块 + 在线 softmax 覆盖；P1 以算法结构（S1S2 分块、online softmax、Cube/Vec 流水）为准，不照搬规格。

## 与 level4 目标的语义差（P0 动作 5：benchmark diff）

level4 定义见 `src/flash_attention_score/level4_target/SelfAttention/{definition.json,workload.jsonl}`——主目标 = **8_SelfAttention 大 S 集**（50 组，全部 S>64，S∈[512,5182]）；次目标小 S 单块集 archive 于 `SelfAttention_smallS/workload.jsonl`（沿用原 definition.json 参照）。

| 维度 | level4 SelfAttention | CANN FlashAttentionScore | 差距定性 |
|---|---|---|---|
| 接口 | run(x, n_heads) → output；**内部先做 4 个 nn.Linear 投影**（q/k/v/out, bias=False, seed=42） | 输入 query/key/value + real_shift/drop/padding/atten_mask/prefix/varlen(actual_seq)/fp8(d_scale、query/key_rope、sink、p_scale)；输出 softmax_max/sum/out、attention_out | level4 = 注意力核(**真子集**) + **4 个投影(在 kernel 外，=MatMul)** |
| 注意力主体 | softmax(QKᵀ/√head_dim)·V，**fp32 累加**（q/k/v 转 fp32 算） | softmax(QKᵀ·scale)·V，online softmax，fp32 累加 | **1:1 对应** |
| 投影参数 | 4 组随机权重（seed=42），仓库外不得存参、逐例重算 | 无投影逻辑（输入即 q/k/v/out 由前端喂入） | 需在卡片**追加「投影四段 = MatMul」指令** |
| 形状 | 统一大 S：S∈[512,5182]、d_model∈[160,8192]、head_dim∈[32,256]、n_heads∈[3,64]、B∈[1,8]，100% S>64 | 工业级全量 + 对齐约束 | level4 形状大部非 2 幂(head_dim>128 落动态 D)，P1 以主路径 S1S2 分块 + 在线 softmax 为准 |
| dtype | f16/bf16/f32 | 同 + fp8 族 | 兼容 |
| 布局 | B,S,D 固定（=BSH 图元） | input_layout 属性多值（BSH/SBH/BSND/BNSD/TND） | level4 恒 BSH |

> **模板变体命中（arch32 template_tiling_key.h）**：大 S + BSH + Sparse=ALL + Bmm1Format=ND + 无 mask/pse/drop/rope
> → 命中 **FP16/BF16/FP32 三组的 S1S2 通用模板**（`_s1s2_bn2gs1.h`，S1TemplateType/S2TemplateType=0 动态、DTemplateType=0 动态
> 或 {5,6,8}=D80/96/128），主实现 `flash_attention_score_s1s2_bn2gs1.h`。head_dim∈{80,96,128} 命中 D 模板档；
> head_dim∈{144,224,256} 超 D 模板上限 128 → 动态 D(DTemplateType=0)。B/TND/sameAB/scalar-const 等模板不命中。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/flash_attention_score_def.cpp` | ✅ | 输入 query/key/value REQUIRED，real_shift/drop_mask/padding_mask/atten_mask/prefix/actual_seq_qlen/actual_seq_kvlen/q_start_idx/kv_start_idx/d_scale_q/k/v/query_rope/key_rope/sink/p_scale OPTIONAL；输出 softmax_max/softmax_sum/softmax_out/attention_out；属性 13：scale_value/keep_prob/pre_tockens/next_tockens/head_num/input_layout/inner_precise/sparse_mode/pse_type/seed/offset/out_dtype/softmax_out_layout | §2、§2.1 |
| 平台配置 | def.cpp AICore().AddConfig | ✅ | ascen950 / ascend910_93 / ascend910b 等多平台 | §3 |
| shape 推导 | `op_host/flash_attention_score_infershape.cpp` | ✅ | softmax_max/sum、attention_out 推导 | §Infershape |
| tiling host 分发 | `op_host/flash_attention_score_tiling.cpp` | ✅ | 按平台/arch 分派 arch32/arch35 | §4 |
| tiling host arch32 | `op_host/arch32/flash_attention_score_tiling_*.cpp`（S1S2/BN2GS1/var_len 多档） | ✅ | 分核（B×N×S1/S2 轴）、D 分档、tilingkey 组装 | §4.1 |
| kernel 入口 | `op_kernel/flash_attention_score.cpp`（含 apt 变体） | ✅ | `__DAV_C220_CUBE__` Cube 分支（:37）；`KERNEL_TYPE_MIX_AIC_1_2`（:379）；空 tensor/常传播 | §4.2、§6 |
| kernel 主实现（arch32，P1 主线） | `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h` | ✅ | `template<class FlashAttentionScoreS1s2Bn2gs1>`（:85）：`Init`(:89)/`Process`(:95)/`InitBuffer`(:135)/`ProcessVec1`(:155)/`ProcessVec2`(:172) 等；sab 变体 `..._sab.h`、单 S1 `..._s1_bn2gs1.h`、var_len `flash_attention_var_len_score.h` | §5、§6、架构差异 |
| kernel 主实现（arch35） | `op_kernel/arch35/flash_attention_score_*.h`（100K） | ✅ | regbase 新流水 | §5、§6 |
| tilingdata 结构 | `op_kernel/arch32/flash_attention_score_tiling.h` | ✅ | 字段语义 P1 逐字段抽取 | §4.1.3 |
| tilingkey 编码 | `op_kernel/arch32/flash_attention_score_template_tiling_key.h` | ✅ | ASCENDC_TPL 位段：S1TemplateNum ∈{0,64,128,512} 等 | §4.1.3 |
| aclnn 接口 | `op_api/aclnn_flash_attention_score.cpp` | ✅ | V1 GetWorkspaceSize(:1007)/run(:1103)；V2 :1215/:1305；V3 :1314 | §2.2、语义分层 |
| Matlab 投影（level4 补） | 仓库外（不在本 kernel） | ❌ | 4×nn.Linear(D,D,bias=False) = CANN **MatMul**，P2 卡片补列 | §2.2 |

## arch 覆盖核实（P0 动作 3）
- `src/flash_attention_score/op_kernel/` 含 **arch32（728K）+ arch35（100K）**，**无 arch22**。
- ⚠️ 与 ScaledDotProduct 不同：本次不补 `flash_attention_score_cann921`（arch22）目录。若后续需要 arch22 硬件代参考，可再从 CANN 9.1.0 opp 包提取；P1 以 arch32/35 为线。

## P0 质量门自查
- [x] 材料齐全：def/infershape/tiling/op_api/docs/tests 六类齐（kernel 本体在仓）
- [x] 低价值判定有据（CV 融合非壳 + ≥4 不可达项 + 投影拆分，均记档）
- [x] arch 覆盖已核实（arch32/35，无 arch22）
- [x] level4 目标对照（投影拆分充分披露）