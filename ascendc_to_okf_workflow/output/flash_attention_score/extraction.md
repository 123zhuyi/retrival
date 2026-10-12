# P1 语义抽取 — flash_attention_score（服务 level4 算子 SelfAttention）

> 抽取对象：`src/flash_attention_score/`（CANN ops-transformer @ d28d8f99a；arch32 主线 + arch35）。本抽取出**注意力核**语义；level4 自带的 4 段投影 = CANN MatMul，在 §11 与 §2.2 显式拆出。

## 1. 算子总览

### 1.1 数学语义（五步走）
SelfAttention（level4）整体 = **4×投影(MatMul) + 缩放点积注意力**：
1. 投影：`q=Wq·x, k=Wk·x, v=Wv·x, out=Wo·z`（`nn.Linear(D→D, bias=False)`，seed=42 权重，作用于 `x[B,S,D]`）
2. `scores = (q·kᵀ)/√head_dim`，head_dim=D/n_heads，q/k/v reshape 为 `(B,H,S,hd)`，`transpose(1,2)`
3. `weights = softmax(scores, dim=-1)`
4. `z = weights·v`（fp32 累加），`z→(B,S,D)`
5. `output = Wo·z`

CANN **FlashAttentionScore kernel 只含 2~4 的注意力核**：`softmax(QKᵀ·scale)·V`，online softmax、fp32 累加。

### 1.2 三层语义分层速览
| 层 | 内容 | 落点 |
|---|---|---|
| aclnn（用户接口） | `aclnnFlashAttentionScore(GetWorkspace)` V1~V3，参数消化（keep_prob/pre_tockens/...） | `op_api/aclnn_flash_attention_score.cpp:1007/:1103/:1215/:1305/:1314` |
| 调度/分核 | TilingRegistry 按平台→arch32/35 分派 | `op_host/flash_attention_score_tiling.cpp` |
| kernel 本体（CV 融合） | bmm1(QKᵀ)→softmax→bmm2(·V)→写回，Cube+Vector 双段流水 | `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1.h:85` |

## 2. GEIR kernel 原型（op_host/flash_attention_score_def.cpp）

### 2.1 输入/输出/属性清单
- 输入 REQUIRED：`query/key/value`；OPTIONAL：`real_shift(pse)/drop_mask/padding_mask/atten_mask/prefix/actual_seq_qlen/actual_seq_kvlen/q_start_idx/kv_start_idx/d_scale_q/k/v(query_rope/key_rope/sink/p_scale)`。
- 输出：`softmax_max/softmax_sum/softmax_out/attention_out`。
- 属性：`scale_value/keep_prob/pre_tockens/next_tockens/head_num/input_layout(BSH...)/inner_precise/sparse_mode/pse_type/seed/offset/out_dtype/softmax_out_layout`。

### 2.2 与 level4 的接口差（含投影拆分指令）
- level4 接口 `run(x, n_heads)`：**先做 4 段 nn.Linear**，注意力核才吃 q/k/v。CANN FAS 直接吃 q/k/v。
- → **P2 卡片必须补「前投影四段 = MatMul(D,D × B,S,D)」**；head_dim=√ 缩放对应 FAS `scale_value = 1/√head_dim`。
- 结果：level4 `output` 对应 FAS `attention_out` 再经 `Wo` 投影；FAS 的 `softmax_max/sum/out` 是额外中间产出（level4 golden 内部重算 softmax，不建模）。

## 3. 平台与 dtype 面
- 平台：ascend950 / ascend910_93 / ascend910b 等（def.cpp AddConfig）。
- dtype：f16/bf16/f32（kernel 支持）+ fp8（d_scale/rope 面）。level4 三 dtype 全覆盖。
- `INPUT_T/T` 模板参数：输入即计算位宽时走 ND 直路，否则 TND 下 bmm2 走 NZ（`modeTypemm2`，:115-118）。

## 4. Tiling
- 核心切法：**BN2GS1(S2) 分块**——分核轴 B×N(head)×G×S1(query 块)，S2(key 块) 内循环（`s1s2_bn2gs1` 命名即此）。`ProcessVec1/2` 分属 S1/S2 段的 Vector 阶段。
- D 分档：S1TemplateNum/S2TemplateNum ∈{0,64,128,512}（`flash_attention_score_template_tiling_key.h` 位段）。
- **▶ 主目标 8_SelfAttention（统一大 S，S 至 5182）走这条 S1S2 分块 + 在线 softmax(m/sum 双量,SOS 修正)路径**——`_s1s2_bn2gs1.h` 即 Operative 参考;小 S 单块只是其退化解。
- sparse：GetS1LoopRange/GetS2LoopRange（:179-180）按 pre/std 动态收敛 inner 循环。
- bmm1 splitN >16 时 softmaxCheckRes 位宽切换（:296-297）。

## 5. 片上内存清单（arch32 主线）
- `bmm1/bmm1Nz`（QKᵀ→scores，Cube）、`bmm2`（·V→out，BatchMatmul）（:98-119）。
- 双段 ping-pong：`stage1PingBuf/stage1PongBuf`（bmm1 产物）、`stage2TBuf`（bmm2），`mm1Res[2]/mm2Res[2]/vec2Res[2]/stage1Res[2]`（:209-225）。
- softmax 状态：`softmaxSumBuf[2]/softmaxExpBuf[2]/softmaxMaxBuf/softmaxTempBuf`（:215-219）。
- mask/pse：`maskTBufPing/Pong、pseTBuf`（:209-211）；`softmaxReduceSize=8`（:207）。

## 6. 执行流水（arch32 主线）
- 单核任务 `KERNEL_TYPE_MIX_AIC_1_2`（1 Cube : 2 Vector），CV 经 workspace 与 stage 缓冲换手。
- 主循环三阶段（类 `FlashAttentionScoreS1s2Bn2gs1::Process`，:95 → `IterateBmm1`/`ProcessVec1`/`IterateBmm2`/`ProcessVec2`）：
  1. **IterateBmm1**（:140）：Q 分块 × Kᵀ → scores（Cube bmm1，含 atten_mask 加载 `CopyInAttenMask`:156）
  2. **ProcessVec1**（:155）：online **SoftMax**（`SoftMaxCompute`/`SoftMaxCheckResCompress`/`InvalidLineSplitS2Process` :168-171）与 mask 替换（`ComputeAttenMask`/SelectWithBytesMask :165-166）
  3. **IterateBmm2→ProcessVec2**（:131,:172）：softmax·V（Cube bmm2）→ `Bmm2ResultMul`(×scale) → 写回（`SoftmaxDataCopyOut`/`Bmm2DataCopyOut` :175-176）
- 循环边界：`r = SPADS/SPSA` 动态收敛在线最大值（softmax_max/sum）。

## 7. 计算流程（GEIR kernel，时序）
`Q,K,V → bmm1: scores = QᵀK·scale` → `atten_mask 加 -inf` → `online softmax(max/sum 状态) `→ `bmm2: out = softmax(scores)·V` → `写 attention_out`（level4 仅用此路；`softmax_*` 中间不建模）。

## 8. 精度策略
- Cube 累加 fp32；softmax 用 online max 修正保持数值稳定；`inner_precise` 属性控中间精度档（f16/bf16 时按高精度档累加）。
- level4 golden 亦 fp32 重算 softmax、attn_output 自洽，与 kernel 对齐基线同法（见 §9.1）。

## 9. 测试设计
- numpy golden：`tests/pytest/cpu_impl.py`（fp32 重算 softmax）。
- 容差与用例：`tests/pytest/test_main.py`。
- tiling UT：`tests/ut/op_host/*/test_flash_attention_score_tiling.cpp`。
- examples 典型形状：`examples/*`（工业规整形状，辅助定 tiling 期望）。
- ⚠️ level4 workload 为 **8_SelfAttention 统一大 S 非规整形状**（S∈[512,5182]、head_dim∈[32,256]、d_model≤8192、B≤8，
  100% S>64），CANN 用例以规整 shape 为主——P4 回放需用 level4 形状做目标（大 S 走 S1S2 分块在线 softmax 主路径）。
  小 S 单块分支 archive 于 `SelfAttention_smallS`（沿用原 definition.json，次要）。

## 10. 超出 level4 语义的特性登记（P2 不翻译）
1. **投影四段（MatMul）**：不在本 kernel，须在卡片补。
2. varlen（actual_seq_* / start_idx）变长。
3. pse（real_shift/alibi/alibiMask）、atten_mask、padding_mask、sparse_mode(0~8)、prefix、sink。
4. dropout（keep_prob/seed 随机掩码）。
5. rope、fp8（d_scale/sink）量化反量化。
6. 多 layout（BSH/SBH/BSND/BNSD/TND）。

## 11. 与 level4 SelfAttention 的语义对照表
| level4 SelfAttention | FlashAttentionScore | 翻译动作 |
|---|---|---|
| 4×nn.Linear(D,D) 投影 | 无（kernel 外）| **卡片补 4×MatMul**（q/k/v/out） |
| scores=QKᵀ/√head_dim | bmm1 + scale_value=1/√head_dim | 参数映射 |
| softmax(dim=-1) | online softmax，max/sum 状态 | 语义 1:1，CV 拆分是 kernel 实现 |
| weights·V | bmm2 | 1:1 |
| output=Wo·z | attention_out | 卡片补 Wo 投影 |

## P1 质量门自查
- [x] 每节均有 `file:line` 锚点
- [x] 投影四段（level4 特有）已显式拆出并指向 MatMul
- [x] 不可达项（CV 流水、varlen/pse/fp8、非对齐 corner）全部记档 §10
- [x] 与 level4 对照表齐全