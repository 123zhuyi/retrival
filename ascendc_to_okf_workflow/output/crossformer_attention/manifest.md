# P0 材料清单 — CrossformerAttention（服务 level4 算子 CrossformerAttention）

> 生成时间：2026-09-21｜服务目标：`src/flash_attention_score/level4_target/CrossformerAttention/`
> 参考源：主 `src/flash_attention_score/`（FAS 前向，pse_shift/atten_mask/scale 语义 + FlashSoftmax 算法），
> 次 `src/swin_attention_score_quant/`（窗口化结构，量化版，仅取窗口分块组织方式）。
> 两者均克隆自 `src/ops-transformer`（gitcode.com/cann/ops-transformer @ d28d8f99a，CANN Open Software License 2.0），
> device 源码与 CANN 9.1.0 安装包 `opp/.../ops_transformer/ascendc/flash_attention_score/` 同源。

**kernel 类型判定：CV 融合（Cube+Vector）**——注意力主体是 Cube 矩阵乘（QKᵀ、softmax、·V），
CANN 参考 `flash_attention_score.cpp` 走 FlashAttention（FlashSoftmaxV2 在线 softmax，Cube 乘 + Vector
exp/max/sum 协同）。**非壳算子**：调度与计算本体在仓（op_kernel 3.0M+、op_host 500K+）。

**🔺 低价值预判：非低价值。**
- ❌ 不可达项 ≥2：① Cube-Vector 异构流水（CANN 的 SkvSplit 在线 softmax + exp 补偿链在 Triton 无原生
  对应，需用 `tl.dot` + loop-carried `tl.maximum`/expacc 复刻 flash 重构）；② 窗口分组重排（feature_shape
  的 view/permute）是 **host 侧 layout 变换**，kernel 内无对应，须在 wrapper 侧用索引折算。
- 注意力是典型 Cube 高价值算子族（FASGrad/SFA 同族），卡片价值密度高。
- **level4 目标语义与此前 flash_attention_score_grad 对齐：是 FAS 的弱子集**，仅取 pse/scale/mask 三条路径。

## 与 level4 目标的语义差（P0 动作 5：benchmark diff）

level4 定义见 `src/flash_attention_score/level4_target/CrossformerAttention/{definition.json,workload.jsonl}`。
golden 参考（definition.json `reference`）以 torch 实现：`to_heads → qh*=scale → attn=qh·khᵀ(+pos_bias)(+mask) → softmax(-1) → ·vh → permute back`。

| 维度 | level4 CrossformerAttention | CANN FAS 参考 | 差距定性 |
|---|---|---|---|
| 接口 | run(x, group_size, q/k/v(opt), num_heads, mask(opt), scale(opt), pos_bias(opt), feature_shape(opt)) → output | 16 输入 + 多属性（pse_shift/atten_mask/scale_value/head_num/sparse_mode/...） | level4 是**真子集**：FAS 无窗口分组概念，分组属 host 侧 layout |
| 分组语义 | feature_shape 时按 (gh,gw) 空间窗口分组，窗口内 MHA；否则整序列单窗口 MHA | FAS 不支持分组，pse/mask 覆盖整个 S_q×S_k | **缺口**：group_size/feature_shape 的窗口重排 CANN 无现成实现 |
| 加性偏置 | pos_bias（additive） | `pse`：score = QKᵀ·scale + pse + ...（docs/aclnnFlashAttentionScore.md:23、:123） | 语义同构，level4 pos_bias ↔ FAS pse |
| 掩码 | mask（additive，本 workload 是数值 + 相加，非 select 布尔） | `atten_mask`：select 遮蔽 true → −inf（FA设计介绍.md §1） | **差异**：level4 是**加性数值 mask**；FAS atten_mask 是**布尔 select 遮蔽**——对标以 level4 加性语义为准 |
| 缩放 | scale（attr，默认 head_dim^-0.5） | `scale_value`（def.cpp:157，默认 1.0） | 语义同构，level4 在 torch 侧预乘 qh |
| softmax | torch softmax(-1)，fp32 重算 | FlashSoftmaxV2 在线重构（FA设计介绍.md §1 步 1~4） | level4 全量 softmax，FAS flash 分块——算法不同但数学等价（对标时都收敛到同一 softmax） |
| dtype | f16/bf16/f32 | FAS 同 + fp8 族 | 兼容 |
| 形状 | 窗口 G∈{7..7921}，C∈{64..1024}，n_heads∈{1..512}，非对齐 | FAS 工业级对齐约束（docs :470-499） | level4 大量非对齐小窗口，FAS 模板不保证支持；P1 以算法结构（flash 重构）为主，不照搬规格 |
| 布局 | 2D (N,C) 或 3D (B,N,C)；feature 时 (H·W,C) | ND ↔ BSH/SBH/BNSD/TND | level4 固定 ND；wrapper 统一为 2D 张量 |

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 承载详设章节 |
|---|---|---|---|---|
| level4 目标定义 | `src/flash_attention_score/level4_target/CrossformerAttention/definition.json` | ✅ | `reference`=torch `run()`（to_heads/permute/分组逻辑）；50 case workload | §1.1、§2 |
| 参考公式 | `src/flash_attention_score/docs/aclnnFlashAttentionScore.md:23` | ✅ | `attention_out = Softmax(Mask(scale*(pse+Q·Kᵀ), atten_mask))·V`（dropout 忽略） | §1.1 |
| 设计文档 | `src/flash_attention_score/docs/FA算子设计介绍.md` | ✅ | §1(1~4) flash 4 步；§3 Tiling(:30)；§4 流水(:61 V 侧 :76 CV)；§5 多模板(:90)；§6 API | §5、§6 |
| 原型定义 | `src/flash_attention_score/op_host/flash_attention_score_def.cpp` | ✅ | pse_shift/atten_mask/scale_value(:157)/head_num(:161 REQUIRED)/value_layout/input_layout；输入 atten_mask(:59) 等 | §2 |
| tiling host | `src/flash_attention_score/op_host/flash_attention_score_tiling.cpp` + arch32/ 变体 | ✅ | 分核（B×H×S2 轴）、S1/S2 语义 | §4.1 |
| kernel 入口 | `src/flash_attention_score/op_kernel/flash_attention_score.cpp`、`_apt.cpp` | ✅ | 模板分支（TEMPLATE_MODE）、AIC/AIV dispatch | §4.2 |
| kernel 主实现 | `op_kernel/arch32/flash_attention_score_s1s2_bn2gs1s2.h`（+s1_bn2gs1/sab/var_len 变体） | ✅ | Cube MM + FlashSoftmaxV2 在线重构；S1/S2 双层 | §5、§6 |
| 窗口/分组结构 | `src/swin_attention_score_quant/`（op_kernel/*.h、格、文档） | ⚠️ | swin 是量化低层 NZ/Transpose，**无清晰窗口算法层**；窗口分块语义以 level4 reference + CrossFormer 结构为准（文档记 host 侧） | §1.1、§5 |
| 测试标杆 | `tests/`（CPU 实现）as FAS；crossformer 无专属 test | ⚠️ | golden 以 level4 `definition.json reference`（torch）为单标杆 | §测试设计 |
| 图级语义 | `op_graph/`（FAS forward 有 framework/op_graph） | ⚠️ | 与 grad 同，本卡不依赖图级融合 | — |

## 缺省项与注意点（P1/P2 必须处理）

1. **窗口分组 = host 侧 layout 变换，kernel 内不可达**：feature_shape 的 token→window 映射
   `(h//gh, w//gw)` 是固定函数，可在 Triton wrapper 用索引折算（div/mod 展开，C-001 无 `%`），
   进 kernel 前把 x/q/k/v 重排为窗口连续（或 kernel 内线性 offs 折算坐标）。非 feature 退化为单窗口。
2. **level4 mask 是加性数值、FAS atten_mask 是布尔 select**：对标以 level4 加性语义为准；
   FAS 只提供『偏置在 softmax 前相加』这条结构证据。
3. **feature + mask/pos_bias 组合在 level4 reference 存在形状不一致**（mask 常为 (H,N,N) 全序列，
   而窗口内 attention 是 (B_flat,G,G)），torch 参考在 asleep 时广播**会报错**——属 reference 缺陷。
   本卡定义：mask/pos_bias 按**窗口局部坐标**索引（每窗口取对应切片），feature 无 mask/pos_bias 的
   case 与 reference 严格一致；组合 case 的语义以本卡定义为契约（写入卡 §2/§1.1）。
4. **Scale 施加位置**：level4 预乘 `qh *= scale`（torch side），FAS 是 score 后乘。对标数学等价，
   Triton 统一在 QKᵀ 后乘 scale（记入卡 §1.1 对齐点）。
5. **大窗口非 feature case（G 可达 3721/7921）**：不能整窗 QKᵀ 一次性算（attn 矩阵放不下 UB），
   必须走 flash 重构（key 分块在线 softmax）——与 FAS 的 S_kv 切分（FlashSoftmaxV2）同型，P2 主线。
6. **precision 契约**：fp32 输入 `tl.dot` 需 `input_precision="ieee"`（C-023 operator_class_patterns §4.2）；
   f16/bf16 中间 attn 用 fp32 累加与 softmax（对齐 level4 `qh.float()` 重算）。