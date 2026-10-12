# P1 语义抽取 — CrossformerAttention（v2：补齐 host tiling 档位阶梯）

> 章节号对齐详设模板。来源：level4 `definition.json reference`（torch golden，**权威算法**）、
> CANN FAS `FA算子设计介绍.md`（flash 结构/流水）、`aclnnFlashAttentionScore.md`（pse/mask 语义）、
> **`op_host/arch35/flash_attention_score_tiling_basic.cpp`（v1.5 新增：host 档位阶梯，第一顺位）**。
> 只做事实搬运，不带转化。

## 1.1 功能分析（算什么）

**一句话**：CrossFormer 分组注意力——把 token 序列（可选 reshape 为 H×W 空间）按 (gh,gw) 窗口分组，
窗内做多组 scaled dot-product 多头注意力（QKᵀ·scale + 加性 pos_bias + 加性 mask → softmax → ·V）。

数学公式（level4 `definition.json` reference 反推）：
```
B_eff, G, C = 窗口批量, 窗口内 token 数, 通道；n_heads；head_dim = C / n_heads
qh[q,h,i,j] = q[q,i,h,j] * scale_val            # to_heads: (B,N,H,hd).permute(0,2,1,3) → (B_eff,H,G,hd)
attn[q,h,i,j] = sum_d qh·khᵀ                    # matmul (B_eff,H,G,G)，fp32
attn += pos_bias + mask                         # 加性偏置（逐头/全头）
attn = softmax(attn, dim=-1)
out[q,i,h,j] = sum_j attn·vh                     # matmul (B_eff,H,G,hd) → permute back → (B_eff,G,C)
scale_val = scale if 给定 else head_dim^-0.5
```
- q/k/v 缺省即 x；feature_shape 时 x 先 reshape (B,H,W,C) → 分窗 → 展平窗口批量 (B_flat, gh·gw, C)。
- 非 feature 分支 = 全序列单窗口 (B,N,C) 整段 MHA。

**与 CANN FAS 对照**（`docs/aclnnFlashAttentionScore.md:23`）：FAS `score=scale*(pse+Q·Kᵀ)`，
attn_mask select 遮蔽后 FlashSoftmax、·V。level4 pos_bias↔pse、scale↔scale_value 同构；mask 语义有差异。

## 2 原型设计（接口）

| 输入/属性 | 类型 | 必选 | 语义 | CANN FAS 对应 |
|---|---|---|---|---|
| x | Tensor(N,C)/(B·N,C) | ✅ | token 序列/空间展平 | query/key/value(self) |
| q/k/v | Tensor | ➖ | 独立 q/k/v（缺省=x） | query/key/value |
| group_size | int 或 [gh,gw] | ✅ | 窗口大小 | —（host layout） |
| num_heads | int | ➖ | 头数默认 1；head_dim=C/num_heads | head_num（def.cpp:161） |
| mask | Tensor | ➖ | 加性掩码 | atten_mask（布尔 select） |
| pos_bias | Tensor | ➖ | 加性位置偏置 | pse |
| scale | float | ➖ | 缩放；缺省 head_dim^-0.5 | scale_value（def.cpp:157） |
| feature_shape | [H,W] | ➖ | 分组前 reshape | — |

dtype：x/q/k/v∈{f16,bf16,f32}；mask/pos_bias 常 f32/f16。

## 2.1 原型约束

- head_dim=C//num_heads 须整除；分组要求 (H%gh)==0 且 (W%gw)==0（reference 直接 view/permute，不 pad）。
- 非 feature 时 group_size 被 reference **忽略**（全序列单窗口）。
- feature+mask/pos_bias 形状不一致是 reference 缺陷——本卡以窗口局部切片为契约。
- **head_dim 范围**：C 可达 1024、num_heads 可达 512 → head_dim∈{24..1024}（非 2 幂常见，如 96）。

## 3 规格约束

- 芯片 Ascend910（本机 AIC 20~24/852 4..9372，arch **910b**），CANN 9.1.0，torch_npu 2.10。
- dtype f16/bf16/f32；中间 attn fp32 累加。非对齐形状普遍，CANN 工业对齐约束不适用。

## 4.1 分核策略

- **v2 主证据 = arch35 host 档位阶梯（第一顺位）**，见下 §4.1.3。
- FAS `tiling.cpp` 任务轴 B×H×S2，`usedCoreNum` 决定核数。level4 语义下任务轴 =
  **(B_eff × n_heads × cdiv(G,BLOCK_M))**；Triton persistent-core grid=min(NTOT, num_core)，核内 grid-stride。

## 4.1.3 tiling 档位阶梯（v1.5 新增核心事实）

**CANN arch35（DAV_3510，910b 生产路径）host 档位** `flash_attention_score_tiling_basic.cpp`：

**D 轴分段 `CalcDBasicBlock`(:51)**：
```
dBasicBlock = AlignUp(dSize + dSizeRope, D_TEMPLATE_SPLIT_SIZE)   # D_TEMPLATE_SPLIT_SIZE=64
dBasicBlock > 256  → DTemplateType::ALIGNED_768，直接返回（D 轴=768）
dBasicBlock == 64  → ALIGNED_64
dBasicBlock == 128 → ALIGNED_128
dBasicBlock == 192 → ALIGNED_192
dBasicBlock == 256 → ALIGNED_256
```
→ **D-tier = {64,128,192,256}（按 64 对齐上浮），>256 一律 768**。这是 BLOCK_D 的权杖来源。

**S1/S2 主档 `CalcS1S2BasicBlock`(:79)**（S1=query 块，S2=key/序列2 块，归 DFN 语义）：
```
① fp8(HIFLOAT8) 且 无 mask/pse/dropout/rope        → S1=128, S2=512
② d==64 && dV==64 && s1%128==0 && s2%64==0
   && 无 mask/pse/dropout 且 非 fp32 非 fp8 且无 rope → S1=128, S2=256   (DN 优化)
③ dSize>256
       fp32(...)                                  → S1=64,  S2=128
       非 fp32 / fp8 族                           → S1=128, S2=128
④ 缺省（dSize≤256）                              → S1=128, S2=128   ← level4 主流命中
```
- **Level4 主流（f16/bf16/fp32, head_dim≤256, 常有 mask/pos_bias）命中④ → 默认档 S1/S2=128/128**。
- fp32 且 head_dim>256 → ③ → **S1=64 / S2=128**。
- DN 优化（②）要求无 mask/无 pse——crossformer 常带 mask/pos_bias，故②基本命中不到。

**关键现实**：CANN 128/128 依赖 L1 3-buffer + D-workspace（`GetWorkspaceSize`(:180) 大 D 才开 bmm2 workspace）。
Triton 无 L1 多 buffer 机制 → 128/128 复刻会 **UB 溢出**（见 skeleton 头部预算）。§P2 取最近可移植近邻。

**regbase 生产路径**：`IsCapable`(:170)=DAV_3510；`GetTilingKey`(:148) 编码 s1/s2/d/dV TemplateType。
内部 MIX_AIC_1_2 流水（taskId ring buffer 等）**不可移植**，仅作排除清单（v1.5 L2）。

## 5 片上内存

- FAS 各 buffer：Q/K/V 分块（L1/low-UB）、ub_attention_out 双 buffer 缓存 running out、
  flash_sum/softmax_max/sum。**D-workspace**：大 D（>256）时 bmm2 结果落 GM workspace（GetWorkspaceSize）。
- Triton UB 预算以 `hardware_platforms.md`（910B A2 192KB，950 248KB）；见 skeleton 头部硬算。

## 6 流水编排

- FAS CV 流水：Cube（QKᵀ、PV）× Vector（mask/scale/exp/max/sum/div），double buffer 掩盖。
  Triton 侧 `tl.dot`+矢量 op+num_stages；**num_stages 在 Triton-Ascend 运行时不生效（C-006）** ⚠️。
- 910b regbase MIX_AIC_1_2 的 taskId 深流水、跨核 flag **不可移植**（L2）。

## 6.x 计算流程（flash 重构，对齐 FA设计介绍.md §1 步 1~4）

步 1: attn_blk = dot(Q块,Kᵀ块)*scale；+= pos_bias/mask 切片        [Cube+Vector]
步 2: m_new = max(m_old, rowmax(attn_blk))                        [Vector]
步 3: p_blk = exp(attn_blk−m_new)；α = exp(m_old−m_new)             [Vector]
步 4: o_acc = α·o_acc + dot(p_blk,V块)；l_acc = α·l_acc + rowsum(p_blk) [Cube+Vector]
循环完成: out = o_acc·(1/l_acc)                                     [Vector]

## 精度策略

- 中间 attn 与 o 累加 fp32（对齐 level4 `qh.float()` 重算与 FAS fp32 累加）。
- bf16 输入 dot fp32 累加；fp32 输入 dot `input_precision="ieee"`（C-023 operator_class_patterns §4.2）。
- f16/bf16 的 P 在 dot(V) 前 cast 回输入 dtype（对齐 CANN expcast 落点）。

## 测试设计

- 单标杆 = level4 `definition.json reference`（torch）。workload 50 case。
- P2.5 smoke：非 feature 小窗/feature 多窗/含 mask/pos_bias/fp32 输入/大窗 flash 多块；v2 另验 head_dim>128
  （D-tier 边界）与 fp32 大 D。