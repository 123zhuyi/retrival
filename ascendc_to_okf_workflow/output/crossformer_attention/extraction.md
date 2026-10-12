# P1 语义抽取 — CrossformerAttention

> 章节号对齐详设模板。来源：level4 `definition.json reference`（torch golden，**权威算法**）、
> CANN FAS `FA算子设计介绍.md`（flash 结构/流水）、`aclnnFlashAttentionScore.md`（pse/mask 语义）、
> `flash_attention_score_tiling.cpp`（分核）。只做事实搬运，不带转化。

## 1.1 功能分析（算什么）

**一句话**：CrossFormer 分组注意力——把 token 序列（可选 reshape 为 H×W 空间）按 (gh,gw) 窗口分组，
组内/窗内做多组 scaled dot-product 多头注意力（QKᵀ·scale + 加性 pos_bias + 加性 mask → softmax → ·V）。

数学公式（level4 `definition.json` reference 反推，逐变量标注）：
```
B_eff, G, C = 窗口批量, 窗口内 token 数, 通道
n_heads = H (默认 1)； head_dim = C / n_heads
qh[q,h,i,j] = q[q,i,h,j] * scale_val            # to_heads: (B,N,H,hd).permute(0,2,1,3)；(B_eff,H,G,hd)
attn[q,h,i,j] = sum_d qh·khᵀ                    # matmul (B_eff,H,G,G)，fp32
attn += pos_bias + mask                         # 加性偏置（逐头/全头）
attn = softmax(attn, dim=-1)                    # 行 softmax
out[q,i,h,j] = sum_j attn[q,h,i,j]·vh[q,h,j,d]  # matmul (B_eff,H,G,hd) → permute back → (B_eff,G,C)
scale_val = scale if 给定 else head_dim^-0.5
```
- q/k/v 缺省即 x；feature_shape 时 x 先 reshape (B,H,W,C) → 分窗 → 展平窗口批量 (B_flat, gh*gw, C)。
- 非 feature 分支 = 全序列单窗口（B, N, C）整段 MHA。

**与 CANN FAS 的对照**（`docs/aclnnFlashAttentionScore.md:23`）：
FAS `score = scale*(pse + Q·Kᵀ)`，经 atten_mask select 遮蔽后 FlashSoftmax、·V。level4 的
pos_bias↔pse、scale↔scale_value 语义同构；**差异在 mask**（level4 加性数值 vs FAS 布尔 select）。

## 2 原型设计（接口）

来自 level4 `definition.json`（torch 函数签名的 inputs/映射到 CANN FAS 原型）：

| 输入/属性 | 类型 | 必选 | 语义 | CANN FAS 对应 |
|---|---|---|---|---|
| x | Tensor(N,C) / (B*N,C) | ✅ | token 序列 / 空间展平 | query/key/value 之一（self） |
| q/k/v | Tensor | ➖ | 独立 q/k/v（缺省=x），KBP 场景 3D | query/key/value |
| group_size | int 或 [gh,gw] | ✅ | 窗口大小（feature 时=空间窗口边长） | —（host 侧 layout，无对应） |
| num_heads | int | ➖ | 头数，默认 1；head_dim=C/num_heads | head_num（def.cpp:161） |
| mask | Tensor(H,N,N)/(N,N) | ➖ | 加性注意力掩码（本 workload 数值相加） | atten_mask（def.cpp:59，布尔 select） |
| pos_bias | Tensor(H,N,N)/(N,N) | ➖ | 加性位置偏置 | pse / real_shift |
| scale | float | ➖ | 缩放；缺省 head_dim^-0.5 | scale_value（def.cpp:157，默认 1.0） |
| feature_shape | [H,W] | ➖ | x 的空间形状，用于分组前 reshape | — |
| output | Tensor | — | 与 x 同 shape | attention_out |

dtype：x/q/k/v∈{f16,bf16,f32}；mask/pos_bias 常 f32/f16 广播相加。

## 2.1 原型约束

- head_dim = C // num_heads 须整除（torch `.view` 隐含）。
- 窗口分组要求 (H%gh)==0 且 (W%gw)==0（reference 直接 view/permute，不 pad）。
- 非 feature 时 group_size 被 reference **忽略**（走全序列单窗口 else 分支）。
- **feature + mask/pos_bias 形状不一致是 reference 缺陷**（mask 全序列 (H,N,N) 与窗口 (B_flat,G,G) 不广播）。
- 参考对齐 level4 加性 mask 语义（每窗口取局部切片）作为契约（见 manifest 缺省项 3）。

## 3 规格约束

- 芯片：Ascend910（本机 AIC 20~24 / AIV 40~48），CANN 9.1.0，sim/kb 环境。
- dtype：f16/bf16/f32（本 workload 全集）；fp32 中间 attn 累加。
- 非对齐形状普遍（窗口 G 多变、C 非 16 对齐），CANN FAS 工业对齐约束不适用，P2 以算法结构为准。

## 4.1 分核策略（CANN FAS 参考）

- FAS `tiling.cpp`：任务轴取 B×H×S2（S2=Skv 分块后序列），`usedCoreNum` 决定核数。
  level4 语义下任务轴 = **(B_eff × n_heads × (G 的 query 分块))**。
- 尾核：Skv 分块循环在核内展开（flash 外循环进循环，不拆核上分块）——与 Triton persistent-core 范式同构。

## 4.1.2 单核内切分

- S1（Skv 分块）：主切分轴，UB 允许进 block；S2（query 分块）：任务轴。
- level4 窗口 G 小则整窗进核；G 大（>~512）须 S2 分块防 UB overflow。

## 4.1.3 tiling（kernel 内部）

- flash 外循环 S1 步长 = key 参与在线 softmax 的块长；循环内维护 running max / running sum，块末按
  exp(max_old−max_new) 补偿（FA设计介绍.md §1 步 4）。
- **本卡 Triton 骨架的 S1 语义 = 窗口内 key 分块长（BLOCK_N），flash 外循环。**

## 5 片上内存（CANN FAS 参考，P2 预算依据）

- FAS kernel 各 buffer：Q/K/V 分块（L1/low-UB）、ub_attention_out 双 buffer 缓存 running out、
  flash_sum (B_eff,n_heads,G)，softmax_max/sum [B,N,S,8] 补位布局（manifest 已注）。
- 本卡 UB 预算以 Triton 表达（backend_facts/hardware_platforms.md：Ascend910B A2 192KB，950 248KB）；
  峰值占用 = q/k/v 分块 × buffer 份数 + attn tile(fp32)。

## 6 流水编排（CANN FAS 参考）

- FAS CV 流水：Cube（QKᵀ、PV 两 matmul）× Vector（scale+mask 加、exp、max/sum 规约、除 sum），
  double buffer 掩盖（`multibuffer`/num_stages）。Triton 侧对应 `tl.dot` + 矢量 op + num_stages，
  **标记 ⚠️ 部分可达**（num_stages 在 Triton-Ascend 运行时不生效，C-006）。

## 6.x 计算流程（flash 重构，对齐 FA设计介绍.md §1 步 1~4）

步 1: attn_blk = dot(Q 块, Kᵀ 块) * scale；+= pos_bias/mask 切片；   [Cube+Vector]
步 2: 对比 running max：m_new = max(m_old, rowmax(attn_blk))            [Vector 规约]
步 3: p_blk = exp(attn_blk − m_new)；α = exp(m_old − m_new)             [Vector]
步 4: o_acc = α·o_acc + dot(p_blk, V 块)；l_acc = α·l_acc + rowsum(p_blk) [Cube+Vector]
循环完成: out = o_acc · (1/l_acc)                                        [Vector 除]

## 精度策略

- 中间 attn 与 o 累加 fp32（对齐 level4 `qh.float()/out_heads.float()` 重算与 FAS 的 fp32 累加）。
- bf16 输入 dot 用 fp32 累加；fp32 输入 dot 需 `input_precision="ieee"`（C-023，operator_class_patterns §4.2 [EXP]）。

## 测试设计

- 标杆：单标杆 = level4 `definition.json reference`（torch）。workload 50 case（见 manifest 形状表）。
- P2.5 smoke：选取通代 torch golden 的代表形状（非 feature 小窗、feature 多窗、含 mask/pos_bias、fp32 输入、
  大窗口 flash 路径）逐一对比。

## 缺省项与语义差（P1 已确认）

1. 窗口分组 host 侧 layout；kernel 线性 offs 折算。
2. level4 mask 加性 vs FAS 布尔 select——以 level4 为准。
3. feature+mask/pos_bias reference 缺陷——本卡以窗口局部切片为契约。
4. scale 施加位置对齐点：统一 QKᵀ 后乘。
5. 大窗须 flash 重构。
6. fp32 dot ieee 契约。