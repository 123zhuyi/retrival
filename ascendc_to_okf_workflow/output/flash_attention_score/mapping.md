# P2 语义转化 — flash_attention_score（Ascend C → Triton，服务 level4 算子 SelfAttention）

> 输入：extraction.md（11 章，2026-09-21）；manifest.md（CV 融合判定，非低价值，P2.5 四门全跑）。
> 骨架：`triton_skeleton.py`（`_sa_attn_fwd_kernel` + wrapper + golden + smoke）。
> 每条 P1 事实在此有去向（转化/合并/显式丢弃+理由），不允许静默丢失。
> 可达性三态：✅ 直接可达 / ⚠️ 部分可达（标「正确性风险|性能风险」分级）/ ❌ 不可达（含降级写法+预期影响）。
> 后端论断取证顺序：backend_facts → zh → IR；本文件引用均已 grep 核对卡内实际内容（见 P2 质量门 4）。

## 0. 总体转化决策

**level4 SelfAttention = 4×投影(对齐 MatMul) + 注意力核；CANN 参考只覆盖注意力核。转化为「注意力核 → 1 个 Triton flash-attn forward kernel，**S1/S2 分块在线 softmax（主，8_SelfAttention 100% 大 S）**；小 S 副分支收缩为全 S 单块 full-softmax；4 段投影 → host GEMM（显式标注：CANN 参考不含它，level4 附加参数层）」+ 输出一次 RTNE 写回。**

论证链：
1. CANN `FlashAttentionScore`（arch32 `_s1s2_bn2gs1.h`）是无投影的纯注意力核：bmm1(QKᵀ·scale) → online softmax(max/sum 状态) → bmm2(·V)，CV 融合 `MIX_AIC_1_2`（extraction §6/§1.1）。它的输入就是 q/k/v，level4 的 4 个 `nn.Linear` 在 kernel 外。
2. level4 golden（definition.json `run()`）**显式 `q.float()`** 再 matmul——语义要求全 fp32 计算域（乘法+累加均 fp32，非 fp16 乘 fp32 累加）。为精确对齐 golden，Triton 侧统一 **load 后 promote fp32 + `tl.dot(...,"ieee")`**。
3. **level4 SelfAttention 无 causal/pse/mask/dropout/varlen**。**⚠️ 包络更正（本次换 8_SelfAttention 主目标）**：主目标 workload 为**统一大 S**——**B∈[1,8]、S∈[512,5182]、d_model∈[160,8192]、head_dim∈[32,256]**（100% S>64）。**big-S（主）**：S1(query/S1 轴) 与 S2(key/S2 轴) 分块 + online softmax（running max + α 补偿），即 CANN `_s1s2_bn2gs1.h` 与 crossformer_attention 已验证形态，见 §4/§5 与 P3 卡 §5/§8；head_dim∈{144,224,256} 超 D 模板上限 128 → Triton 动态 D（BLOCK_HD=next_pow2）。**small-S（副分支，archive `SelfAttention_smallS`）**：整条 K 轴塞单 tile → 单块 full-softmax（P2.5 初轮已实测）。**统一一 kernel 参数化覆盖 full-softmax↔在线两态**（主目标全走在线）。
4. 投影（host GEMM）+ 注意力核（Triton）的拆分依据：CANN 参考即为纯注意力（`src/flash_attention_score` 无投影逻辑，manifest 材料归集已证）；投影是平凡 GEMM + 跨 head 参数层，并入注意力核会破坏 head 分组清晰性且无 CANN 参考锚点。**代价**：生成的 Triton 算子"注意力核是主体，投影可并列标准 tl.dot GEMM"——在 P3 卡片标注"投影可选并入"，本题 P2.5 用 host GEMM + Triton 注意力核的端到端复现校验注意力核正确性（两路投影用同一份 seed=42 权重 → smoke 精确隔离注意力核）。

## 0.5 检索记录（P2 六任务，全部命中 backend_facts 首层）

| # | 检索任务 | 命中 | 结论 |
|---|---|---|---|
| ① | 骨架各 tl op lowering 分级 | `tl_op_lowering.md` §2 | load/store/arange/expand_dims/指针算术=T1；dot/max/sum/exp/broadcast/where=linalg T2；**无 T3/T4**（无 scan/gather/device_print） |
| ② | ⚠️/❌ 判定依据 | `backend_contracts.md` C-002/006/010/011/013/016/018/019/023/026 | 全部 grep 核对原文后引用（见各节依据列）；CV 流水/position/手排双缓冲 ❌ 降级见 §5 |
| ③ | 编译选项 | `compiler_options.yaml` + `operator_class_patterns.md`:120 | 骨架无 num_stages/num_warps（C-006 运行时不生效 [EXP]）、无 autotune（C-007 µs 噪声 [EXP]）、multibuffer 编译期默认开 |
| ④ | UB/对齐预算 | `hardware_platforms.md` §1/§3 | 950 UB 248KB 可用；C-010 尾轴 32B(VV)/512B(CV)；核数运行时查询禁写死 |
| ⑤ | dtype 陷阱 | `dtype_support.md` §3/§4 | promote fp32 计算域（§4 累加精度）；store/load mask 整型（C-002）；int64 仅标量块基址（4e） |
| ⑥ | 契约违禁写法 | `backend_contracts.md` preflight | 无 `%`（C-001）；无 tl.zeros 初始化（C-016，用全 False mask load 产零+显式 .to(fp32)）；无 gather（C-018，K/V 镜像 2D 仿射 load）；环境红线 TORCH_DEVICE_BACKEND_AUTOLOAD=0（C-019） |

---

## 1. 分核策略 → grid 设计

| Ascend C 事实（extraction 锚点） | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| 分核轴 B×N(head)×G×S1，S2(key) 内循环（`_s1s2_bn2gs1.h` 命名/§4 表；GetS1LoopRange:179 GetS2LoopRange:180） | **注意力核任务轴 = B·H·cdiv(S,BLOCK_M)**（query/S1 分块为任务,key/S2 进内循环在线 softmax,见 §4/§8 骨架）。任务空间 `task = pid`，`bh=task//num_m; mblk=task−bh·num_m`（无 `%`，C-001） | ✅ | S1(query 块) 是输出驻留轴,天然任务轴；S2(key) 进 flash 内环做增量 softmax——与 CANN `_s1s2_bn2gs1.h` 的 S1 外/S2 内同构（§0 论证 3） |
| 主形状：B≤8、H∈[3,64]、S 至 5182、head_dim≤256 | grid = `min(B·H·cdiv(S,BLOCK_M), np)`，persistent grid-stride（`for task in range(pid, NTOT, np)`），`np=num_aicore` 运行时查询禁写死 | ⚠️ **性能风险**（大 head_dim 256 + BLOCK_M 小 → 任务多,铺核开销；小 head 任务多 byte 率低） | C-011 [EXP+DOC]；hardware_platforms §3（950 核数跨度大）；UB 大 shape 降 BLOCK_M/N 控峰（§6） |
| 950 Cube 4~36 核 | `np=triton.runtime.driver.active.get_device_properties(0)`（须在 active 后端管理下，C-026） | ✅ | C-011；hardware_platforms §3；除 C-019 红线 |

> **P2.5 备注（初轮版本 grid=(1,) 串行 head 循环）**：初轮骨架为小 S 单 tile 形态，采用 grid=(1,) 串行 head 循环、全 S 单块。**主目标已换 8_SelfAttention（100% 大 S）后不可再用串行单块**——必须回到本 §1 统一 tiled 形态 grid=min(B·H·cdiv(S,BLOCK_M), np) + persistent grid-stride + key/S2 内环在线 softmax（见 §8 骨架）。小 S 单块仍为合法副分支（archive `SelfAttention_smallS`）。**head 分块必须 host 侧提供行主序连续张量，见 §4/§5 连续性契约。**

## 2. tilingdata → BLOCK 超参

| tilingdata 字段（extraction §4 锚点） | 语义 | Triton 决策 | 可达性/推导 |
|---|---|---|---|
| S1TemplateNum ∈{0,64,128,512} | S1(query) 轴分档 | **BLOCK_S1: tl.constexpr**（query 块,任务轴 `cdiv(S,S1)`）；大 S 主走 S1 分块（§1/§8） | ✅ 主目标 100% 大 S |
| S2TemplateNum ∈{0,128,256,512} | S2(key) 轴分档 | **BLOCK_S2: tl.constexpr**，`for s2 in range(0, S_block, BLOCK_S2)` K 内环 + 在线 softmax（running max/α）（主） | ✅ 与 `_s1s2_bn2gs1.h` 同构；小 S 副分支单 K 块 |
| DTemplateNum ∈{0,64,128,256} | D 轴分档 | **BLOCK_HD: tl.constexpr = next_pow2(head_dim)**（hd∈[32,256]→{32,64,128,256}） | ✅ wrapper 按 head_dim 选；hd>128 动态 D（CANN DTemplateType=0,144/224/256 不命中 D 档）；≥32 满足 C-010 |
| bmm1/bmm2 的 Cube C 尺寸 | 阶段输出 tile | scores [BLOCK_S1,BLOCK_S2] fp32；o acc [BLOCK_S1,BLOCK_HD] fp32（跨 s2 内环累加） | ✅ S1S2 分块 + 在线 |
| coreNum/blockOuter | 占用核数 | grid=min(B·H·cdiv(S,BLOCK_S1), np) | ✅ 见 §1 |
| dSize 对齐（dSizeAlign16 等，:187-190） | kv 尾轴 16 对齐 | BLOCK_HD 恒 next_pow2（32 的倍数）满足 32B（C-010）；store `mask=others<HD` | ✅ |
| scale_value 属性 | 缩放 | `scale=1/tl.sqrt(HD)`（fp32 标量） | ✅ level4 语义 `1/√head_dim`（definition.json） |

BLOCK 推导与 UB 静态上界（[SRC] 950 UB 248KB；单 program 持 BLOCK_S1 行 + o acc，跨 s2 内环）：

| kernel/dtype | 主要 tile（fp32 计算域） | 峰值 | 占 248KB |
|---|---|---|---|
| S1=64×HD256（大 S 最坏,主 shape hd=256） | q 64×256 fp32=64KB；kt/v 各 64×256 fp32=128KB；s 64×64=16KB；o acc 64KB | **≈272KB+** | ⚠️ 贴顶/超 → 降 S1/N=32 或 HD>128 分半 |
| S1=64×HD128 | q 32KB；kt/v 64KB；s 16KB；o 32KB | ≈176KB | ≈71% ⚠️ |
| S1=64×HD64（主 shape hd≤64） | 各半 | ≈112KB | ≈45% 安全 |
| S1=32×HD64 | 减半 | ≈56KB | 安全 |

⚠️ **性能风险（大 S + 大 head_dim 贴顶）**：head_dim∈{144,160,224,256} 或大 BLOCK 时峰值接近/超 248KB，需降 BLOCK_S1/N=32 或对 HD>128 分半；静态上界未含 multibuffer 多份。小 S 副分支（S1 64）落在安全区（P2.5 初轮已过）。

## 3. tilingkey 模板分支 → constexpr 开关

| tilingkey 位段（extraction §4/§6 锚点） | 分支语义 | Triton 处理 | 可达性/理由 |
|---|---|---|---|
| bmm1Format ND/NZ（_s1s2_bn2gs1.h `bmm1Nz` :104-105） | QKᵀ Cube 格式 | **免 ND/NZ 概念**：寄存器直通 tl.dot | ✅ 无 NZ 格式映射需求（同 grad §6） |
| InputDType {FP32,BF16,FP16}（level4 三 dtype） | 输入实例化 | 指针 dtype 自动分派 + **统一 promote fp32 进 dot**（`load().to(tl.float32)`，dot `"ieee"`） | ✅ dtype_support §4（fp32 计算域）；C-023② 用 wrapper 传 constexpr 而非 kernel 内 dtype 分支 |
| IsDrop/IsPse/IsAttenMask（bits） | dropout/pse/外部掩码 | 丢弃（level4 无） | §8 |
| IsTnd/actual_seq_*（varlen） | 变长 | 丢弃（level4 规整 S） | §8 |
| softmax max/sum 双量 + SOS 在线修正（extraction §7） | 大 S 稳定性工程 | **映射为手动在线 softmax**：`running max + α=exp(m_old−m_new) + l=l·α+sum(exp(x−m_new)) + o=o·α+p·V`（跨 s2 内环） | ✅ 数学等价（crossformer 同法）；fp32 域无稳定性风险；小 S 副分支单 K 块退化为 max 减+exp+sum+dot/l |
| S1S2 模板组合（S1/S2/DTemplateType 变体） | 语义×调度矩阵 | **保留并映射**：S1TemplateType→BLOCK_S1、S2TemplateType→BLOCK_S2、DTemplateType→BLOCK_HD（三个 constexpr + 在线 softmax）；不"丢弃"——主目标 100% 大 S 必走此路径 | ✅ §2 之一 |
| atomics/排序写回（跨核累加） | 大 S 多块 → 部分和 | **不需要**：每个 S1(query) 块任务独立持其输出行(s2 内环为其部分和),每输出元素一次写回 | ✅ 结构性更优（同 grad §0） |

## 4. 片上内存 → 隐式化 + UB 预算校核

| Ascend C buffer（extraction §5 锚点） | Triton 等效 | 去向 |
|---|---|---|
| bmm1 (QKᵀ→scores, Cube) `:102` / bmm2 (·V) `:119` | 寄存器 tile：scores [BLOCK_S,BLOCK_S]、o [BLOCK_S,BLOCK_HD] | ⚠️→编译器分配（multibuffer 默认开 [SRC]；缓冲份数不可见，§2 校核按单份上界） |
| stage1Ping/1PongBuf + stage2TBuf（双缓冲，:212-214） | **无**（单 tile，无多块 ping-pong） | ✅→不需要（§0 论证 3） |
| softmaxSumBuf[2]/softmaxExpBuf[2]/softmaxMaxBuf（:215-218） | `m`,`l` 一维寄存器 [BLOCK_S]（`tl.max`/`tl.sum` axis=-1） | ✅ 隐式化；无 max/sum 双量 GM 协议（单 tile 就地） |
| maskTBufPing/Pong、pseTBuf（:209-211） | 无（level4 无 mask/pse） | §8 |
| mm1Res[2]/mm2Res[2]/vec2Res[2]/stage1Res[2] GM（:222-225） | **无 GM 中间量**（单 tile 全 UB，输出直写） | ✅→不需要 |
| workspace（tiling 约定） | 无 | ✅→不需要（无跨核部分和） |

- loop-carried 进位/状态初值本算子**无**（单 tile，非累加循环）；C-016（禁 tl.zeros 初值）不触发，但产零/掩码统一 `other=0.0` 走 C-002/C-018 合规写法。
- 对齐：BLOCK_HD∈{32,64,128}、BLOCK_S∈{16,32,48?→64} 行宽均 ≥32B（C-010）；store mask 整型（C-002）。

## 5. 流水编排 → Triton 可达手段

| Ascend C 手段（extraction §6 锚点） | 可达性 | Triton 写法 / 降级与预期影响 |
|---|---|---|
| Cube bmm1 ↔ Vector softmax ↔ Cube bmm2 CV 换手（`MIX_AIC_1_2` :379；`WaitBmm1Result`/`WaitBmm2Result` :129-130） | ❌ **不可达**（CV 异构 flag 同步，Triton 无编程模型） | **降级：编译器隐式流水**——`load(q,k,v)`→`dot`→`max/exp/sum`→`dot`→`store` 序列由 triton-to-linalg 调度 + multibuffer 做软件流水。预期影响：达不到 CANN 手工 ping-pong 深度；但单 tile 无内层循环，流水收益本就有限，P2.5 pipe 诊断量化 |
| IterateBmm1/Bmm1SetTensorA*（:140-153） | ✅（数学 1:1） | `scores=dot(q_blk.to(f32), k_t.to(f32))` |
| SoftMaxCompute/SoftMaxCheckResCompress/InvalidLineSplitS2Process（:168-171） | ✅（单 tile 简化） | `m=tl.max(scores,1); p=exp(scores−m[:,None]); l=tl.sum(p,1)` |
| IterateBmm2→Bmm2ResultMul/Div→Bmm2DataCopyOut/SoftmaxDataCopyOut（:131,:173-176） | ✅（合并 + 归一） | `o=dot(p,v); o/=l[:,None]; store`（归一化并入，无需在线比例） |
| arrow：`bmm1` 结果 scale 乘 | ✅ | `scale` fp32 标量乘 scores（对齐 CANN scale_value） |
| nopython/loop 多块 | ❌→不需要 | 单 tile，无 S 循环（§0） |

### 连续性契约（P2.5 ②门后端发现，host 侧必守）

kernel 用平坦 `base + rows*HD + hdc` 行主序寻址，**要求入参为行主序连续 [BH,S,HD]**。
wrapper 的 `.view(B,S,H,HD).transpose(1,2).reshape(BH,S,HD)` 在 **B=1**（或个别 batch）时返回
**非连续交错视图**（实测 stride=(24,72,1)，实为 [S,H,HD] 布局），kernel 读错偏移 → 静默垃圾。
B≥2 reshape 恰好给出行主序才「偶然全对」。**修复/契约**：`.contiguous()` 强制 [BH,S,HD]。此属
host torch 层契约（kernel 寻址自洽），非后端缺陷（不新增 backend_contracts，映射记载）。

## 6. Compute → 骨架逐指令映射

tl op 分级：全 T1/T2（§0.5 ①），无 T3/T4。

| 步（extraction §1.1 / level4 golden） | CANN 实现锚点 | Triton 翻译（triton_skeleton.py 行号） | 可达性 |
|---|---|---|---|
| ① per-(b,h) **load Q/K/V** 单 tile | bmm1 a1Type/b1Type GM ND（:98-99） | `q=load(base_q+bh*S*HD + rows[:,None]*HD + hdcols[None,:], mask, other=0).to(f32)` 及其 K/V 镜像（:kern 行） | ✅ C-018 合规：K/V 以 [HD,S] 镜像 2D 仿射 load（rows=hdcols、cols=rowsS），非 gather |
| ② **scores** = QKᵀ·scale | bmm1 Cube（:102, Std 转置由 NZ/镜像承担） | `scores=dot(q_blk[S,HD], k_t[HD,S])*scale`（:–） | ✅ fp32 ieee dot；scale fp32 标量 |
| ③ full-**softmax** | SoftMaxCompute online max/sum（:168） | `m=amax(scores,1); p=exp(scores−m[:,None]); l=sum(p,1)`（:–） | ✅ 单 tile max 减数值稳定；无 SOS 重标度需要 |
| ④ **o** = p·V | bmm2 BatchMatmul（:119） | `o=dot(p[S,S], v[S,HD])`（:–） | ✅ |
| ⑤ **归一 + 写回** z_h | Bmm2ResultDiv + SoftmaxDataCopyOut（:173-176） | `o/=l[:,None]; store(o, out+…, mask, .to(out_dtype))` | ✅ 一次 RTNE 回写；输出行行尾 mask |
| 投影 qkv=wXᵀ / out=w⊺ 4 段 | kernel 外（level4 附加，CANN 无） | host `torch.matmul(x, Wq.t())` fp32（wrapper） | ⚠️ **标注**（非 kernel）：平凡 GEMM，P3 可并入 tl.dot 版本；本题 host 复现以隔离注意力核正确性（§0 论证 4） |

## 7. 精度策略 → 标杆与验收

| 项 | 决策 | 依据 |
|---|---|---|
| 计算域 | **fp32 全程**：投影 host fp32；q/k/v promote fp32 → scores/p/o fp32；唯一舍入点 = 末段 store `.to(out_dtype)`（RTNE） | level4 golden 显式 `.float()`（definition.json）；dtype_support §4 |
| scale 落点 | `1/√head_dim` fp32 乘 scores | definition.json；extraction §1.1 |
| 标杆 golden | `golden_run`（triton_skeleton.py）= definition.json `run()` 逐字移植（fp32；seed=42 权重 | activation 逐例重算） | level4_target/SelfAttention/definition.json；extraction §9 |
| 容差（按 dtype 分档） | **fp32**：max_abs≤1e-3 且 bad_ratio≤1e-3（ieee dot 无精度借口）；**f16/bf16**：CANN 口径 | element 差 |a−b|>max(0.005·max(|a|,|b|),2.5e-5)，bad_ratio>0.005 判挂 | extraction §9.2（对齐 grad 骨架已实现口径） |
| 投影参数一致 | 两路（golden 与 Triton 路径）共享**同一份** seed=42 Wq/k/v/o 设备张量 → 隔离注意力核误差 | §0 论证 4 |

## 8. 显式丢弃 / 不涉及清单（对应 extraction §10 + 转化过程丢弃项）

| 项 | 处理 | 理由 |
|---|---|---|
| pse（real_shift/pse_type/IsPse）、atten_mask、padding_mask、dropout（keep_prob/drop_mask/seed）、sparse_mode 0~8（level4 无 causal 档） | 丢弃 | 超 level4（extraction §10） |
| varlen（actual_seq_*/start_idx、IsTnd/Swizzle） | 丢弃 | 超 level4（S 规整） |
| rope、fp8（d_scale、query/key_rope、sink、p_scale） | 丢弃 | 超 level4（extraction §10） |
| 多 layout（BSH/SBH/BSND/BNSD/TND 5 种） | 丢弃 | level4 恒 BSH 连续（manifest diff 表；wrapper assert contiguous） |
| aclnn 多入口 + softmax_max/sum/out 中间产出 | 丢弃（softmax 就地，不建模中间量） | level4 单 run()；extraction §2.2 |
| online softmax 多块重标度 / S1S2 双环 / Cube-Vector flag 流水 | 结构性消失（单 tile） | §0 论证 3 / §5 ❌ |
| 4 投影并入 kernel | **标注候选项**，本题 host | §0 论证 4（P3 卡片标注"可选并入 tl.dot GEMM"） |
| arch22/arch35 平台差异、NY/NZ 格式、workspace、跨核部分和 | 丢弃/不涉及 | manifest；§4/§5 |

## 9. Triton 设计总图

- **kernel 数**：1（`_sa_attn_fwd_kernel`，单 tile full-softmax flash-attn 前向）；无 goroutine loop、无 atomic。
- **grid**：`(1,)` 单 program 串行 head 循环（P2.5 定稿；B·H 并行仍为性能可选路线 §1）。投影 host GEMM 不在 grid。
- **BLOCK**：BLOCK_S=next_pow2(S)（≤64）；BLOCK_HD=next_pow2(head_dim)（{32,64,128}）。constexpr：BLOCK_S×BLOCK_HD×（指针 dtype 自动）。
- **计算**：`load(q,k,v)f32 → scores=dot(Q,Kᵀ)·(1/√hd) → m/l → p=exp(s−m)/l → o=dot(p,V) → store(RTNE)`；末段输出与层归一统一。
- **精度/验收**：fp32 计算域、末段单次 RTNE；golden=definition.json 移植；容差按 dtype；smoke 覆盖 workload 代表面（f32/f16/bf16、HD 档、S 边界）。
- **tl op 分级摘要**：全 T1/T2，无 T3/T4；`tl.exp` 列级「未经后端确认」（preflight 无违禁）。

## P2 质量门自查

1. **P1 11 章覆盖**：§1 数学→§0/§6 对照✅；§2 接口→§6 步①/§8 丢弃✅（投影拆 Host 已登记）；§3 平台 dtype→§3 InputDType 行✅；§4 tiling→§2 BLOCK 表 + §5✅；§5 内存→§4 表✅（Ping/Pong/softmax 状态单 tile 消失）；§6 流水→§5 表✅（CV flag ❌ 降级编译器流水）；§7 计算→§6 表✅（softmax 单 tile 简化）；§8 精度→§7 表✅；§9 测试→§7 容差 + 骨架 smoke✅；§10 超范围→§8 全登记✅；§11 对照表→§0/§6/§7✅。**未覆盖项：空**。
2. **⚠️ 分级核对**（共 3 处，全标级）：§1 grid 铺不满核=性能；§2 最坏 UB 71% 贴顶=性能（①门探包络）；§4 multibuffer 不可见=性能。无正确性风险级 ⚠️（C-002/C-018/C-019 均按契约给确定性写法）。
3. **❌ 核对**（共 2 处，均有降级+预期影响）：§5 CV flag 换手→编译器隐式流水（单 tile 无内环，收益有限）；§5 online softmax BizS2 双环→单 tile full-softmax 结构性消失。另 §6 投影=⚠️（host 标注，非 ❌，P3 可并入）。
4. **引用核对清单**（本会话 grep 核对卡内实际内容）：C-001/002/006/007/010/011/016/018/019/023/026、dtype_support §3/§4、hardware_platforms 248KB/核数、operator_class_patterns :120 multibuffer/:130 fp32 ieee dot、compiler_options.yaml。全部属实；`tl.exp` 列级标「未经后端确认」。
5. **骨架一致性 + 静态自检**：mapping 表与 triton_skeleton.py docstring 逐项一致；对照 golden 符号推演——非 causal × 尾块(S 非 2 幂)/整除 × 三 dtype 全走查：① masked 行/列经 `other=0.0` 对 dot 贡献恒 0；② `exp(scores−m)` 中 masked 行 m=0 → exp(−m)=有限（倍乘 p/归一后贡献 0，因该行 softmax 分母 l 也含该行 0·行，实为整行被 store-mask 丢弃）；③ 单 tile 无需在线重标度，无 max 漂移(fp32)；④ projection host 与 golden 同权重，误差隔离到注意力核。未走查遗漏：无。
6. **P1 偏差修正**：extraction 锚点本会话抽查（main 类 :85、Process :95、bmm1 :102/bmm2 :119、SoftMax :168、Bmm1Nz :104）全部属实；无修正。