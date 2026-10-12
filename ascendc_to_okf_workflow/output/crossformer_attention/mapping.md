# P2 语义转化映射 — CrossformerAttention（Ascend C / level4 → Triton）

> 决策表：level4/FAS 事实 → Triton 决策 → 可达性 → 依据锚点。全部点已在
> `triton_skeleton.py` 落地并经 **P2.5 ①编译门 + ②正确性门**（真机 Ascend910+CANN 9.1.0）
> 验证：7 组代表形状对 torch golden 全 PASS，feature 多窗口 case rmse~1e-8。

## §1 分核策略 → grid

| level4/FAS 事实 | Triton 决策 | 可达性 | 锚点 |
|---|---|---|---|
| FAS tiling 任务轴 B×H×S2，usedCoreNum 分核 | 任务 = (NW·n_heads·cdiv(G,BLOCK_M))；grid=min(NTOT, core)，核内 grid-stride | ✅ | manifest P0 / extraction §4.1 |
| CV 融合算子取 Cube 核数 | grid=num_aicore（本机 20） | ✅ | C-011 / operator_class_patterns §0 |
| 非对齐小窗铺不满核，属固有上限 | 登记 ⚠️（BLOCK 初选偏大须降） | ✅ | add 经验（README P2） |

## §2 tilingdata → BLOCK

| 事实 | Triton 决策 | 可达性 | 锚点 |
|---|---|---|---|
| S1（key 分块）/S2（query 任务） | S1→BLOCK_N（flash 外循环步长）；S2→BLOCK_M（任务内 query 分块） | ✅ | extraction §4.1.3 |
| head_dim 非 2 幂（96 等） | BLOCK_D=128 + d<HEAD_DIM mask | ✅ | 骨架 |
| G 多变（7~7921） | BLOCK_M/BLOCK_N 固定 + mask；大 G 走 flash 多块 | ✅ | P2.5 n13f G=7921 通过 |

## §3 片上内存 → 预算校核

| 事实 | Triton 决策 | 可达性 | 锚点 |
|---|---|---|---|
| UB 192KB（910B A2） | fp16 q=64×128×2B=16KB、k/v 各 16KB、attn fp32 64×128×4=32KB ≈96KB<sup>①</sup> + acc 16KB → ≈68%，安全 | ✅ | hardware_platforms.md（A2 192KB） |
| fp32 输入同尺寸自动涨 4× | fp32 时 BLOCK 降档（BLOCK_M=32 基线留 P2.5） | ✅ 两态 | hardware_platforms.md |

①256KB×... 此处为估算，P2.5 ①编译门实测通过未触发 UB overflow，故预算在包络内。

## §4 流水编排 → 可达性三态

| level4/FAS 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| FlashSoftmax 在线重构 + exp 补偿链 | tl.dot + loop-carried tl.maximum/tl.exp/tl.sum + α rescale | ⚠️ 部分可达（正确性补足：α=exp(m_old−m_new) 须手动维护；l 累积显式） | FA设计介绍.md §1 步1~4 / 骨架 |
| double buffer 掩盖 | multibuffer/num_stages | ⚠️ 性能风险：Triton-Ascend 运行时 num_stages 不生效（C-006） | C-006 |
| 窗口分组重排（host 布局） | wrapper 侧 `idx`/`inv` gather（host 数据移动） | ✅（不在 kernel） | extraction §6 缺口 |
| scale 施加 | `s = dot(q,k) * scale`（reference 预乘等价，数学相等） | ✅ | manifest 缺省项 4 |

## §5 Compute → 骨架（关键）

| 元素 | Triton 表达 | 备注 |
|---|---|---|
| QKᵀ | `_dot_new(q,k,IS_FP32)` fp32 | fp32 输入 ieee（C-023 §4.2 [EXP]）；非 fp32 用缺省 hf32 |
| scale | `s = s*scale` | runtime 标量（C-013 两态） |
| pse+mask | bias tensor `(n_heads,G,G)` 全窗共享，s+=bias | 加性、softmax 前 |
| key 尾列 | `s = tl.where(n_ok, s, -inf)` | **关键**（防假质量）P2.5 实测生效 |
| softmax | flash 在线：m_new=max; p=exp(s−m); α=exp(m_old−m_new); l=αl+Σp | C-016：累加器全 False mask 产零，勿 tl.zeros |
| P·V | `_dot_new(p_cast, v, IS_FP32)`；V 加载方向=(BLOCK_N,BLOCK_D)（key,feature） | **V 方向 bug 是 P2.5 ①门实测抓出** |
| 归一+写回 | `o/l`；逆排列 `o[_inv]`（feature） | layout 对齐 |

## §6 不可达 / 降级点

- ❌ **CANN FlashSoftmax 硬件原生在线 softmax 指令**：Triton 用 tl.dot+显式循环复刻（数学等价），无原生 API，性能与向量化由编译器决定。
- ❌ **窗口分组的 ND→NZ/Transpose（swin 量化路径）**：与 level4 无关（host 布局），不做。
- ❌ **Reg 级 gather/scatter（kernel 内窗口重排）**：host 侧 `inv` 一次 gather 完成，kernel 保持连续仿射 load（不触发 C-018）。

## P2.5 ①编译门 + ②正确性门实测（[EXP-skeleton], Ascend910 + CANN 9.1.0 + torch_npu 2.10）

| shape 组合 | dtype | 门①编译 | 门②对 golden rmse | 判定 |
|---|---|---|---|---|
| G=196 单窗 | f16 | ✅ | 1.7e-5 | PASS |
| feature 8×14 窗口+pos_bias+mask | fp32 | ✅ | 3.1e-8 | PASS（ieee 路径） |
| G=729 单窗+mask | f16 | ✅ | 1.9e-5 | PASS |
| feature 11×7 窗口 | bf16 | ✅ | 4.6e-6 | PASS |
| **G=7921 大窗 flash** | f16 | ✅ | 1.3e-4 | PASS（多块在线 softmax） |
| feature 16×14 窗口+mask+pos | f16 | ✅ | 6.4e-6 | PASS |
| G=49 小窗 | f16 | ✅ | 1.8e-6 | PASS |

**测试中抓出的后端/骨架事实（回流）**：
1. Ascend `tl.dot` 输入 `input_precision="tf32"` 被拒——非 fp32 缺省走 hf32，勿写 tf32（映射 §5）。
2. V 在 `p@V` 的加载方向须 `(BLOCK_N, BLOCK_D)`（key×feature），勿沿用 Kᵀ 的 `(BLOCK_D, BLOCK_N)`。
3. flash 在线 softmax 尾列须显式 `-inf`（G<BLOCK_N 时）。

> 此三条已回流共享层（见 backend_facts 维护记录，P2.5 证据纪律第 5 条）。