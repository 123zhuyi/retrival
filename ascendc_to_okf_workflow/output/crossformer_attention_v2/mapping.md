# P2 语义转化映射 — CrossformerAttention（v2，几何绑定 CANN host 档位阶梯）

> 决策表：level4/FAS 事实 → Triton 决策 → 可达性 → 依据锚点。核心变化 vs v1：**BLOCK 几何不再盲拍，
> 绑定 CANN arch35 host 档位阶梯**（`CalcDBasicBlock`/`CalcS1S2BasicBlock`），并修正原骨架
> **硬编码 BLOCK_D=128 ⇒ 单头大 C 会算错**的缺陷。全部点落地 `triton_skeleton.py` 并经 P2.5 ①②门验证。

## §1 分核策略 → grid

| level4/FAS 事实 | Triton 决策 | 可达性 | 锚点 |
|---|---|---|---|
| FAS tiling 任务轴 B×H×S2，usedCoreNum 分核 | 任务=(NW·n_heads·cdiv(G,BLOCK_M))；grid=min(NTOT, num_core)，核内 grid-stride | ✅ | extraction §4.1 |
| CV 融合算子取 Cube 核数 | grid=num_aicore（本机 20 保守） | ✅ | C-011 |
| 非对齐小窗铺不满核 | 登记 ⚠️（固有上限） | ✅ | README P2 |

## §2 tilingdata → BLOCK（v1.5：绑定 host 档位阶梯）

| CANN host 事实（`arch35/flash_attention_score_tiling_basic.cpp`） | Triton 决策 | 可达性 | 锚点 |
|---|---|---|---|
| **D-axis 分段 `CalcDBasicBlock`(:51)**：`AlignUp(D+rope,64)`→{64,128,192,256}/768 | **BLOCK_D = ≥head_dim 最小 2 幂**（D-tier Triton 化；head_dim≤128→128；192→256；256→256） | ✅ | extraction §4.1.3 |
| **S1/S2 主档 `CalcS1S2BasicBlock`(:79)**：默认 128/128（level4 主流④）、D>256 fp32→64/128 | **CANN 128/128 需 L1 3-buffer+D-workspace ⇒ Triton 不可直接复刻**（UB 溢出实证）→ 取 64/128 近邻（head_dim≤128） | ⚠️ 两态 | p25_report ① |
| D>128 时 fp32 O 累加器撑爆 UB（requires 1587200 vs 1572864 bits） | **BLOCK_D>128 ⇒ 自动降 BLOCK_M=32/BLOCK_N=64**（逃逸验证 rmse=5e-8）；D>256 需 D-split（登记） | ⚠️ | p25_report |
| head_dim 非 2 幂（96/192） | BLOCK_D 上浮 2 幂 + d<HEAD_DIM mask | ✅ | 骨架 |

## §3 片上内存 → 预算校核（v1.5 重新硬算）

**head_dim≤128（primary）**：q 64×128×2=16KB + k/v 各 16KB + s/p 各 64×128×4=32KB +
o_i 64×128×4=32KB ≈ **112KB ≈ 60%**（≤192KB，富余，P2.5 过）。
**head_dim=192/256（大 D）**：BLOCK_M=32/N64/D256 → q 32×256×2=16KB + k 256×64×2=32KB +
v 64×256×2=32KB + s/p 各 8KB + o_i 32×256×4=32KB ≈ 128KB（过）；BLOCK_M=64 则
o_i64K+s32K+p32K+... ≈ **超 192KB（UB overflow，p25 实证——故自动降 M）**。

> 结论：**fp32 O 累加器 = BLOCK_D 主导项**；大 D 靠降 BLOCK_M 平衡。CANN 128/128 经 L1 3-buffer
> 获得额外 UB，Triton 无此机制 = L2 边界（v1.5 归责）。

## §4 流水编排 → 可达性三态

| level4/FAS 事实 | Triton 决策 | 可达性 | 依据 |
|---|---|---|---|
| FlashSoftmax 在线重构+exp 补偿链 | tl.dot + loop-carried tl.maximum/tl.exp/tl.sum + α=exp(m_old−m_new) 手动维护 | ⚠️ 部分可达（正确性补足） | FA设计介绍.md §1 步1~4 |
| double buffer 掩盖 | num_stages **运行时不生效（C-006）** | ⚠️ 性能风险 | C-006 |
| 910b regbase MIX_AIC_1_2 深流水/跨核 flag | **不可移植**（L2 排除清单） | ❌ | extraction §6 |
| 窗口分组重排（host 布局） | wrapper `idx`/`inv` gather | ✅（不在 kernel） | extraction §6 缺口 |
| scale 施加 | `s=dot(q,k)*scale`（QKᵀ 后乘） | ✅ | manifest 缺省项 4 |

## §5 Compute → 骨架（关键）

| 元素 | Triton 表达 | 备注 |
|---|---|---|
| QKᵀ | `_dot_new(q,k,IS_FP32)` fp32 | fp32 ieee；非 fp32 缺省 hf32 |
| scale | `s=s*scale` | runtime 标量（C-013） |
| pse+mask | bias tensor `(n_heads,G,G)` 全窗共享，s+=bias | 加性、softmax 前 |
| key 尾列 | `s=tl.where(n_ok,s,-inf)` | **关键**（防假质量）P2.5 生效 |
| softmax | flash 在线：m_new=max; p=exp(s−m); α=exp(m_old−m_new); l=αl+Σp | C-016 累加器全 False load 产零/产−inf |
| P·V | `_dot_new(p_cast,v,IS_FP32)`；**V 加载方向=(BLOCK_N,BLOCK_D)**（key×feature） | **V 方向 bug**（原 P2.5 ①门抓出，保留） |
| 归一+写回 | `o/l`；逆排列 `o[_inv]`（feature） | layout 对齐 |
| **BLOCK_D 动态** | `bd=next_pow2(head_dim)`；bd>128 时 M/N 自动降 | **v2 修复** 原硬编码 128 缺陷 |

## §6 不可达 / 降级点

- ❌ **CANN 硬件原生在线 softmax / regbase 深流水**：Triton 用 tl.dot+循环复刻（数学等价），无原生。
- ❌ **窗口分组的 ND→NZ/Transpose（swin 量化）**：不做（host layout）。
- ❌ **CANN D-workspace（大 D bmm2 落 GM）**：Triton 无对应 → **head_dim>256 需自建 D-split（登记边界）**。
- ❌ **CANN 128/128 大档位**：L1 3-buffer 支撑，Triton UB 复刻溢出 → 64/128 近邻（L2）。

## P2.5 ①②门实测汇总（[EXP-skeleton], Ascend910b/9372 + CANN 9.1.0）

| shape | dtype | BLOCK | rmse | 判定 |
|---|---|---|---|---|
| G=196 单窗 | f16 | 64/128/128 | 1.74e-5 | PASS |
| feature 8×14 posmask | fp32 | 64/128/128 | 3.17e-8 | PASS（ieee） |
| G=729 单窗+mask | f16 | 64/128/128 | 1.87e-5 | PASS |
| feature 11×7 | bf16 | 64/128/128 | 4.67e-6 | PASS |
| G=7921 大窗 flash | f16 | 64/128/128 | 1.29e-4 | PASS |
| feature 16×14 mask+pos | f16 | 64/128/128 | 6.40e-6 | PASS |
| G=49 小窗 | f16 | 64/128/128 | 1.76e-6 | PASS |
| **hd256（v2 新增）** | f16 | 32/64/256 | 2.69e-8 | **PASS** |

**编译门另实证**：hd192/hd256 在 BLOCK_M=64 时 **UB overflow**（1587200>1572864 bits），
自动多缓冲/code-motion 关闭仍溢出 → BLOCK_M=32 逃逸成功（rmse 5e-8）。→ **回流 backend_facts**
（大 D UB 溢出 + BLOCK_M 为 fp32 O 累加器主导项，新 C 条目）。