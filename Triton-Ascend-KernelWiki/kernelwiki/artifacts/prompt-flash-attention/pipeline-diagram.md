# prompt_flash_attention — 流水编排分析

来源：`docs/PFA算子设计介绍.md`（L9-138）+ `op_kernel/arch35/service/*` + `vf_softmaxflashv3.h`。MTE/BufferID/queue/event/手动 CV 同步属 CANN 专有，仅抽象为 tile 级语义。

## 1. 四阶段 CV 流水（与训练 FA 同骨架）

PFA 按标准 FlashAttention 正向流程（设计介绍 L9-22）：

| 阶段 | 单元 | arch35 入口 | 操作 | 数据依赖 |
|---|---|---|---|---|
| Cube1 (Bmm1) | AIC | `service/prompt_flash_attention_mm1.h` | Q·K^T → attention_score（FP32 累加） | Q, K |
| Vector1 (Vec1) | AIV | `service/prompt_flash_attention_vector1.h` | pse add/mul → ×scale → atten_mask select(→负极小) → FlashSoftmax(在线 max/sum) | Bmm1 结果, pse, mask |
| Cube2 (Bmm2) | AIC | `service/prompt_flash_attention_mm2.h` | P·V（P 为 softmax 结果） | Vec1 结果, V |
| Vector2 (Vec2) | AIV | `service/prompt_flash_attention_vector2.h` | rescale(÷softmax_sum) → 可选 dropout/quant → 写回 attention_out | Bmm2 结果 |

依赖链：`Bmm1 → Vec1 → Bmm2 → Vec2`。PFA 是**推理/prefill**，无反向，故不输出 softmax_max+softmax_sum（训练 FA 那套），而是输出 `softmaxLse`（kernel 入口 `prompt_flash_attention.cpp` L36）供下游增量解码复用。

## 2. 在线 FlashSoftmax（可迁移的数学语义）

FlashSoftmax 对 Skv 轴切分，在线刷新（设计介绍 L13-22）：

```
for i in range(SkvSplits):                       # Skv 轴切分
    s_i = QK[:, s2_start:s2_end] · scale          # 本分块 attention_score
    s_i = mask_select(s_i, atten_mask)            # true → -inf
    m_ij = row_max(s_i)
    s_i = s_i - m_ij
    p_i = exp(s_i)
    l_ij = row_sum(p_i)
    if i == 0:
        m_i = m_ij;  l_i = l_ij;  acc = p_i · V_i
    else:
        alpha = exp(m_i - m_ij)                    # rescale factor  (设计介绍 L17: exp[i]=e^(max[i-1]-max[i]))
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + p_i · V_i              # PV 累加器按 alpha 缩放后累加新分块 (设计介绍 L20)
        m_i = max(m_i, m_ij)
attention_out = acc / l_i                          # 行除 softmax_sum (设计介绍 L22)
softmaxLse = log(l_i) + m_i                        # 推理输出 lse（needs_verification: 精确公式以源码为准）
```

公式 `exp[i] = e^(max[i-1] - max[i])`（设计介绍 L17）与 PV 累加 `MM[PV][i] = MM[PV][i-1]·exp[i] + MM_new[PV][i]`（设计介绍 L20）是**可直接迁移**的 online softmax。

## 3. CV 流水编排（CANN 专有，抽象为“计算搬运重叠”）

设计介绍 L119-138：

- **Cube 双发机制**：提前发射两块 Cube 计算，Cube1/Cube2 衔接提升 Cube 利用率。
- 两场景：
  - C 耗时 > V 耗时：Cube 搬运掩盖 V 计算 → MTE2/Cube bound。
  - C 耗时 < V 耗时：C 提前发射喂 V → V bound。
- **V 侧 double buffer**：ping-pong 两片独立内存，DataCopy 与 Clc 交替（设计介绍 L99-117）。

Triton 抽象：tile 级数据复用 + fused load/store + 减少 GM 中间写回；**不**迁移 BufferID/queue/event/手动同步。

## 4. UB 内存区域（arch32/35）

Vector 侧 FP32 中间（设计介绍 L59，最大一份 32KB）：

| UB 区域 | 用途 |
|---|---|
| mm1ResUb (128×128, FP32) | QK 结果 |
| softmaxMaxUb / softmaxSumUb / softmaxExpUb | FlashSoftmax 行 max/sum/exp（FP32） |
| mm2ResUb | PV 结果（FP32） |
| pseUb / attenMaskUb | 位置编码 / 掩码 |
| apiTmpBuffer | Softmax/Dropout API 临时 |
| antiquant 相关 | 反量化 scale/offset（quant 场景） |

arch35 另有 `inputQue`/`outputQue`（Cube 队列）与 workspace 分区偏移（`needs_verification`：精确字节预算以源码/平台为准）。

## 5. Workspace 布局

`workspace = sysWorkspace(16MB, 910B) + coreNum × (softmaxLse/max/sum + mm1Res×2 + mm2Res×2)`（tiling.cpp L465-490）。`×2` = double buffer。PA 场景额外 kvMerge；MLA 场景额外 qPre（含 rope）。

## 6. PFA 相对训练 FA 的流水差异

| 差异 | 训练 FA | PFA |
|---|---|---|
| 场景 | 训练前向（含反向中间量） | 推理 prefill（全序列） |
| 输出中间量 | softmax_max + softmax_sum（供反向） | softmaxLse（供增量解码） |
| KV cache | 无 | `vf_flashupdate.h` 写回路径（`needs_verification`：prefill 是否同时写 cache） |
| PagedAttention | 无 | blockTable 驱动（MatMulType MM_PA） |
| 量化 | FP8(arch35) | deq_scale1/2 + quant_scale1/2 + antiquant + MSD |
| MLA | 部分 | D=576(nope512+rope64)，MatMulType MM_IFA_MLA |
| 共享前缀 | 无 | keySharedPrefix/valueSharedPrefix |
| RoPE 解耦 | 部分 | queryRope/keyRope 独立输入 |

## 7. 阶段→Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| Bmm1 QK | Cube mm + L0C FP32 累加 | `tl.dot(q, k.T)` in FP32 accumulator | 是（数学） |
| mask | atten_mask select → -inf | `tl.where(mask, -inf, s)` | 是 |
| FlashSoftmax | 在线 max/sum/exp + PV rescale | running max/sum + `exp(m_old-m_new)` rescale | 是（算法） |
| Bmm2 PV | Cube mm | `tl.dot(p, v)` | 是（数学） |
| rescale 写回 | ÷softmax_sum | `o = acc / l_i` | 是 |
| CV 流水/double buffer | MTE/queue/event/双发 | tile 复用/fused load-store（编译器调度） | 否（机制） |
| ND↔NZ | DataCopyPad 格式转换 | Triton layout（编译器） | 否（机制） |
