# incre_flash_attention - 流水编排分析

来源：`docs/IFA算子设计介绍.md`（L12-173）+ `op_kernel/arch32/*` + `op_kernel/arch35/service/*` + `vector_api/vf_softmax_const.h`。MTE/BufferID/queue/event/手动 CV 同步属 CANN 专有，仅抽象为 tile 级语义。

## 1. 四阶段 CV 流水（FlashAttention 正向骨架）

IFA 按标准 FlashAttention 正向流程（设计介绍 L18-32）：

| 阶段 | 单元 | arch35 入口 | arch32 模板内 | 操作 | 数据依赖 |
|---|---|---|---|---|---|
| Cube1 (Bmm1) | AIC | `service/mm1_processor.h` | C+V/preload 模板内 | Q·K^T -> attention_score（FP32 累加）；S1=1 | Q, K |
| Vector1 (Vec1) | AIV | `service/vec1_processor.h` | 模板内 | pse add -> ×scale -> atten_mask select(->负极小) -> FlashSoftmax(在线 max/sum) | Bmm1 结果, pse, mask |
| Cube2 (Bmm2) | AIC | `service/mm2_processor.h` | 模板内 | P·V（P 为 softmax 结果） | Vec1 结果, V |
| Vector2 (Vec2) | AIV | `service/vec2_processor.h` | 模板内 | rescale(÷softmax_sum) -> 可选 quant -> 写回 attention_out | Bmm2 结果 |

依赖链：`Bmm1 -> Vec1 -> Bmm2 -> Vec2`。IFA 是**推理增量解码**，无反向，不输出 softmax_max+softmax_sum（训练 FA 那套供反向的中间量）。

## 2. 在线 FlashSoftmax（可迁移的数学语义）

FlashSoftmax 对 Skv 轴切分，在线刷新（设计介绍 L22-32）：

```
for i in range(SkvSplits):                       # Skv 轴切分
    s_i = QK[:, s2_start:s2_end] · scale          # 本分块 attention_score（S1=1）
    s_i = mask_select(s_i, atten_mask)            # true -> -inf
    m_ij = row_max(s_i)
    s_i = s_i - m_ij
    p_i = exp(s_i)
    l_ij = row_sum(p_i)
    if i == 0:
        m_i = m_ij;  l_i = l_ij;  acc = p_i · V_i
    else:
        alpha = exp(m_i - m_ij)                    # rescale factor  (设计介绍 L27: exp[i]=e^(max[i-1]-max[i]))
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + p_i · V_i              # PV 累加器按 alpha 缩放后累加新分块 (设计介绍 L30)
        m_i = max(m_i, m_ij)
attention_out = acc / l_i                          # 行除 softmax_sum (设计介绍 L32)
```

公式 `exp[i] = e^(max[i-1] - max[i])`（设计介绍 L27）与 PV 累加 `MM[PV][i] = MM[PV][i-1]·exp[i] + MM_new[PV][i]`（设计介绍 L30）是**可直接迁移**的 online softmax。

## 3. 三模板流水差异（CANN 专有，抽象为“计算搬运重叠”）

| 模板 | 编程视角 | CV 流水 | 适用 | 来源 |
|---|---|---|---|---|
| C+V（`split_Bbn2s2_Us2`） | VEC 视角（1 VEC 处理 1 次 matmul 全结果） | 顺序流水 | 绝大多数场景 | 设计介绍 L40 |
| 全 Vector（`allvec_new`） | matmul 由 vector 实现 | 无 Cube 启动/CV 通信开销 | 推理加速卡全用；A2 非 PA/非 GQA/全 FP16 | 设计介绍 L41-45 |
| matmul 基础 API（`preload`） | CUBE 视角（1 次 matmul 切 2 份，2 VEC 各处理 1 份） | **N-Buffer 流水**（连续 N 次某阶段再连续 N 次下一阶段）；CUBE 核内 Buffer 在两个 matmul 间统一调度 | `EnableCubeViewMM` 判定 | 设计介绍 L46-52 |

- **N-Buffer 流水**（设计介绍 L49）：连续执行 N 次某个计算阶段之后再连续 N 次下一个阶段，提升 Cube/Vec 核间流水重叠。
- **Cube 核内流水优化**（设计介绍 L50）：CUBE 核内 Buffer 资源在 FA 的两个 matmul 之间统一调度，使搬运与计算流水更紧凑。

Triton 抽象：tile 级数据复用 + fused load/store + 减少 GM 中间写回；**不**迁移 BufferID/queue/event/N-Buffer/Cube 双发等手动同步机制。

## 4. FlashDecode 规约（IFA 招牌，可迁移）

S2 外切分配到不同核完成 Attention 后，对结果 reduce（设计介绍 L119-138）：

```
# combine 阶段，单核流程（设计介绍 L123-138）
SyncAll()                                         # 核间同步
splits = get_real_splits_of_this_block()          # 本大块实际子块数
lse = load_lse_of_this_block()                    # 各子块的 lse
scale[0:splits] = exp(lse[i]) / Sum(exp(lse[i]))  # 加权系数
res = 0
for j in range(splits):
    res += split_res[j] * scale[j]                # 加权求和
output(res)
```

这是 split-KV attention 的标准 lse 加权规约，**可迁移**为 Triton 两段式（split kernel 输出 partial lse + partial o；combine kernel 加权 reduce）。精确 lse 公式 `needs_verification`（以 `vf_softmax_const.h` 为准）。

## 5. AntiQuant-MSD 算法（反量化优化，rewrite）

经典反量化需对整个 B（KV）矩阵搬入 Vector 反量化，数据量大、性能差。IFA 的 MSD 方案（设计介绍 L140-155）：A（Q）矩阵小，**变换 A 适配 B**：

1. A（FP16/BF16）进入 Vector 展开成多行，每行 `An` 用 int8 存储；
2. 打包成 AA，`CC = AA · B`（int8 × int8 = int32）；
3. 对 CC reduce 得 C。

Triton 抽象：Q 展开 int8 -> `tl.dot(int8, int8, out_dtype=tl.int32)` -> reduce；**不**迁移 CANN Vector 搬运细节。精度边界 `needs_verification`。

## 6. PageAttention 回调拷贝（CANN 专有，抽象为间接 load）

KV block 内存不连续，CANN MatMul 提供回调函数做 B 矩阵拷贝（GM->L1），回调在 Cube 中执行，Vector 设参数后经 GM 通知 MatMul（设计介绍 L156-158）。

Triton 抽象：`tl.load(block_table)` 查物理 block id -> 计算 KV 物理偏移 -> 间接 `tl.load` KV；**不**迁移回调/DCCI/手动通知机制。

## 7. GQA 切分（rewrite）

G = queryHeadNum / KvHeadNum，Vector 上 G 轴切分由当前操作的 UB 大小决定（设计介绍 L160-173）：

```
g = target_ub_size / column_size
if g > G: g = G
process_sub_block(g, column)          # sub_block: g × column
```

Triton 抽象：G 维循环，sub_block 大小由 constexpr 容量决定。

## 8. UB 内存区域

Vector 侧 FP32 中间（设计介绍 L59，PFA 同源，最大一份 ~32KB）：

| UB 区域 | 用途 |
|---|---|
| mm1ResUb | QK 结果（S1=1 × BLOCK_N，FP32） |
| softmaxMaxUb / softmaxSumUb / softmaxExpUb | FlashSoftmax 行 max/sum/exp（FP32） |
| mm2ResUb | PV 结果（FP32） |
| pseUb / attenMaskUb | 位置编码 / 掩码 |
| antiquant 相关 | 反量化 scale/offset（quant 场景） |
| FlashDecode partial | split-KV 子块的 partial lse / partial o（GM workspace） |

arch35 另有 `inputQue`/`outputQue`（Cube 队列）与 workspace 分区偏移（`needs_verification`：精确字节预算以源码/平台为准）。

## 9. Workspace 布局

`workspace = sysWorkspace + coreNum × (softmax lse/max/sum + mm1Res×2 + mm2Res×2)` + PA 场景 `mmPACallBackDataSize×2` + antiquant/MSD 扩展（`tiling.cpp` L3597-3625；`tiling_v2.cpp` L3620-3622）。`×2` = double buffer。FlashDecode 场景额外需 split 数 × partial lse/o。精确字节预算 `needs_verification`。

## 10. IFA 相对 PFA / 训练 FA 的流水差异

| 差异 | 训练 FA | PFA（prefill） | IFA（增量 decode） |
|---|---|---|---|
| query S1 | 1~1M | 1~1M | **固定 1** |
| 场景 | 训练前向（含反向中间量） | 推理 prefill（全序列） | 推理增量解码 |
| 输出中间量 | softmax_max + softmax_sum（供反向） | softmaxLse（供增量解码） | 无（纯 decode 输出） |
| KV cache | 无 | prefill 写回 | 历史累积 KV（actualSeqLengths 驱动） |
| FlashDecode（split-KV） | 无 | 无 | **BN2<0.4×核数 时使能** |
| PagedAttention | 无 | blockTable 驱动 | blockTable 驱动（回调拷贝） |
| 模板数 | 多（arch32 6 类） | arch32 多 + arch35 服务化 | **3 种**（C+V/全V/preload） |
| 量化 | FP8(arch35) | deq/quant/antiquant/MSD | deq/quant/antiquant/**MSD** |
| KV 左 padding | 无 | 无 | **V4 新增** |
| MLA | 部分 | D=576 | MLA 全量化模板 IFA 不调用（设计介绍 L56） |

## 11. 阶段 -> Triton 抽象

| 阶段 | CANN 机制 | Triton 抽象 | 可迁移 |
|---|---|---|---|
| Bmm1 QK | Cube mm + L0C FP32 累加 | `tl.dot(q, k.T)` in FP32（S1=1） | 是（数学） |
| mask | atten_mask select -> -inf | `tl.where(mask, -inf, s)` | 是 |
| FlashSoftmax | 在线 max/sum/exp + PV rescale | running max/sum + `exp(m_old-m_new)` rescale | 是（算法） |
| Bmm2 PV | Cube mm | `tl.dot(p, v)` | 是（数学） |
| rescale 写回 | ÷softmax_sum | `o = acc / l_i` | 是 |
| FlashDecode | split-KV + lse 加权 reduce | 两段 kernel（split + combine） | 是（算法） |
| AntiQuant-MSD | A 变换 int8×int8->int32 reduce | `tl.dot(int8,int32)`+reduce | 是（算法，精度 needs_verification） |
| PageAttention | 回调 GM->L1 拷贝 | `tl.load(block_table)` 间接 load | 是（语义） |
| GQA | UB 驱动 G 切分 | G 维循环 constexpr | 是（语义） |
| CV 流水/N-Buffer/double buffer | MTE/queue/event/Cube 双发 | tile 复用/fused load-store（编译器调度） | 否（机制） |
| ND↔NZ | DataCopy 格式转换 | Triton layout（编译器） | 否（机制） |
