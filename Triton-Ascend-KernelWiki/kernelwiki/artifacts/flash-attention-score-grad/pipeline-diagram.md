# FlashAttentionScoreGrad - 流水抽象 (S6)

来源：`docs/FAG算子设计介绍.md` L9-110、`op_kernel/flash_attention_score_grad.cpp` INVOKE 宏 L59-484、`op_kernel/arch35/flash_attention_score_grad_block_cube.h`/`block_vec.h`。**不复制 CANN 代码**；MTE/BufferID/queue/event/手动 CV 同步抽象为 tile 级复用/融合存取。

## 1. 反向 6 步计算流程（算法骨架，transferable）

以变量 S、P 表示：`S=Mask(QK^T/√d + pse, atten_mask)`，`P=Dropout(Softmax(S),keep_prob)`，`Y=PV`。

| 步 | 公式 | 说明 |
|----|------|------|
| 1 重算 p | `p = SimpleSoftmax(Mask(Matmul(query, key^T) + pse) * scale)` | 重算前向 softmax 结果 P，存 UB（FAG doc L9） |
| 2 dp | `dp = Dropout(Matmul(dy, value^T))` | 左矩阵 dy，右矩阵 V^T（FAG doc L10） |
| 3 ds | `ds = p * Sub(dp, FlashSoftmaxGrad(dy, attention_in))` | FlashSoftmaxGrad 入参 dy、前向输出 attention_in(=Y)（FAG doc L11） |
| 4 dq | `dq = Matmul(ds, key) * scale` | ds·K（FAG doc L12） |
| 5 dk | `dk = Matmul(ds^T, query) * scale` | ds^T·Q（FAG doc L13） |
| 6 dv | `dv = Matmul(Dropout(p)^T, dy)` | Dropout(P)^T·dy（FAG doc L14） |

> FlashSoftmaxGrad 实质：`ds = P ⊙ (dP − rowsum(dP ⊙ P))`，其中 `dP = dY · V^T`（步骤 2 的 dp）。attention_in(=Y=PV) 与 dy 参与 softmax 反向归一化项。反向**重算** P（不强制依赖前向存的 softmax_max/sum，但 softmax_max/sum 作为可选输入可加速重算，见 `def.cpp` L72-83）。

## 2. 三阶段 Pipeline 入口（SameAB 模板示例，FAG doc L20-23）

| 阶段 | 入口 | 单元 | 对应步 |
|------|------|------|--------|
| Process | `Process` | AIC+AIV | 任务块分配 |
| Cube1 | `ComputeMM1` | AIC | 步 1（重算 p：QK^T matmul + softmax） |
| Vector1 | `ComputeVec` | AIV | 步 2-3（dp、ds：FlashSoftmaxGrad + elementwise） |
| Cube2 | `ComputeMMDqkv` | AIC | 步 4-6（dq=ds·K、dk=ds^T·Q、dv=P^T·dy matmul） |

> 共 4 次 matmul：mm1=Q·K^T（重算）、mm2=dy·V^T(dp)、mm3=ds·K(dq) / ds^T·Q(dk)、mm4=P^T·dy(dv)。TilingData 含 `mm1TilingData/mm2TilingData/mm3TilingData`（S1s2Bn2gs1s2）或 `mm1AndMm2/mm31/mm32AndMm4/mm4`（B/BN 模板），见 `op_kernel/flash_attention_score_grad.cpp` L64-66, L291-294, L415-417。

## 3. 阶段映射表（CANN 机制 -> 目的 -> Triton 抽象 -> 可迁移?）

| 阶段 | CANN 机制 | 目的 | Triton 抽象 | 可迁移? |
|------|----------|------|------------|--------|
| 数据读入 | `DataCopyPad` GM->UB/L1，ND↔NZ 格式转换 | Q/K/V/dy 载入片上 | `tl.load` + stride；masked 尾块 | 算法迁移，格式转换不移植 |
| 重算 QK^T | Cube `IterateBmm1`，L1 存 Q/K，L0C 累加 FP32 | 步 1 matmul | `tl.dot(q, k.T)` FP32 累加 | 是（matmul 语义） |
| FlashSoftmax（重算 P） | Vector `ProcessVec1`：pse add/mul、scale、atten_mask select、在线 max/sum/exp | 步 1 softmax | 在线 softmax（running max/sum + rescale） | 是 |
| dp = dy·V^T | Cube `Bmm2` | 步 2 | `tl.dot(dy, v.T)` | 是 |
| FlashSoftmaxGrad | Vector `vec_sfmg`/`vf_softmax_grad_front_cast*`：Sub(dp, FlashSoftmaxGrad(dy,Y)) | 步 3 softmax 反向 | `ds = P*(dP - rowsum(dP*P))` 在线归约 | 是 |
| dq/dk/dv matmul | Cube `ComputeMMDqkv`：ds·K、ds^T·Q、P^T·dy；L1 复用 P/ds | 步 4-6 | `tl.dot` 三次；tile 复用 | 是 |
| 中间保存 | workspace（A2）或 UB/L1（950PR） | mm1 重算结果、dq/dk/dv 累加 | on-chip `tl` tensor / drop | 是（drop workspace） |
| 写回 | `CopyUBToGM`/cast FP32->dtype、muls scale | Post 段输出 | `tl.store` masked + cast | 是 |
| 双缓冲 | V 侧 ping-pong double buffer | 搬运与计算重叠 | 编译器排流；skeleton 表达循环 | 意图迁移，机制不移植 |
| CV 协同 | A2 Cube 双发(preload 2)/950PR preload；`fixp_l0c_to_ub`/`copy_ub_to_l1` | Cube/Vector 并行，减少空闲 | 编译器排流；tile 内复用 | 意图迁移，机制不移植 |
| 同步 | BufferID `SetFlag`/`WaitFlag`、TQue/TBuf event | CV 间数据依赖 | 不移植（编译器管 pipe） | 否（CANN-specific） |

## 4. 架构差异

- **arch32（A2）**：AIC/AIV 分离，CV 通路为 L2/GM；C-Tiling 128×1024，V-Tiling 8×1024，nRatio=8，1:16 配比；Cube 双发（preload 2 块）达成 MTE2/Cube bound（FAG doc L93-101）。多模板（8 类）。
- **arch35（950PR）**：AIC/AIV 融合，CV 高速通路 L0C->UB、UB->L1；C 128×128，V 64×128（C:V=1:2，Cube 算完均分两 V 核）；统一模板（2 文件）+ 确定性特化（FAG doc L52-72）。低阶 AscendC API，Cube 为主核，极致内存复用。
- **确定性计算**：通过特定分核避免多核同地址累加（`FAG算子设计介绍.md` L357-366，`kernel_deter.h`）。
