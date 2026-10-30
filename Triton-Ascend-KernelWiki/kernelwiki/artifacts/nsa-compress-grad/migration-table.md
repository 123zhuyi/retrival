# nsa_compress_grad - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源。NSA compress grad 为 NSA 压缩算子的反向，仅 A2 训练（arch32）。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| dw = dk_cmp · K^T（weight 梯度） | README L18-25 | `tl.dot(dk_cmp, K.T)` + reduce | transferable | inferred | no |
| dk = W^T · dk_cmp（input 梯度） | README L18-25 | `tl.dot(W.T, dk_cmp)` | transferable | inferred | no |
| TND 变长（actSeqLen cumsum → per-batch seqLen） | `nsa_compress_grad.h` L171-186（`GetSeqLenTarBch`） | cumsum + per-batch mask/offset | transferable | inferred | yes（cumsum 解析精确公式） |
| 压缩滑窗 blockSize > blockStride | `tiling.cpp` L112-118 | 滑窗切分逻辑 | transferable | inferred | no |
| compressBlockSize 与 compressStride 16 对齐 | `tiling.cpp` L220-233 | `tl.constexpr` 对齐约束 | transferable | inferred | no |
| AtomicAdd 累加 weightGrad（多核写同一位置） | `nsa_compress_grad.h` L376-388（`SetAtomicAdd<float>` + `DataCopyPad`） | `tl.atomic_add` | rewrite | inferred | yes（Triton atomic_add 支持） |
| overlap 场景 inputGrad 累加（blockSize > blockStride） | `nsa_compress_grad.h` L79（`isOverLap_`）/ L405-409 | atomic_add 或 double buffer + reduce | rewrite | inferred | yes（overlap 边界算术） |
| weightGrad reduce（BlockReduceSum + WholeReduceSum） | `nsa_compress_grad.h` L330-349 | `tl.sum` 沿 headDim 轴 reduce | rewrite | inferred | no |
| inputGrad broadcast（weight 沿 headDim 广播） | `nsa_compress_grad.h` L352-370（`AscendC::BroadCast`） | `tl.broadcast_to` | rewrite | inferred | no |
| 分核：block 轴外切（按 aivNum） | `tiling.cpp` L87-136（`DoOpTiling`） | `tl.program_id(0)` 沿 block 维 | rewrite | inferred | no |
| workspace 清零 + ping-pong buffer 搬运 inputGrad | `nsa_compress_grad.h` L190-232 / L522-562 | Triton 编译器管理（不迁移手动 DataCopy） | CANN-specific | inferred | no |
| MTE DataCopyPad / SetFlag / WaitFlag 同步 | `nsa_compress_grad.h` L304-327 / L393-399 | 不迁移；Triton load/store | CANN-specific | inferred | no |
| DeterministicComputeSumWtGrad 规约 | `nsa_compress_grad.h` L440-500 | Triton `tl.atomic_add` 或分段规约 | CANN-specific | inferred | yes（确定性算法差异） |
| TilingKey=0 单模板 | `tiling.cpp` L152-155 | 单 `@triton.jit` kernel | rewrite | inferred | no |
| compressBlockSize/Stride 最优值 | `tiling.cpp` L112-118 | constexpr 值 source-reported | needs verification | inferred | yes |
| UB 字节预算 | `nsa_compress_grad.h` L118-156（`InitBuffers`） | 不声称；需硬件实测 | needs verification | inferred | yes |
| actSeqLenType=0（cumsum）语义 | README L36 | 以源码 `GetSeqLenTarBch` 为准 | needs verification | inferred | yes |