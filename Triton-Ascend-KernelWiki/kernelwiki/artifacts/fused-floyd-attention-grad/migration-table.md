# fused_floyd_attention_grad - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。Floyd Attention Grad 是 Floyd Attention 的反向（双流 K1/V1+K2/V2），6 个 matmul 反向计算。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| dS = softmax * dY（下游梯度） | README L31 | `dS = softmax * dY` | transferable | inferred | no |
| dV1 = P^T · dY | README L35 | `tl.dot(P.T, dY)` | transferable | inferred | no |
| dV2 = P^T · dY | README L39 | `tl.dot(P.T, dY)` | transferable | inferred | no |
| dQ = (dS·K1 + dS·K2) / √d | README L43 | `(tl.dot(dS, K1) + tl.dot(dS, K2)) / sqrt(d)` | transferable | inferred | no |
| dK1 = (dS^T · Q) / √d | README L47 | `tl.dot(dS.T, Q) / sqrt(d)` | transferable | inferred | no |
| dK2 = (dS^T · Q) / √d | README L51 | `tl.dot(dS.T, Q) / sqrt(d)` | transferable | inferred | no |
| 双流注意力（K1/V1 + K2/V2 Q 共享） | README L17 | 两次 matmul 后相加 | transferable | inferred | no |
| mask select -> -inf | README L23 | `tl.where(mask, -inf, s)` | transferable | inferred | no |
| scale_value 缩放 | def.cpp L152 | `s *= sm_scale` | transferable | inferred | no |
| FP32 累加 | def.cpp dtype（FLOAT 支持） | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| softmax_max/sum 中间量（前向输出） | def.cpp L78-96 | 已保存的 forward 中间量 | transferable | inferred | no |
| 三段式 kernel（Pre->S1S2->Post） | kernel.cpp L41-76 | 单 kernel 或 3 个 kernel launch | rewrite | inferred | no |
| TilingKey 十进制分发 | kernel.cpp L91-120 | `if constexpr` / launcher dispatch | rewrite | inferred | no |
| 核间切分 S1S2 BN2GS1S2 | `tiling_s1s2_bn2gs1s2.cpp` | `tl.program_id` + grid | rewrite | inferred | no |
| tiling 参数数值 | `tiling.cpp` | constexpr 值 source-reported | needs verification | inferred | yes |
| UB/L1 字节预算 | `kernel.cpp` L11-16（KFC_L1_RESERVER_SIZE=0） | 不声称；需硬件实测 | needs verification | inferred | yes |
| 三段式 CV 同步 | kernel.cpp L55/67（SyncALLCores） | 不迁移；Triton kernel launch barrier | CANN-specific | inferred | no |