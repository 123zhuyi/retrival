# recurrent_gated_delta_rule - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| S_t 更新公式（α·Diag(αk)·S_{t-1} + β·(v - α·Diag(αk)·S_{t-1}·k)·k^T） | README L24-26 | `tl.dot` + `tl.where` + 外积 | transferable | inferred |
| 门控 α=e^g / αk=e^gk | README L89-97 | `tl.exp(g)` | transferable | inferred |
| 输出 o_t = S_t·q·scale | README L28-30 | `tl.dot(S, q) * scale` | transferable | inferred |
| AIV-only 执行（无 Cube） | kernel L31 | 单 program family | transferable | inferred |
| BF16 输入 FP32 中间 | tiling_data.h | `tl.dot(..., dtype=tl.float32)` | transferable | inferred |
| 变长 actual_seq_lengths | def.cpp L47-50 | 前缀和步进 | transferable | inferred |
| 分核（vectorCoreNum） | tiling_data.h | `tl.program_id` + grid | rewrite | inferred |
| sBlockNum / vStep 分块 | tiling_data.h | `tl.constexpr` BLOCK_* | rewrite | inferred |
| 可选 g/gk/numAcceptedTokens | def.cpp L57-71 | `if constexpr` 分支 | rewrite | inferred |
| BUFFER_NUM=1 单 buffer | kernel.h L27 | 不迁移（编译器调度） | CANN-specific | inferred |
| 变长不支持的约束 | 仅 BF16/dk≤256? | n/a | needs_verification | inferred |