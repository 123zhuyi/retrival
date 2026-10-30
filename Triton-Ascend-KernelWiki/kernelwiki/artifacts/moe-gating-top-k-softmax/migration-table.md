# moe_gating_top_k_softmax - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| softmax(x, axis=-1) | README L19-21 | `tl.softmax` / 在线 softmax | transferable | inferred |
| topK(softmaxOut, k) | README L23-25 | `tl.topk` / 自定义并行 topk | transferable | inferred |
| rowIdx reshape+transpose | README L27-33 | `tl.arange` + reshape | transferable | inferred |
| finished 条件填充 | README L16/56-61 | `tl.where(finished, num_expert, idx)` | transferable | inferred |
| AIV-only | kernel L53-55 | 单 program family | transferable | inferred |
| 18 tiling key 分支 | kernel L57-111 | `if constexpr` | rewrite | inferred |
| 3 模板族（E_K/K FullLoad/Perf） | kernel L22-47 | 数据量选择策略 | rewrite | inferred |
| ColRange 优化（<8/8-64/>64） | kernel L85-110 | expert 数选择策略 | rewrite | inferred |
| arch32 vs arch35 | def.cpp L51-59 | launcher 选 kernel | rewrite | inferred |
| arch35 regbase | arch35/*.h | 不迁移（CANN 机制） | CANN-specific | inferred |