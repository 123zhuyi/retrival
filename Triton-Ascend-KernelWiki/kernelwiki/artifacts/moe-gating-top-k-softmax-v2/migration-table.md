# MoeGatingTopKSoftmaxV2 - 迁移分类 (S3)

| 知识项 | CANN 来源 | Triton-Ascend 处理 | 桶 | Confidence |
|--------|-----------|-------------------|------|------------|
| renorm=0：softmax(x) -> topk | README.md 公式1 | 先 softmax 再 topk | transferable | inferred |
| renorm=1：topk(x) -> softmax(topkOut) | README.md 公式2 | 先 topk 再 softmax | transferable | inferred |
| softmax 语义（axis=-1） | README.md | `tl.softmax` 跨 expert 轴 | transferable | inferred |
| topk 语义（k 值 + indices） | README.md | `tl.topk(x, k)` | transferable | inferred |
| finished 标记（True 则 expert_idx=num_expert） | README.md | mask 替换 | transferable | inferred |
| FP32 累加（softmax 中间 FP32） | kernel | `tl.float32` 计算 | transferable | inferred |
| AIV-only 无 matmul | kernel L63 MIX_AIV_1_0 | 纯 softmax + topk | transferable | inferred |
| k/renorm/output_softmax_result_flag 语义 | def.cpp L50-52 | 作为标量参数 | transferable | inferred |
| 4 模板策略（EKFullLoad/KFullLoad/KRenorm/Perf） | kernel L68-176 | launcher 按 shape/renorm/colRange 选 constexpr | rewrite | inferred |
| colRange 3 档（<8 / 8-64 / >64） | kernel Perf | 按 expert 数选 BLOCK | rewrite | inferred |
| 核间切分 | tiling | grid 映射 | rewrite | inferred |
| MTE/DataCopy | kernel | `tl.load/tl.store` | CANN-specific | inferred |
| 手动同步 | kernel | 编译器排流 | CANN-specific | inferred |