# ring_attention_update - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| online softmax rescale 合并公式 | README L18-28 | `alpha=exp(max_prev-max_new)` 元素乘加 | transferable | inferred |
| FP32 中间计算 | kernel.h | `tl.exp`+乘除 FP32 | transferable | inferred |
| SBH/TND layout 寻址 | kernel | `tl.load` stride 参数 | transferable | inferred |
| 6 tiling key 分支 | ring_attention_update.cpp L32-84 | `if constexpr`/launcher | rewrite | inferred |
| 950 regbase 3 模板 | ring_attention_update_apt.cpp L33-60 | launcher 选 kernel | rewrite | inferred |
| Kirin 仅 FP16/FLOAT | def.cpp L109-111 | dtype 分支 | rewrite | inferred |
| 310P BF16 排除 | kernel L41/67 `#if` | 编译期排除 | CANN-specific | inferred |