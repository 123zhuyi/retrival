# nsa_selected_attention - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 公式：Gather(K, topKIndices) -> Softmax(scale·Q·K^T)·V | docs L22-28 | `tl.load(topkIndices)` -> `tl.dot(q, k.trans)` -> softmax | directly transferable | inferred | no |
| 在线 FlashSoftmax rescale（running max/sum） | .h L524-581 | `tl.max/sum` + `exp` rescale | directly transferable | inferred | no |
| scale_value 缩放 | .h L546 | `s *= scale` | directly transferable | inferred | no |
| GQA gSize 分组 | .h L241-245 | Q 布局 `[S1,N2,G,D]`，KV 共享 | directly transferable | inferred | no |
| softmaxMax/Sum 输出（供反向用） | .h L573-577 | `tl.store(max_ptr, m_i); tl.store(sum_ptr, l_i)` | directly transferable | inferred | yes（shape dim=8 具体含义） |
| topKIndices gather K/V | .h L447-481,484-521 | `idx=tl.load(topkIndices); tl.load(k_ptr+idx*...)` | rewrite | inferred | no |
| BMM1（Q·selected_K^T） | .h L688-715 | `tl.dot(q, tl.trans(k))` | rewrite | inferred | no |
| BMM2（P·selected_V） | .h L725-739 | `tl.dot(p, v)` | rewrite | inferred | no |
| HandleMask（causal） | .h L621-666 | `tl.where(causal_mask, s, -inf)` | rewrite | inferred | no |
| core id / S1 切分 | tiling_general.cpp L579-593 | `tl.program_id` + `grid` | rewrite | inferred | no |
| 2 tiling key（0=无mask，1=causal） | tiling_general.cpp L565-572 | `if constexpr` / 独立 kernel | rewrite | inferred | no |
| V gather 与 BMM1 重叠 | .h L382-384 | 不迁移（CANN 流水） | CANN-specific | inferred | no |
| MTE DataCopy / ND↔NZ | - | 不迁移 | CANN-specific | inferred | no |
| D_qk=192, D_v=128 固定 | README L62 | constexpr 值 source-reported | needs verification | inferred | yes |
| 仅 A2 训练（A3 不支持） | docs L5-12 | 架构约束 | needs verification | inferred | yes |
| 仅 TND | docs L191,371 | layout 约束 | needs verification | inferred | yes |
| softmaxMax/Sum shape `[T,N1,8]` dim=8 含义 | .h L248,573-577 | 8=每 G 组 reduce 宽度 | needs verification | inferred | yes |
| tiling 参数具体数值 | tiling_general.cpp | constexpr 值 source-reported | needs verification | inferred | yes |