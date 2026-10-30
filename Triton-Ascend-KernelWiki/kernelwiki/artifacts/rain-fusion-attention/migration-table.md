# rain_fusion_attention - CANN -> Triton-Ascend 迁移分类

| 知识点 | CANN 来源 | Triton 处理方式 | 分类 | 可信度 |
|---|---|---|---|---|
| selectIdx 驱动块级稀疏 QK/PV | README L28-30 | `tl.load(select_idx)` + 循环 | transferable | inferred |
| blockShape [X,Y] 块大小 | README L74-76 | `tl.constexpr BLOCK_M/N` | transferable | inferred |
| online softmax + LSE 输出 | kernel.h | running max/sum + rescale | transferable | inferred |
| FP32/Half softmax 精度选择 | tilingkey.h L21 | `if constexpr` | transferable | inferred |
| FP16/BF16 输入 | def.cpp L21-22 | dtype constexpr | transferable | inferred |
| MIX_AIC_1_2 | kernel L34 | 不迁移 | CANN-specific | inferred |
| CATLASS 模板库 | README L5 | 不迁移 | CANN-specific | inferred |
| tiling key 十进制组装 | tilingkey.h L17-26 | `if constexpr`/launcher | rewrite | inferred |
| BNSD layout（源码支持，README 说不支持） | kernel L59-70 vs README L134 | 以源码为准 | needs verification | inferred |
| innerPrecise 属性 | def.cpp L73 | softmax 精度分支 | rewrite | inferred |