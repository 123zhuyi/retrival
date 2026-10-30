# mla_prolog - CANN -> Triton-Ascend 迁移分类

四分类。每条带 CANN 来源 path Lxx。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| 4 matmul 公式（x·W^DQ / cQ·W^UQ / qC·W^UK / x·[W^DKV\|W^KR]） | 设计介绍 L36-44 | `tl.dot`（FP32 acc） | transferable | inferred | no |
| RmsNorm 公式 γ·x/√(mean(x²)+eps) | 设计介绍 L191-197 | `tl.sum(x*x)/N` + `rsqrt` + 乘 γ | transferable | inferred | no |
| RoPE 旋转位置编码 | 设计介绍 L277-415 | 复数旋转乘 | transferable | inferred | no |
| W^UQ\|W^QR 权重拼接合并 matmul 后 split | 设计介绍 L269-273 | 合并 kernel 或 2 个 dot + split | transferable | inferred | no |
| W^DKV\|W^KR 权重拼接合并 matmul 后 split | 设计介绍 L213-217 | 合并 kernel 或 2 个 dot + split | transferable | inferred | no |
| Cache scatter 写回（cache_index） | service_scatter_cache.h | `tl.store(cache + idx*stride)` | transferable | inferred | yes（PA offset） |
| FP32 累加（BF16 输入） | 设计介绍 | `tl.dot(..., out_dtype=tl.float32)` | transferable | inferred | no |
| 3 量化场景（非量化/部分量化/全量化） | mla_prolog.cpp L69-99 | dtype 分支 + cast 融合 | transferable | inferred | yes（per-channel 实现） |
| 空 tensor 分支（emptyMode=2 return） | mla_prolog.cpp L54-56 | early-return | transferable | inferred | no |
| tiling key 二进制位（14 bits） | 设计介绍 L70-79 | `if constexpr`/launcher 分支 | rewrite | inferred | no |
| 8 模板参数 + 3 场景 | mla_prolog.cpp L20-99 | 独立 kernel / `if constexpr` | rewrite | inferred | no |
| BS 合轴 M 维 + 核间切分 | 设计介绍 L36-44 | `tl.program_id` + grid | rewrite | inferred | no |
| 4 matmul 基本块 | tiling_data.h | `tl.constexpr`（BLOCK_M/K/N） | rewrite | inferred | yes（最优值） |
| 950 FP8 支持 | def.cpp L148-267 | 配置型 dtype 分支 | rewrite | inferred | yes（FP8 精度） |
| 6 CV sync event + 全核同步 | 设计介绍 L47-67 | 不迁移（抽象为 tile 依赖） | CANN-specific | inferred | no |
| ND↔NZ cache 格式 | 设计介绍 L439-454 | 不迁移（Triton layout） | CANN-specific | inferred | no |
| MIX_AIC_1_2 CV 配比 | kernel 入口 L51 | 不迁移（CANN 算力配比） | CANN-specific | inferred | no |
| 权重 FRACTAL_NZ 格式 | def.cpp weight_dq/wuq/etc | 不迁移（Triton 编译器） | CANN-specific | inferred | no |
| tiling 参数数值（m0/k0/n0 最优） | tiling.cpp | constexpr 值 source-reported | needs verification | inferred | yes |
| 预留参数（dequant_scale_x/w_dq/w_dkv_kr，未使用） | README L78-84 | 忽略 | needs verification | source-reported | no |
| FP8 路径精度边界 | def.cpp 950 config | 需运行时验证 | needs verification | inferred | yes |
| MLA 权重吸收（W^UQ·W^UK 折叠） | 设计介绍 公式(a) vs 3 matmul | 吸收与否 needs_verification | needs verification | inferred | yes |