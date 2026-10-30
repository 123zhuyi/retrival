# mla_prolog_v3 - CANN -> Triton-Ascend 迁移分类

四分类（directly transferable / rewrite to Triton-Ascend / CANN-specific do-not-transfer / needs later verification）。每条带 CANN 来源。MLA Prolog V3 为 DeepSeek V2/V3 核心 MLA 的前处理融合算子（4 路 matmul + 2 个 RmsNorm + 2 个 RoPE + Cache 更新）。

| 知识点 | CANN 来源（path Lxx） | Triton-Ascend 处理方式 | 分类 | 可信度 | 需后续验证 |
|---|---|---|---|---|---|
| c^Q = x · W^{DQ} 下采样 matmul | 设计介绍 L39-40/170-176 | `tl.dot(x, W_dq)` | transferable | inferred | no |
| c^{KV} = x · W^{DKV} + k^R = x · W^{KR} 拼接 matmul | 设计介绍 L42-43/198-216 | `tl.dot(x, [W_dkv, W_kr])` 拼接 -> split | transferable | inferred | no |
| q^C = c^Q · W^{UQ} + q^R = c^Q · W^{QR} 拼接 matmul | 设计介绍 L44/256-273 | `tl.dot(c_q, [W_uq, W_qr])` 拼接 -> split | transferable | inferred | no |
| q^N = q^C · W^{UK} 上采样 matmul | 设计介绍 L45/427-433 | `tl.dot(q_c, W_uk)` | transferable | inferred | no |
| RmsNorm 公式 | 设计介绍 L180-196 | `tl.var` + `tl.rsqrt` + `* gamma` | transferable | inferred | no |
| RoPE 偶奇拆分 + cos/sin 旋转 | 设计介绍 L275-415 | Triton `tl.cos`/`tl.sin` + 偶奇 split/merge | transferable | inferred | yes（精确公式以设计介绍 L361-414 为准） |
| α_q / α_kv 尺度矫正因子 | 设计介绍 L167-168；README L14 | `c_q *= alpha_q` / `c_kv *= alpha_kv` | transferable | inferred | no |
| KVCache scatter 更新（ND/NZ/PA） | 设计介绍 L435-448 | `tl.store(cache + cache_index)` | rewrite | inferred | yes（NZ stride 算术） |
| KRCache scatter 更新 | 设计介绍 L450-454 | 同 KVCache | rewrite | inferred | no |
| 动态量化（rowmax + smooth_scale） | 设计介绍 L229-253 | `tl.max` + `scale = max/127` + `round(x/scale)` | rewrite | inferred | yes（per-tile 量化模式） |
| 静态量化 per-channel | 设计介绍 L253 | `tl.dot` 后 `*= scale` | rewrite | inferred | no |
| 分核：MatmulCq 沿 Hcq 切 AIC 核数 | 设计介绍 L39-40 | `tl.program_id` 沿 Hcq 维 | rewrite | inferred | no |
| 分核：MatmulCkvKr 分成 64 块 | 设计介绍 L42 | `tl.program_id` 沿 Hckv 64 块 | rewrite | inferred | no |
| 分核：MatmulQcQr 沿 Hqc 切 AIC 核数 | 设计介绍 L44 | `tl.program_id` 沿 Hqc 维 | rewrite | inferred | no |
| 分核：MatmulQn 不切 N | 设计介绍 L45 | 单点 | rewrite | inferred | no |
| **CV 流水 6 同步点**（设计介绍 L94-140） | 设计介绍 L47-68 | Triton 不迁移手动同步 | CANN-specific | inferred | no |
| 核间同步 SYNC_MMCQ_NORMSCQ / SYNC_MMCKVKR_NORMROPE / SYNC_ALL_VECTOR / SYNC_MMQCQR_ROPEQR / SYNC_ALL_CUBE | 设计介绍 L94-140 | 不迁移；Triton kernel launch barrier | CANN-specific | inferred | no |
| CrossCoreSetFlag/CrossCoreWaitFlag 显式同步 | 设计介绍 L94-140 | 不迁移 | CANN-specific | inferred | no |
| TilingKey 26 位二进制（9 字段） | 设计介绍 L70-85 | `if constexpr`/模板参数分支 | rewrite | inferred | no |
| UB 内存布局（TBuf 偏移） | `mla_prolog_v3.cpp` InitBuffers | 不迁移 | CANN-specific | inferred | no |
| tiling 参数具体数值 | `tiling.h` | constexpr 值 source-reported | needs verification | inferred | yes |
| 共享 kernel 实现（mla_prolog/） | `mla_prolog_v3.cpp` L17-21 | V3 与 V1/V2 共享 core 实现 | needs verification | inferred | yes |
| per-tile 量化精度边界 | README L144-204 | 需运行时验证 | needs verification | inferred | yes |
| cacheIndex 与 cache 更新关系 | 设计介绍 L435-454 | 以源码为准 | needs verification | inferred | yes |
| Mxfp8 量化模式 | README L144-204 | 需确认 Triton 支持 | needs verification | inferred | yes |