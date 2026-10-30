# mla_prolog - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/mla_prolog/`。`fixture: true`。有 arch32 + arch35 双架构。

| 文件（相对算子目录） | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 |
|---|---:|---|---|---|---|
| `README.md` | 361 | doc | core | 功能/四路数据流 L12-60；参数 L62-89；约束 L91-128；3 场景(非量化/部分量化KV非量化/KV量化) L97-125 | `bd0059922d16b0f8...` |
| `docs/MlaProlog算子设计介绍.md` | 454 | doc | core | 数据流 L9-22；4 matmul 切分 L36-44；CV 流水 6 sync event L47-67；tiling key 二进制位 L68-79；Process 伪代码 L83-146；完整公式 L149-416；RoPE 优化公式(15) L359-415 | `2c079807b5efafca...` |
| `docs/aclnnMlaProlog.md` | 742 | doc/api | core | aclnn API 语义 | `228c925df2d06b73...` |
| `op_host/mla_prolog_def.cpp` | 271 | host/def | core | `MlaProlog` OpDef：输入 L23-117；输出 L118-133；属性 L134-136；AICore config(910b/910_93 L145-146 + 950 L148-267) | `d9d8661e9b5c8fa5...` |
| `op_host/mla_prolog_infershape.cpp` | 137 | host | core | infershape | `3ff2d50fcf67e516...` |
| `op_host/mla_prolog_tiling.cpp` | 841 | host/tiling | core | tiling 主逻辑 | `f62cfa3324e05ad3...` |
| `op_host/mla_prolog_tiling.h` | 392 | host/tiling | core | `MlaPrologCompileInfo`/`MlaPrologTiling` 类 | `fb5b3d6784a52a18...` |
| `op_host/mla_prolog_tiling_check.cpp` | 1061 | host/tiling | core | tiling 校验 | `15eeaa690d8bc5fd...` |
| `op_host/mla_prolog_tiling_register.cpp` | 30 | host/tiling | core | 注册 | `e6c3597d804f5708...` |
| `op_kernel/mla_prolog.cpp` | 100 | kernel/entry | core | `mla_prolog` 模板入口 L20-50；8 模板参数 L20-22；MIX_AIC_1_2 L51；3 场景区分(NoQuant/PartialQuantKVNoQuant/PartialQuantKVQuant) L69-99 | `2309ef39ee78bc47...` |
| `op_kernel/kernel_mla_prolog_split_n.h` | 2241 | kernel | core | `MlaPrologVecS1CubS2` 主体（Process/MatmulCq/MatmulCkvKr/MatmulQcQr/MatmulQn/RmsNorm/RoPE/ScatterCache） | `cc9e9ccfa531f1fd...` |
| `op_kernel/mla_prolog_comm.h` | 436 | kernel/common | supporting | 公共结构/常量 | `a183daefada693a1...` |
| `op_kernel/mla_prolog_template_tiling_key.h` | 411 | kernel/dispatch | core | tiling key 二进制位编码 L68-79；模板参数组合 | `1188bed8af1320cd...` |
| `op_kernel/mla_prolog_tiling_data.h` | 75 | kernel | core | `MlaPrologTilingData`/`MlaPrologBaseParams` | `804be1946799622d...` |
| `op_kernel/mla_prolog_vector_comm.h` | 536 | kernel/common | supporting | Vector 公共 | `a179bba54eec3bf1...` |
| `op_kernel/service_matmul.h` | 832 | kernel/service | core | Cube matmul 基础服务 | `e951e748aaa9cef3...` |
| `op_kernel/service_rms_norm.h` | 170 | kernel/service | core | RmsNorm 服务 | `fddf4682c65c9d5b...` |
| `op_kernel/service_rope.h` | 73 | kernel/service | core | RoPE 服务 | `5436b1abb8d55d47...` |
| `op_kernel/service_rotary_position_embedding.h` | 136 | kernel/service | core | RoPE 公共实现 | `18f32d12c378066e...` |
| `op_kernel/service_scatter_cache.h` | 196 | kernel/service | core | Cache scatter 写回 | `3e4385b4de756919...` |
| `op_kernel/service_dequant.h` | 200 | kernel/service | supporting | 反量化服务 | `01a07236fffa54ea...` |
| `op_kernel/service_dynamic_quant_qn_mul_qr.h` | 261 | kernel/service | supporting | 动态量化 | `1bb5ff19caaaee26...` |
| `op_kernel/service_gather_sin_cos.h` | 64 | kernel/service | supporting | 正余弦 gather | `055f7192238c77ac...` |
| `op_kernel/arch32/rms_norm.h` | 102 | kernel/arch32 | supporting | arch32 RmsNorm | `8d970f483005ce06...` |
| `op_kernel/arch35/vf/vf_{rms_norm,rope,dequant,dynamic_quant,quant_perchannel,quant_pertensor,mul_qr}.h` | 69-429 | kernel/arch35 | supporting | arch35 Vector 服务 | 见前述 SHA-256 |

另含 `examples/`(3)、`tests/ut/`(5)、`op_host/fallback_mla_prolog.{cpp,h}`(140/92L)。