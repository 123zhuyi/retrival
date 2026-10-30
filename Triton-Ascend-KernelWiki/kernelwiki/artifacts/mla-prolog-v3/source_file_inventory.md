# mla_prolog_v3 - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`（本地快照 `kernelwiki/artifacts/ops-transformer/`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`）。
算子目录：`attention/mla_prolog_v3/`。`fixture: true`。源码 ~9298 行（含 test）。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

MLA Prolog V3 是 DeepSeek V2/V3 核心创新 MLA（Multi-head Latent Attention）的前处理融合算子：4 路 matmul + 2 个 RmsNorm + 2 个 RoPE + KVCache/KRCache 更新。三架构（arch32 A2/A3 + arch35 950）。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 228 | doc | core | 产品支持（三架构√）L5-9；功能更新（V3 相对 V2）L13-27；四路计算流程 L28-28；公式 L31-78；参数表 L80-121；约束 L123-204 | `5628fb05be17240f…` | 算子总览 |
| `docs/MlaPrologV3算子设计介绍.md` | 454 | doc | **core** | 实现原理（MLA 四路）L15-25；数据切分 L27-45；**CV 流水设计（6 个同步点）** L47-68；**TilingKey 26 位二进制** L70-85；**主流程伪码** L88-143；完整公式 L145-433；KVCache L435-448；KRCache L450-454 | `d57e211a68c30f69…` | **设计权威** |
| `docs/aclnnMlaPrologV3WeightNz.md` | 1657 | doc/api | core | V3 aclnn API 定义（token_x/weight_dq/weight_uq_qr/… → query/query_rope/kv_cache/kr_cache）；参数表；约束；量化场景表 | `d354d130e94c8565…` | 接口语义 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/mla_prolog_v3_def.cpp` | 319 | host/def | core | `MlaPrologV3` OpDef：输入输出 dtype（token_x/weight: BF16/INT8/FLOAT8_E4M3FN；weight_uk: BF16/FP16；cache: FP16/BF16/INT8）；AICore config（910b L171、910_93 L172、950 L315） | `3df4652526c8dd37…` | 算子定义 |
| `op_host/mla_prolog_v3_infershape.cpp` | 166 | host | core | `InferShapeMlaPrologV3` + `InferDataTypeMlaPrologV3` | `a5b71d974c058e35…` | shape/dtype 推导 |
| `op_host/mla_prolog_v3_infershape.h` | 55 | host | core | infershape 头 | `7c20ff8c831c3855…` | infershape 头 |
| `op_host/mla_prolog_v3_tiling.h` | 25 | host/tiling | core | tiling 结构 | `8adb18fa36cc2c33…` | tiling 结构 |
| `op_host/mla_prolog_v3_tiling_register.cpp` | 20 | host/register | core | tiling 注册 | `9424c13c1fe93cea…` | tiling 注册 |
| `op_host/fallback_mla_prolog_v3.cpp` | 156 | host/fallback | supporting | host fallback 执行 | `4fd0ace91006f085…` | fallback |
| `op_host/fallback_mla_prolog_v3.h` | 87 | host/fallback | supporting | fallback 头 | `0d70d1d984bc7729…` | fallback 头 |
| `op_host/op_api/aclnn_mla_prolog_v3_weight_nz.cpp` | 210 | api | core | aclnn V3 包装 | `ecd5cad2627f4060…` | aclnn C++ API |
| `op_host/op_api/aclnn_mla_prolog_v3_weight_nz.h` | 49 | api | core | aclnn 头 | `9779609992e6aed3…` | aclnn 头 |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/mla_prolog_v3.cpp` | 182 | kernel/entry | core | `mla_prolog_v3<CacheMode,Scenario,QuantMode,EnableDequantOpt,EnableGroupComputeOpt,EmptyTensorMode,ActualSeqLenMode,SplitMMode,CvMode>` 模板 kernel L25-28；入口函数 L28+；调用 `kernel_mla_prolog_split_n.h`（共享 mla_prolog 公共实现） | `e22a2cacc755f5d6…` | kernel 入口（模板参数化） |

> 注意：`mla_prolog_v3.cpp` 包含 `#include "../../mla_prolog/op_kernel/kernel_mla_prolog_split_n.h"`（L17-21），实际 kernel 实现来自 **mla_prolog 公共目录**（非 `mla_prolog_v3/` 内）。这表示 V3 与 V1/V2 共享核心实现，差异在模板参数与入口配置。

## tests / examples

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_host/test_mla_prolog_v3_tiling.cpp` | 5389 | test | indirect | `b75822921c9724b0…` | tiling UT |
| `tests/ut/op_host/test_mla_prolog_v3_infershape.cpp` | 760 | test | indirect | `4cd7ac034315bf27…` | infershape UT |
| `tests/ut/op_kernel/test_mla_prolog_v3.cpp` | 1029 | test | indirect | `cee1ea1cac637197…` | kernel UT |
| `tests/ut/op_kernel/mla_prolog_tiling.cpp` | 38 | test | indirect | `0051c423f60acbc5…` | tiling 测试工具 |
| `tests/ut/op_kernel/mla_prolog_tiling_data.h` | 75 | test | indirect | `4fbc136351b62b66…` | tiling 测试数据 |
| `examples/test_aclnn_mla_prolog_v3.cpp` | 341 | example | indirect | `f3ce97dd394b50c1…` | aclnn 调用示例 |
| `examples/arch35/test_aclnn_mla_prolog_v3_fqkvq.cpp` | 397 | example | indirect | `7566fd6ddc2a9f0e…` | arch35 示例 |

注：
- MLA Prolog V3 的核心 kernel 实现来自 `mla_prolog/` 共享目录（`kernel_mla_prolog_split_n.h`），`mla_prolog_v3.cpp` 仅定义模板参数化入口（9 个 uint8_t 模板参数对应 TilingKey 的 26 位二进制位：CacheMode/Scenario/QuantMode/EnableDequantOpt/EnableGroupComputeOpt/EmptyTensorMode/ActualSeqLenMode/SplitMMode/CvMode）。
- 设计文档（454 行）是 attention 类中最完整的之一，包含 4 路 matmul 分核策略、6 个 CV 同步点、TilingKey 二进制位定义、完整主流程伪码。
- 950 架构（arch35）的 AICore config 与 910b 不同（L174-315），单独定义了 dtype/layout（FLOAT8_E4M3FN 支持）。
- `aclnnMlaPrologV3WeightNz` 接口名中的 "WeightNz" 表示 weight 用 NZ 格式。
- 三架构均支持：arch32（A2 A3）+ arch35（950）。