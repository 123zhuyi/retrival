# fused_floyd_attention_grad - 源文件清单

仓库：`kunpengcompute/CANN-ops-transformer`，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。
算子目录：`attention/fused_floyd_attention_grad/`。`fixture: true`。源码 ~5k 行。

centrality：core = 直接实现该算子；supporting = 公共/框架依赖；indirect = 测试/示例/文档。SHA-256 为前 16 位。

Floyd Attention Grad 是 Floyd Attention（Moonshot Kimi 长上下文双流注意力）的反向梯度算子，训练场景，仅 A2/A3 训练（ascend910b/910_93），无 950。核心是 6 个 matmul 反向计算（dQ/dK1/dV1/dK2/dV2）。

## docs

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `README.md` | 195 | doc | core | 产品支持（A2/A3 训练）L7-13；公式（6 个反向 matmul）L17-52；参数表 L54-171；约束 L173-189 | `2870f402281f5e92…` | 算子总览 |
| `docs/aclnnFusedFloydAttentionGrad.md` | 652 | doc/api | core | aclnn API 定义 | `68dec780cceac911…` | 接口语义 |

## op_host

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_host/fused_floyd_attention_grad_def.cpp` | 171 | host/def | core | `FusedFloydAttentionGrad` OpDef：输入7+输出5（query/key1/value1/key2/value2/dy/dy_grad → dq/dk1/dv1/dk2/dv2）；AICore config（910b L164、910_93 L165）；属性 scale_value L152 | `129b803140567954…` | 算子定义 |
| `op_host/fused_floyd_attention_grad_infershape.cpp` | 86 | host | core | `InferShapeFusedFloydAttentionGrad` + `InferDataType` | `1a8110f0a09054a8…` | shape/dtype 推导 |
| `op_host/fused_floyd_attention_grad_tiling.cpp` | 221 | host/tiling | core | tiling 主逻辑 | `018e789e52a1cfdf…` | tiling 主逻辑 |
| `op_host/fused_floyd_attention_grad_tiling.h` | 39 | host/tiling | core | `FusedFloydAttentionGradTilingData` 结构体 | `631c51af4428b88b…` | tiling 结构体 |
| `op_host/fused_floyd_attention_grad_tiling_common.cpp` | 149 | host/tiling | core | tiling 公共逻辑 | `5c318bd2392629f0…` | tiling 公共 |
| `op_host/fused_floyd_attention_grad_tiling_common.h` | 319 | host/tiling | core | tiling 公共头 | `76f69f156ad47ca0…` | tiling 公共头 |
| `op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2.cpp` | 979 | host/tiling | core | S1S2 BN2GS1S2 tiling | `9f169a1b3778f7c3…` | S1S2 tiling |
| `op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2.h` | 183 | host/tiling | core | S1S2 tiling 头 | `10561c6b057a9b449a2…` | S1S2 tiling 头 |
| `op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2_def.h` | 122 | host/tiling | core | `FusedFloydAttentionGradTilingDataS1s2Bn2gs1s2` 结构体 | `b45565e9f421b9d1…` | S1S2 tiling 结构体 |

## op_kernel

| 文件 | 行数 | role | centrality | 关键符号 / 区域 | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|---|
| `op_kernel/fused_floyd_attention_grad.cpp` | 122 | kernel/entry | core | `fused_floyd_attention_grad` kernel L79-122；`INVOKE_FFAG_GENERAL_S1S2_BN2GS1S2_IMPL` 宏 L41-76；三段式（Pre → S1S2 → Post）；TilingKey 分发 L91-120 | `06d12b6c0de20625…` | kernel 入口 |
| `op_kernel/fused_floyd_attention_grad_pre.h` | 163 | kernel | core | `FusedFloydAttentionGradPre`：反向预处理（dS = (softmax * dY)） | `01dbd93eabf2db5a…` | 反向预处理 |
| `op_kernel/fused_floyd_attention_grad_s1s2_bn2gs1s2.h` | 1595 | kernel | core | `FusedFloydAttentionGradS1s2Bn2gs1s2`：6 个 matmul 反向（dQ/dK1/dV1/dK2/dV2） | `4abc8e9c487cc176…` | 核心反向计算 |
| `op_kernel/fused_floyd_attention_grad_post.h` | 577 | kernel | core | `FusedFloydAttentionGradPost`：反向后处理（梯度搬运/格式转换） | `204c74cdf042531c…` | 反向后处理 |

## tests

| 文件 | 行数 | role | centrality | SHA-256 | 与算子关系 |
|---|---:|---|---|---|---|
| `tests/ut/op_host/test_fused_floyd_attention_grad_tiling.cpp` | 176 | test | indirect | `4d8cb72949575562…` | tiling UT |
| `tests/ut/op_host/test_fused_floyd_attention_grad_infershape.cpp` | 89 | test | indirect | `b453826a5e69b0e2…` | infershape UT |
| `examples/test_aclnn_fused_floyd_attention_grad.cpp` | 273 | example | indirect | `287227fcc73f1a7e…` | 调用示例 |

注：
- Floyd Attention Grad 是 Floyd Attention 的反向算子，双流注意力（K1/V1 + K2/V2），核心公式 6 个 matmul 反向。
- kernel 三段式：`Pre`（dS = softmax * dY）→ `S1S2`（6 个 matmul 反向计算 dQ/dK1/dV1/dK2/dV2）→ `Post`（梯度搬运/格式转换）。
- 仅 A2/A3 训练（ascend910b/910_93），无 950，无 310P。支持 FP16/BF16/FLOAT。