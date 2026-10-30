---
id: ops-transformer-fused-floyd-attention
title: CANN ops-transformer FusedFloydAttention source snapshot
type: source
repo: cann/ops-transformer
path: attention/fused_floyd_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer FusedFloydAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/fused_floyd_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。

`FusedFloydAttention` 是训练推理场景下使用 FloydAttention 算法实现的多维自注意力计算，支持 2 组 KV head group。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/fused_floyd_attention/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-31

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `5b7dd16e4953a6539954c54d4b675e8b098646e6db0a08afda10cb5cbb21f499` | 算子总览、公式、I/O、约束 |
| `op_host/fused_floyd_attention_def.cpp` | `94162a4f25f585790331f04e80d1b9804a08dcb31e361860ce98f614170b3eff` | OpDef（query/key1/value1/key2/value2/atten_mask/softmax_max/sum/attention_out）|
| `op_host/fused_floyd_attention_tiling.cpp` | `ca793331479ab46dd56242ac432e77d88aaca39d4fa14eb325acbd904fcbd8ba` | tiling入口、架构分发 |
| `op_kernel/fused_floyd_attention.cpp` | `8e75a5663850d672d4cb07fd41765e9edb2de31208cafd3c4cb5656c06a59a48` | kernel入口（4 分支）|
| `op_kernel/fused_floyd_attention_common.h` | `992a599c4bc33e36fbb62f4463d4598af8af35b823de37a1f4d78e9d68d49299` | 公共定义 |

## 3. 关键代码区域摘要

### 3.1 核心公式（README.md L19-25）

```
weights = Softmax(attenMask + scale * (einsum(Q, K1^T) + einsum(Q, K2^T)))
attention_out = einsum(weights, V1) + einsum(weights, V2)
```

4 次 matmul: Q·K1^T + Q·K2^T + P·V1 + P·V2

### 3.2 输入输出（def.cpp L20-74）

- 输入：query、key1、value1、key2、value2（FP16/BF16/FLOAT）、atten_mask（BOOL/UINT8）
- 输出：softmax_max/softmax_sum（FLOAT）、attention_out（FP16/BF16/FLOAT）
- 属性：scale_value(1.0)

### 3.3 约束（README.md L112-124）

B:1~2K, H:1~256, N:16~1M(N%16==0), M:128~1M(M%128==0), K:128~1M(K%128==0), D:32/64/128

### 3.4 Kernel dispatch（kernel L114-150）

单模板 `FusedFloydAttentionS1s2Bn2gs1`，4 分支：FP16/BF16 × hasAttenMask。仅 BNSD layout，仅 ascend910b/910_93。

## 4. 衍生引用

- `wiki/kernels/fused-floyd-attention.md`
- `artifacts/fused-floyd-attention/`