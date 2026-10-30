---
id: ops-transformer-moe-token-unpermute-with-ep
title: CANN ops-transformer MoeTokenUnpermuteWithEp source snapshot
type: source
repo: cann/ops-transformer
path: moe/moe_token_unpermute_with_ep/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-31
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MoeTokenUnpermuteWithEp 来源摘要

仓库快照固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（2026-07-09）。`MoeTokenUnpermuteWithEp` 是 MoE token 反重排（unpermute）算子，根据 sortedIndices 从 permutedTokens 取数据（与 probs 相乘）并合并累加。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `moe/moe_token_unpermute_with_ep/`

## 2. 核心源文件清单与 SHA-256

| 源文件 | SHA-256 | 作用 |
|------|------|------|
| `README.md` | `a7b47ded7cf32ead36cbab3df5879be59386c66e2889b16fe102fedde957e2e4` | 算子总览、前向公式、约束 |
| `op_host/moe_token_unpermute_with_ep_def.cpp` | `e2517df4698f59a9a0f416c89a56d1f45f4ad14edb398683ff063625eb130e60` | OpDef（AICore 配置 ascend910b/910_93 + kirin）|
| `op_host/moe_token_unpermute_with_ep_tiling.h` | `0f26e6d391d13cbe29f9126b2b5ab585edf4effa846b707e2c7c5157b2fc709a` | TilingData 结构体 |
| `op_kernel/moe_token_unpermute_with_ep.cpp` | `ca1b77465bbadf78b9c30407f2f9a5eb51efe0c4fc073015a3215b94f8ff28f7` | kernel 入口（4 tiling key）|
| `op_kernel/moe_token_unpermute_with_ep.h` | `22dcf1dafdf3142b0a294a7939215f997517996aec243628d504c1963b1ffe12` | KernelMoeTokenUnpermuteWithEp 模板 |

## 3. 关键代码区域摘要

### 前向公式（README.md）
1. `sortedIndices = sortedIndices[range[0] <= i < range[1]]` - range 过滤
2. `permutedTokens = permutedTokens.indexSelect(0, sortedIndices)` - gather
3. 若 probs 非 None：`permutedTokens_k = permutedTokens_k * probs_{i,j}` - 乘 probs
4. `out_i = Σ_{k=i*numTopk}^{(i+1)*numTopk-1} permutedTokens_k` - 跨 topK reduce sum

### 输入输出（def.cpp）
- 输入：permuted_tokens（BF16/FP16/FLOAT）、sorted_indices（INT32）、probs（可选）
- 输出：unpermuted_tokens（BF16/FP16/FLOAT）
- 属性：num_topk(1)、range([0,0])、padded_mode(false)、restore_shape（仅 nullptr）

### 架构（def.cpp L61-66）
ascend910b/910_93 + kirinX90/kirin9030（不支持 ascend950）

### Kernel dispatch（kernel L39-49）
4 tiling key：0（无 probs）/ 1（float probs）/ 2（half probs）/ 3（bf16 probs）。1 模板类 `KernelMoeTokenUnpermuteWithEp`。

### 约束（README.md）
topK_num <= 512；不支持 paddedMode=True；rangeOptional 为空时回退到 `MoeTokenUnpermute`

## 4. 衍生引用
- `wiki/kernels/moe-token-unpermute-with-ep.md`
- `artifacts/moe-token-unpermute-with-ep/`