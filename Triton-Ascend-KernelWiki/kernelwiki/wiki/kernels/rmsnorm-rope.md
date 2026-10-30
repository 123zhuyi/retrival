---
id: kernel-rmsnorm-rope
title: RMSNorm-RoPE fused kernel on Ascend 910B1
type: kernel
tags: [rmsnorm, rope, normalization, vecutil_low, bf16]
techniques: [load_order, contiguous_access, fp32_accumulation]
hardware_features: [unified_buffer, vector_core]
architectures: [Ascend910B1]
kernel_types: [norm, posembedding]
operators: [rmsnorm, rope]
symptoms: [vecutil_low, precision_mismatch]
confidence: experimental
reproducibility: snippet
artifact_dir: artifacts/demo-pr-993
related: [pattern-vecutil-low, technique-contiguous-load-order, technique-fp32-accumulation]
sources: [pr-triton-ascend-fixture-993, run-rmsnorm-rope-001]
performance_claims: []
---

# RMSNorm + RoPE

Accumulate the sum of squares in FP32 and convert only at the output boundary. Keep the RoPE even/odd channel pairing explicit. For the documented small-batch, hidden-4096 case, inspect whether lanes access adjacent addresses before changing tile size.

## Agent actions

1. Preserve RMSNorm and RoPE semantics.
2. Check UB use for input, weight, FP32 temporaries, and output.
3. Apply a tail mask when the hidden dimension is not tile-aligned.
4. Compare the current load order with the contiguous hidden dimension.
5. Validate correctness before accepting a profiling improvement.
