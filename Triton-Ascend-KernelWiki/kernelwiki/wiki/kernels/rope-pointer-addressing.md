---
id: kernel-rope-pointer-addressing
title: RoPE paired-channel pointer addressing
type: kernel
tags: [rope, pointer_rewrite, compile_error]
techniques: [pointer_rewrite, tail_mask]
architectures: [Ascend910B1]
kernel_types: [posembedding]
operators: [rope]
symptoms: [compile_error]
confidence: experimental
reproducibility: snippet
related: [technique-tail-mask, lang-triton-ascend-api]
sources: [doc-triton-ascend-api, run-rope-pointer-001]
performance_claims: []
---

# RoPE pointer addressing

Derive `pair_base` from a non-negative channel index, then add the lane within the pair. Do not form a pointer with `current_offset - 1`; use a boundary predicate for incomplete pairs.
