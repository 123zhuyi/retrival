---
id: kernel-elementwise-add
title: Elementwise add with a tail mask
type: kernel
tags: [elementwise, add, tail_mask, bf16]
techniques: [tail_mask, fp32_accumulation]
architectures: [Ascend910B1]
kernel_types: [math]
operators: [add]
symptoms: [precision_mismatch]
confidence: experimental
reproducibility: snippet
artifact_dir: artifacts/elementwise
related: [technique-tail-mask, technique-fp32-accumulation]
sources: [doc-triton-ascend-api, run-elementwise-tail-001]
performance_claims: []
---

# Elementwise add

Build one-dimensional offsets from the program id and block size. Use `offsets < n` on all loads and the final store. If accumulation or nonlinear work is sensitive, compute the intermediate value in FP32 before conversion.
