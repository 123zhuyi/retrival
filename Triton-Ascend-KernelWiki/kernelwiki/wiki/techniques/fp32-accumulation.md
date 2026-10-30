---
id: technique-fp32-accumulation
title: Use FP32 for sensitive accumulation
type: technique
tags: [fp32_accumulation, bf16, precision]
techniques: [fp32_accumulation]
architectures: [Ascend910B1]
kernel_types: [norm, math]
symptoms: [precision_mismatch]
confidence: experimental
reproducibility: snippet
prerequisites: []
related: [kernel-rmsnorm-rope, kernel-elementwise-add]
sources: [run-rmsnorm-rope-001, run-elementwise-tail-001]
---

# FP32 accumulation

Convert low-precision inputs to FP32 for reductions and sensitive intermediate expressions, then convert at the output boundary. Validate the accepted error threshold for every supported shape.
