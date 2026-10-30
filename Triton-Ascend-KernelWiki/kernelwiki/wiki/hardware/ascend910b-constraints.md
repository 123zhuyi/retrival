---
id: hw-ascend910b-constraints
title: Ascend 910B1 kernel constraints
type: hardware
tags: [Ascend910B1, unified_buffer, alignment, grid]
hardware_features: [unified_buffer, alignment, vector_core, cube_core, cube_vector_1_to_2]
architectures: [Ascend910B1]
confidence: source-reported
aliases: [910b1, ascend910b1]
related: [lang-triton-ascend-api, technique-tail-mask]
sources: [doc-triton-ascend-api]
---

# Ascend 910B1 constraints

Estimate peak Unified Buffer use from simultaneously live inputs, outputs, and FP32 temporaries. Reduce the tile when the estimate exceeds the project-specific device limit.

Compute grid size with ceiling division. Every non-divisible tail must be protected by a mask. Treat alignment and supported API behavior as explicit preconditions rather than optimization suggestions.
