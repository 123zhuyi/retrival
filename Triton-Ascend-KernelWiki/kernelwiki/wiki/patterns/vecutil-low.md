---
id: pattern-vecutil-low
title: Low vector utilization diagnosis
type: pattern
tags: [vecutil_low, profiling, contiguous_access, load_order]
techniques: [load_order, contiguous_access]
architectures: [Ascend910B1]
kernel_types: [norm, math]
symptoms: [vecutil_low]
confidence: experimental
candidate_techniques: [technique-contiguous-load-order]
related: [kernel-rmsnorm-rope]
sources: [pr-triton-ascend-fixture-993, run-rmsnorm-rope-001]
---

# Low vector utilization

Check address continuity, tile orientation, scalarized index expressions, and load order. A proposed change is promoted only after semantic, correctness, resource, and profiling checks pass.
