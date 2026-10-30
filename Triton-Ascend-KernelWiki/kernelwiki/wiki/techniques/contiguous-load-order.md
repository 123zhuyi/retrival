---
id: technique-contiguous-load-order
title: Organize loads along the contiguous dimension
type: technique
tags: [load_order, contiguous_access, vecutil_low]
techniques: [load_order, contiguous_access]
architectures: [Ascend910B1]
kernel_types: [norm, math]
symptoms: [vecutil_low]
confidence: experimental
reproducibility: snippet
prerequisites: [hw-ascend910b-constraints]
related: [pattern-vecutil-low, kernel-rmsnorm-rope]
sources: [pr-triton-ascend-fixture-993, run-rmsnorm-rope-001]
---

# Contiguous load order

Map neighboring lanes to neighboring addresses on the innermost contiguous dimension. Confirm the benefit with profiling; code appearance alone is not sufficient evidence.
