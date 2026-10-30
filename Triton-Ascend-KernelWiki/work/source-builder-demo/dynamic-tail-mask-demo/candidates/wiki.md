---
id: technique-dynamic-extent-tail-mask
title: Dynamic-extent tail-mask intersection
type: technique
architectures:
- Ascend910B1
tags:
- tail_mask
- dynamic_extent
confidence: experimental
reproducibility: snippet
prerequisites:
- logical extent is available at kernel launch
related:
- technique-tail-mask
sources:
- pr-source-builder-fixture-1200
symptoms:
- illegal_memory_access
- shape_mismatch
evidence_basis:
- fixture diff and fixture correctness result
techniques:
- tail_mask
hardware_features:
- vector_core
- unified_buffer
kernel_types:
- math
---

# Dynamic-extent tail-mask intersection

## Problem

A final tile can be inside the allocated tensor but outside the logical
runtime extent. A mask based on only one of these boundaries is incomplete.

## Method

Construct the effective mask as the intersection of the allocated extent
and the logical runtime extent:

```python
offsets = block_start + tl.arange(0, BLOCK)
mask = (offsets < allocated_n) & (offsets < logical_n)
values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
```

## Applicability

Apply this pattern when the generated kernel receives a logical extent that
may be smaller than the physical allocation. Preserve the operator's
required semantics for padded elements.

## Verification

Cover zero, one-element, exact-block, and partial-tail cases. Rerun the full
correctness suite after applying the pattern. This fixture makes no
performance claim.
