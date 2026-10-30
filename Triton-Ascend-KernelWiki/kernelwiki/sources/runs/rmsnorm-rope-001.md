---
id: run-rmsnorm-rope-001
title: RMSNorm-RoPE demo evaluation run
repo: local-evaluation
url: local://runs/rmsnorm-rope-001
source_category: reproduction
captured_at: 2026-07-15
tags: [rmsnorm, rope, vecutil_low, load_order, bf16]
architectures: [Ascend910B1]
kernel_types: [norm, posembedding]
fixture: true
---

# Evaluation evidence

Fixture conditions: small batch, hidden size 4096, BF16, Ascend 910B1. The run checks compilation, numerical agreement, and profiler vector utilization before and after a contiguous-dimension load-order change. No numeric speedup is asserted in this Demo fixture.
