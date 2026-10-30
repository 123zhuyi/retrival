---
id: run-elementwise-tail-001
title: Elementwise tail-mask demo evaluation run
repo: local-evaluation
url: local://runs/elementwise-tail-001
source_category: reproduction
captured_at: 2026-07-15
tags: [elementwise, tail_mask, bf16]
architectures: [Ascend910B1]
kernel_types: [math]
fixture: true
---

# Evaluation evidence

The fixture covers lengths 0, 1, BLOCK-1, BLOCK, and BLOCK+1. A single `offsets < n` mask is applied to every load and store.
