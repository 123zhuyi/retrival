---
id: run-rope-pointer-001
title: RoPE pointer rewrite demo evaluation run
repo: local-evaluation
url: local://runs/rope-pointer-001
source_category: reproduction
captured_at: 2026-07-15
tags: [rope, compile_error, pointer_rewrite]
architectures: [Ascend910B1]
kernel_types: [posembedding]
fixture: true
---

# Evaluation evidence

The failing negative-offset expression is rewritten as `pair_base + lane_in_pair`; incomplete pairs are excluded by a mask. Compilation and minimal-shape correctness checks are required.
