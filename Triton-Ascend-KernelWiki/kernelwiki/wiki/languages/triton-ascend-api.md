---
id: lang-triton-ascend-api
title: Triton-Ascend load, store, constexpr, and pointer rules
type: language
tags: [triton_ascend, tl_load, tl_store, constexpr, tail_mask, pointer_rewrite]
architectures: [Ascend910B1]
symptoms: [compile_error, precision_mismatch]
confidence: source-reported
reproducibility: snippet
related: [hw-ascend910b-constraints, technique-tail-mask]
sources: [doc-triton-ascend-api, run-rope-pointer-001]
---

# API rules

- Use the same boundary predicate for related loads and stores.
- Reserve `tl.constexpr` for compile-time specialization parameters.
- Construct pointer addresses from a valid base and explicit non-negative offsets.
- Re-run compilation and minimal-shape correctness tests after an index rewrite.
