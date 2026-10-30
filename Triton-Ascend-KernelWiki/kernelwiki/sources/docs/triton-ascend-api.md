---
id: doc-triton-ascend-api
title: Triton-Ascend API source notes
repo: Ascend/triton-ascend
url: https://github.com/Ascend/triton-ascend
source_category: official-doc
retrieved_at: 2026-07-15
tags: [triton_ascend, tl_load, tl_store, constexpr, tail_mask]
architectures: [Ascend910B1]
---

# API source notes

Static launch parameters may use `tl.constexpr`; runtime shapes remain scalar arguments. Non-divisible shapes require a boundary mask on load and store. Pointer expressions should be constructed from a valid base plus explicit non-negative offsets.
