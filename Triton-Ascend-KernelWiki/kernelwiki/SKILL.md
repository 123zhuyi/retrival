---
name: triton-ascend-kernelwiki
description: Query a source-grounded Triton-Ascend kernel knowledge base before kernel design, debugging, or optimization.
---

# Triton-Ascend KernelWiki

Use `scripts/query.py` to identify relevant pages, `scripts/get_page.py` to read selected pages, and `scripts/grep_wiki.py` for exact API or symptom text. Follow each Wiki page's canonical `sources` links. The Agent adapter may assemble these same operations into a page bundle; it must not introduce a second ranking model.

For upstream Triton-Ascend documentation, filter with `--repo triton-lang/triton-ascend`. Imported document snapshots remain source pages until a reviewed synthesis explicitly cites them from `wiki/`.

Use `--tag <kernel_type>` for CANN-aligned operator categories such as `math`, `norm`, `matmul`, `attention`, or `moe`. Use `--symptom <error_type>` for documented problems such as `compile_error`, `precision_mismatch`, or `ub_overflow`. Architecture tags use exact chip models such as `Ascend910B1` and `Ascend950PR`; NpuArch, A2/A3/A5, product-family, and archXX inputs are expanded as query selectors. Language tags and a language filter are intentionally not part of this version. The controlled vocabulary and evidence commits are recorded under `data/`.
