# FlashAttentionScoreGrad Operator Ingestion Report

## Task Parameters

| Parameter | Value | Status |
|-----------|-------|--------|
| SOURCE_REPO_PATH | `kernelwiki/artifacts/ops-transformer/attention/flash_attention_score_grad` | Confirmed |
| KERNELWIKI_ROOT | `kernelwiki/` | Confirmed |
| OPERATOR_NAME | `flash_attention_score_grad` | Confirmed |
| OPERATOR_TYPE | `attention` | Confirmed |
| VARIANTS | `backward` | Confirmed |

## Summary

This report summarizes the static ingestion of the CANN FlashAttentionScoreGrad backward operator into the Triton-Ascend KernelWiki. The ingestion process analyzed the complete CANN source code for the operator, extracted semantics, tiling, pipeline, and dispatch logic, and created structured knowledge base entries.

## Source Inventory

Created source page: `kernelwiki/sources/ops-transformer/flash-attention-score-grad.md`

The source inventory includes all critical files:
- Core API implementation files
- Architecture-specific kernel code for arch32 and arch35
- Tiling implementations for both architectures
- Documentation and examples
- Test suite

## Kernel Semantics

Extracted and documented:
- Full forward/backward computation formulas
- Complete input/output tensor specifications
- All attributes and configuration parameters
- Data type support range
- Layout and shape constraints
- Special behaviors (RoPE, sparse attention, deterministic mode)

## Tiling & Core Partition

Created tiling analysis: `kernelwiki/artifacts/flash-attention-score-grad/tiling-analysis.md`

- 64-bit tiling key structure with 18 distinct fields
- Bit-range definitions for all tiling parameters
- Value mappings for each field
- Architecture-specific tiling implementations (arch32 vs arch35)
- Workspace requirements

## Kernel Templates & Dispatch

Created template analysis: `kernelwiki/artifacts/flash-attention-score-grad/kernel-templates.md`

- All specialized kernel templates for both architectures
- Dispatch logic and trigger conditions
- Special key signatures (0x1434 sameab, 0x134 bn2, etc.)
- Architecture differences in template organization

## Pipeline Abstraction

Created pipeline analysis: `kernelwiki/artifacts/flash-attention-score-grad/pipeline-abstraction.md`

- High-level pipeline stages
- Core-specific pipeline details (Cube/Vector)
- Double-buffer optimization
- Pipeline parallelism patterns
- Hardware-specific optimizations
- Triton-Ascend abstraction mapping

## Triton-Ascend Skeleton

Created skeleton implementation: `kernelwiki/artifacts/flash-attention-score-grad/flash-attention-score-grad-triton-skeleton.py`

- Generic kernel skeleton with grid and block configuration
- Deterministic variant skeleton
- Autotuned configurations for different hardware
- Mapping from CANN primitives to Triton DSL

## Knowledge Base Pages

Created wiki kernel page: `kernelwiki/wiki/kernels/flash-attention-score-grad.md`

The page includes:
- Comprehensive operator documentation
- Architecture-specific implementation details
- Tiling and pipeline optimization descriptions
- Template specialization information
- Constraints and usage guidelines

## Validation Results

✅ KernelWiki validation passed successfully
✅ All required schema fields are present
✅ All tags use correct controlled vocabulary
✅ Query returns expected results for both operator name and type

## Artifact Bundle

Bundle directory: `kernelwiki/artifacts/flash-attention-score-grad/`

Files created:
1. `PROVENANCE.yaml` - Automated provenance tracking
2. `tiling-analysis.md` - Tiling and core partition analysis
3. `kernel-templates.md` - Template and dispatch analysis
4. `pipeline-abstraction.md` - Pipeline abstraction
5. `flash-attention-score-grad-triton-skeleton.py` - Triton kernel skeleton

## Unverified Items

The following items could not be verified through static analysis alone:
- Runtime performance claims
- Exact numerical precision edge cases
- API behavior under all error conditions
- Real-world benchmark results

## Migration Classification

Based on static analysis, the operator components fall into these migration categories:

| Category | Items |
|----------|-------|
| directly transferable | Math semantics, boundary rules, mask rules, FP32 accumulation |
| rewrite to Triton-Ascend | CANN tiling, core-partition, template-choice → Triton block/grid abstractions |
| CANN-specific, do not transfer | MTE, BufferID, queue/event, manual sync, ND↔NZ physical moves |
| needs later verification | API semantics, runtime-dependent behaviors |

## Next Steps

1. Review the generated pages for accuracy
2. Run performance validation if runtime access is available
3. Publish the knowledge base entries using triton-kb-source-builder
4. Add to regression test suite
