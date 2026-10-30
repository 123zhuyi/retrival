# Orchestrator integration contract

## Contents

1. Invocation points
2. Input contract
3. Category and architecture mapping
4. Failure mapping
5. Output contract
6. Consumer contract
7. Prohibited behavior

## 1. Invocation points

### Phase 2: design

Invoke after Phase 1 produces the normalized task and after Phase 2 template precheck locks `category`, `template_path`, and Layer 1 constraints. Invoke before `triton-op-designer`.

Do not retrieve before Phase 1 because the orchestrator does not yet have normalized operator semantics and shapes.

### Phase 3: repair

Invoke only after verifier failure and Conductor A-class classification. Use the complete verifier failure set rather than the first failure. Skip retrieval for PyTorch-fallback AST violations when the AST checker already supplies the deterministic fix. Preserve the existing B- and C-class termination behavior.

### Phase 4: optimize

Invoke only after the Phase 3 correctness gate. Build one context per optimizer iteration when the operator, diagnosed bottleneck, or requested technique changes.

## 2. Input contract

Required inputs:

| Field | Source |
|---|---|
| `mode` | Current orchestrator phase |
| `workdir` | Phase 0 work directory |
| `op_name` | Phase 0/1 normalized task name |

Optional inputs:

| Field | Source and rule |
|---|---|
| `semantic_op[]` | Exact PyTorch operation or operator aliases found in `Model.forward` |
| `category` | Explicit template category from `precheck.json` |
| `kernel_type` | Exact CANN category; omit if uncertain |
| `arch` | Phase 0 architecture |
| `symptom` | Conductor classification using the controlled vocabulary |
| `verifier_error_file` | File containing all verifier failures |
| `keyword[]` | Exact API, profiler finding, or requested technique; never use generated speculation |
| `iteration` | Current Phase 3 or Phase 4 iteration |

The script accepts 1–5 results per lane and defaults to 2. Keep the default unless the operator has no useful results.

## 3. Category and architecture mapping

Conservative template mapping:

| Template category | Default `kernel_type` |
|---|---|
| `index-computation` | `index` |
| `math-compute` | `math` |
| `normalization` | `norm` |
| `quantization` | `quant` |
| `tensor-transform` | `conversion` |
| `object-detection` | none; query exact operators such as `nms` or `iou` |
| `transformer-inference` | none; pass the exact type such as `attention`, `moe`, `ffn`, `gmm`, `mamba`, `mc2`, `mhc`, or `posembedding` |

The explicit `--kernel-type` always overrides the category default. Do not invent a tag for a broad category.

Architecture normalization:

```text
Ascend910B1                              -> Ascend910B1
A2                                       -> Ascend910B1/B2/B2C/B3/B4
A3                                       -> Ascend910_93
DAV_2201 / arch22 / Ascend910B / 2201    -> all A2 and A3 chip models
A5 / DAV_3510 / arch35 / Ascend950       -> Ascend950DT and Ascend950PR
```

Page tags always use exact chip-model names from the `npu-hardware-params.md` product table. NpuArch, SocVersion, A2/A3/A5, archXX, and family names are only one-to-many query selectors. Official-source queries deliberately omit an architecture filter because many imported Triton-Ascend documents do not identify an exact model. Hardware lanes expand a selector into its chip-model set. Do not hardcode core counts or capacities from a model tag; retrieve runtime-varying values through the platform API.

## 4. Failure mapping

Pass an explicit controlled symptom when Conductor has already classified the failure. Otherwise the script recognizes these deterministic patterns:

| Failure evidence | Symptom |
|---|---|
| output mismatch, allclose failure, relative or absolute error | `precision_mismatch` |
| shape, size, or dimension mismatch | `shape_mismatch` |
| MLIR, compilation, Python syntax/type failure | `compile_error` |
| unsupported operation/API | `unsupported_operation` |
| illegal address or out-of-bounds access | `illegal_memory_access` |
| UB/unified-buffer overflow | `ub_overflow` |
| allocation failure or OOM | `out_of_memory` |
| misaligned address or alignment requirement | `alignment_error` |
| other recognized device runtime failure | `runtime_error` |

If no pattern matches, stop retrieval and require Conductor to select a symptom. Do not default every unknown failure to `runtime_error`.

## 5. Output contract

Default paths:

```text
WORKDIR/knowledge/design_context.json
WORKDIR/knowledge/repair_iter_N.json
WORKDIR/knowledge/optimize_iter_N.json
```

Top-level schema:

```json
{
  "schema_version": "1.0",
  "mode": "design|repair|optimize",
  "request": {},
  "retrieval": {
    "engine": "kernelwiki_title_tag_body",
    "second_stage_reranking": false,
    "limit_per_lane": 2,
    "lanes_are_independent": true
  },
  "precedence": [],
  "lanes": [],
  "warnings": []
}
```

Each lane contains its original KernelWiki bundle. Preserve page `score`, frontmatter, body, cited source excerpts, and warnings. Empty lanes are valid and are reported in top-level warnings.

The skill does not merge or rerank pages across lanes. Ranking inside every lane remains KernelWiki title/tag/body scoring with stable path tie-breaking.

## 6. Consumer contract

### Designer

Pass:

```text
history_context = wiki-kernel + wiki-technique lanes
inspirations = official lane
hardware_context = hardware lane
```

Require architecture-decision comments to cite page IDs when retrieved evidence changes the sketch. Template Layer 1 remains the hard constraint.

### Coding

Add optional input `knowledge_context` to `triton-op-coding`. On retry, pass:

```text
knowledge_context.initial = design context path
knowledge_context.repair = current repair context path
```

Keep `previous_code`, `verifier_error`, and `conductor_suggestion` unchanged.

### Latency optimizer

Add optional input `optimization_knowledge_context`. Treat every suggested optimization as a hypothesis. The existing optimized verifier and measured benchmark decide acceptance.

### Verifier

Pass no retrieval context. The verifier must remain independent from generation evidence.

## 7. Prohibited behavior

- Do not query with the complete task or generated code as one free-form keyword string.
- Do not treat retrieval score as confidence, correctness, or speedup.
- Do not copy GPU-specific or upstream code without adapting it to Ascend.
- Do not overwrite task semantics or template Layer 1 constraints.
- Do not skip all-case correctness verification.
- Do not report historical performance as current-task performance.
- Do not automatically write generated code, reports, or summaries into KernelWiki.
