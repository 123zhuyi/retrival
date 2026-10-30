# Final report template

Write to `reports/<operator>-ingest-report.md`. Every section must be present; mark each task parameter field as `source-confirmed` / `inferred` / `needs_verification`. This report is the human-facing summary; the KB pages are the machine-retrievable artifact.

```markdown
# <OPERATOR_NAME> CANN->Triton-Ascend ingestion report

## 1. Task parameters
| Parameter | Value | Status |
|---|---|---|
| SOURCE_REPO_PATH | ... | source-confirmed |
| KERNELWIKI_ROOT | ... | source-confirmed |
| OPERATOR_NAME | ... | source-confirmed |
| OPERATOR_TYPE | ... | inferred (reason) |
| VARIANTS | ... | inferred |
| mode | static_ingest | — |

## 2. Source file inventory
(table: path | role | centrality | key symbols | line ranges | deps | op relation)
Full table in artifacts/<bundle>/source_file_inventory.md.

## 3. Operator semantics
inputs/outputs, attributes, shape/dtype/layout constraints, core formula, boundaries,
mask/predicate, workspace, dynamic shape, special behaviors, variants.

## 4. Stage / pipeline table
| Stage | Entry | Unit | Op | (attention/norm/matmul/elementwise extras as applicable)

## 5. Tiling parameters
CANN original tiling/core-partition table.

## 6. Core partition strategy
per-core data range, tail, multi-core fill, CV matching (if attention).

## 7. Kernel templates & dispatch
count, files, naming, triggers (dtype/layout/shape/feature/arch/tiling-key),
which rules transfer vs which are CANN compile/binary registry.

## 8. Pipeline analysis
| Stage | CANN mechanism | Purpose | Triton abstraction | Transferable? |

## 9. CANN -> Triton-Ascend migration classification
| Knowledge item | CANN source (path Lxx) | Triton handling | Bucket | Confidence | Needs verification |
(buckets: transferable / rewrite / CANN-specific / needs-verification)
Full table in artifacts/<bundle>/migration-table.md.

## 10. Triton-Ascend skeleton
Reference artifacts/<bundle>/<operator>-triton-skeleton.py.
reproducibility: pseudocode | snippet. Not compiled / not benchmarked.

## 11. KB pages written
| Path | type | id | sources | confidence | reproducibility |
(list source page(s), wiki kernel/technique/pattern page(s), artifact bundle files + PROVENANCE)

## 12. Validation & retrieval
- validate.py: <output or "not run: reason; commands; expected checks">
- generate-indices.py: <...>
- query.py <OPERATOR_NAME> --compact: <hits>
- query.py <OPERATOR_TYPE> tiling --compact: <hits>
- query.py <OPERATOR_TYPE> mask --compact: <hits>
- query.py <OPERATOR_TYPE> triton ascend --compact: <hits>

## 13. Unverified / needs-hardware items
list every needs_verification item + why (runtime/precision/layout/perf).

## 14. Modified / created files
list every file created or edited, with one-line purpose.
```

## Quality checklist before declaring done

- [ ] Every `sources:` resolves to a source page id; every wiki page cites a source.
- [ ] `architectures/hardware_features/techniques/kernel_types/symptoms` are exact `tags.yaml` values.
- [ ] `validate.py` passes (no missing PROVENANCE, no sha256 mismatch, no unresolved sources, no unknown vocab).
- [ ] `performance_claims: []` unless baseline+env+shapes+repeats+result exist (they do not in static_ingest).
- [ ] `confidence` is `inferred`/`source-reported` (not `verified`); `reproducibility` honest (pseudocode/snippet/source).
- [ ] No claim of "runs / verified / benchmarked" anywhere.
- [ ] No CANN-specific mechanism (MTE/BufferID/queue/event/manual sync/ND↔NZ) presented as Triton guidance.
- [ ] Report lists every created/edited file.
