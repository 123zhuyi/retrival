---
name: triton-kb-cann-ingest
description: Statically analyze a CANN operator source tree, abstract it to Triton-Ascend semantics, and ingest a source-grounded, traceable knowledge bundle into the Triton-Ascend KernelWiki (sources + wiki + artifacts). Use when the user asks to analyze/migrate/abstract a CANN operator (e.g. ops-math Abs, ops-transformer FlashAttention) and write the result into the KernelWiki as reusable, searchable knowledge; also use when the user says "ingest <operator> into the KB" or "do the CANN->Triton-Ascend migration analysis + KernelWiki entry". Do NOT use for read-only retrieval (use triton-kernelwiki-retriever), running/profiling kernels, correctness verification, or governed publishing via triton-kb-source-builder.
---

# Triton-Ascend KernelWiki: CANN static ingestion

Turn one CANN operator (or operator family) into a **source-grounded, traceable, searchable** KernelWiki entry. This is `static_ingest` mode: **static source analysis only**. Never claim code was run, verified, benchmarked, or profiled. Never copy the CANN repo into the KB.

This skill encodes the 12-section ingestion task. It produces three layers, all of which already exist as repo conventions — do not invent new directories or frontmatter fields:

- `sources/` — immutable source summary (evidence, not tutorial)
- `wiki/{kernels,techniques,patterns,languages,hardware}/` — reusable knowledge, each citing `sources`
- `artifacts/<bundle>/` — derived material (skeleton, tables, pseudocode) + `PROVENANCE.yaml`

The canonical reference outputs are `flash-attention-score` (attention family, multi-page) and `elementwise-abs` (elementwise, single page). Match their shape. For frontmatter *values*, follow `flash-attention-score` as the gold standard: it uses vocab-canonical `confidence: inferred` / `reproducibility: pseudocode`. The `elementwise-abs` page predates strict vocab enforcement and carries non-canonical `confidence: high` / `reproducibility: source` - treat its structure as illustrative, not its confidence/reproducibility values.

## Inputs

Resolve these before starting. If the user omits one, infer it from the source tree and flag it `inferred` in the final report.

| Parameter | Env / source | Example |
|---|---|---|
| `SOURCE_REPO_PATH` | arg or `CANN_SOURCE_REPO` env | `.../kernelwiki/artifacts/ops-transformer` |
| `KERNELWIKI_ROOT` | arg or `TRITON_ASCEND_KB_ROOT` env; default `kernelwiki/` under repo root | `.../kernelwiki` |
| `OPERATOR_NAME` | arg | `flash_attention_score`, `abs` |
| `OPERATOR_TYPE` | arg, one of `tags.yaml#kernel_types` | `attention`, `math`, `matmul`, `norm`, `moe`, ... |
| `VARIANTS` | arg, optional | `forward`, `backward`, `inference`, `incremental`, `sparse`, `causal`, `rope`, `quant`, `fp16`, `bf16` |

Run the scaffolder to create the artifact bundle + PROVENANCE + report skeleton first:

```bash
python .claude/skills/triton-kb-cann-ingest/scripts/scaffold.py init \
  --root "$KERNELWIKI_ROOT" \
  --source "$SOURCE_REPO_PATH" \
  --operator "$OPERATOR_NAME" --type "$OPERATOR_TYPE" \
  [--variants "forward,inference"]
```

## Mandatory reads (do these first, in order)

1. `kernelwiki/SKILL.md` and `README.md` — KB contract.
2. `kernelwiki/data/schemas.yaml` — frontmatter required fields per page type.
3. `kernelwiki/data/tags.yaml` — controlled vocab (architectures, hardware_features, techniques, kernel_types, symptoms).
4. 1–2 existing pages of the same type: `kernelwiki/wiki/kernels/flash-attention-score.md` and/or `kernelwiki/wiki/kernels/elementwise-abs.md`, plus a `wiki/techniques/*` and `artifacts/*/PROVENANCE.yaml`.

Never edit `tags.yaml`, `architecture-aliases.yaml`, `doc-tagging-rules.yaml`, `aliases.yaml`, or any `schemas.yaml`. Use only chip-model values from `tags.yaml#architectures` (e.g. `Ascend910B1`, `Ascend950PR`); `A2/A3/A5/archXX/SocVersion` are query selectors, not page tags.

## Stages → report sections

Each stage maps to a section of the final report. Work top-to-bottom; keep every claim traceable to a source path + line range.

### S1 · Locate source (report §2)
Search the CANN tree across: `op_host`, `op_kernel`, `op_api`, `op_graph`, `docs`, `examples`, `tests`, `tiling`, `config`, `template`, `dispatch`, `cmake`, `scripts`, `common`, `inc`, `src`, `json`, `yaml`. Record a file inventory: path (relative to repo root), role (host/kernel/tiling/api/config/test/doc/common), centrality (core/supporting/indirect), key functions/classes/macros/templates, line ranges, dependencies, relation to the target op. Write it to `artifacts/<bundle>/source_file_inventory.md`. Track shared `common/`/`template/` deps the operator uses.

For heavy fan-out, spawn an `Explore` subagent per subdirectory to inventory in parallel; keep only the conclusions.

### S2 · Operator semantics (report §3 + §4)
Inputs/outputs (dtype, layout, shape constraints), attributes/optional params, core formula, boundary conditions, mask/predicate logic, workspace, dynamic-shape support, special behaviors (inplace/broadcast/reduce/transpose/sparse/quant), and forward/backward/inference variants. For attention add Q/K/V/O, scale, mask, causal, sparse_mode, attention_mode, rope/alibi, online softmax, dropout, lse/max/sum, KV cache, prompt/incremental, padding/invalid-index. For norm add mean/var/rms, eps, affine, FP32 accumulation, tail mask. For matmul add M/N/K, transpose, batch, bias/activation fusion, accumulate dtype, block_m/n/k, cube templates. For elementwise add broadcast, vectorization, alignment, tail mask, dtype promotion, multi-input fusion.

**Anything not confirmable from source must be tagged `needs_verification`** — never write an inference as a fact.

### S3 · Migration classification (report §9)
Classify each knowledge item into one of four buckets. Use the rubric in [references/migration-rubric.md](references/migration-rubric.md).

| Bucket | Meaning |
|---|---|
| directly transferable | math semantics, boundary rules, mask rules, online softmax, FP32 accumulation |
| rewrite to Triton-Ascend | CANN tiling / core-partition / template-choice → Triton `block_*`, `program_id`, `tl.constexpr`, `grid` |
| CANN-specific, do not transfer | MTE, BufferID, queue/event, manual Cube/Vector sync, ND↔NZ physical moves |
| needs later verification | runtime-dependent or unverifiable from source (API semantics, precision edges, layout offsets) |

### S4 · Tiling & core partition (report §5 + §6)
Extract Host-side tiling + Kernel-side core partition: primary shape dims, block/tile params, per-core/block/program data range, tail-block handling, workspace, UB/L1/L0/L2/GM usage, multi-tiling-key triggers, big/small/special-shape branches, dtype/layout effects. Produce:
- A. CANN original tiling/core-partition table (key/branch → trigger → params → core/block map → workspace → source line).
- B. Triton-Ascend abstraction map (`core/block id → tl.program_id/grid`, `block_m/n/k → tl.constexpr`, `tail → mask load/store`, `workspace → intermediate tensor / drop`). See [references/artifact-bundle-layout.md](references/artifact-bundle-layout.md).

### S5 · Kernel templates & dispatch (report §7)
Count kernel templates, their files, naming, and dispatch triggers (dtype / layout / shape·head_dim·block size / mask·sparse·causal·quant·rope / Ascend arch / tiling key / CMake·config·json·yaml). State which dispatch rules are still useful for Triton-Ascend vs. which are CANN compile/binary/template-registry mechanics that do not transfer.

### S6 · Pipeline (report §8)
Abstract the Ascend C/C++ pipeline (data read → format convert → main compute → reduction/softmax/matmul/elementwise → intermediate save → writeback → double buffer / queue / event / vector·cube·scalar coordination / overlap / memory reuse / alignment·tail). **Do not copy CANN code.** MTE/BufferID/queue/event/manual sync are CANN-specific; abstract them to tile-level reuse, mask, constexpr meta, program_id mapping, fused load/store, online algorithms, fewer intermediate writebacks.

### S7 · Triton-Ascend skeleton (report §10)
Author a Triton-style skeleton expressing: kernel signature, `tl.constexpr` meta, `tl.program_id` mapping, block/tile offsets, input load, mask/boundary, core compute, accumulation dtype, optional branches, output store, layout/stride. Add attention/matmul/norm/elementwise specifics per the task. See [references/triton-skeleton-templates.md](references/triton-skeleton-templates.md). If not guaranteed compilable, set `reproducibility: pseudocode` (or `snippet`). Save to `artifacts/<bundle>/<operator>-triton-skeleton.py`.

## S8 · Ingest (report §11)

Write into the KB following the frontmatter rules below. Source page first (evidence), then wiki page(s) citing it, then artifact bundle.

### Source page

For a CANN source-code snapshot, follow the `flash-attention-score` convention (passes validation; computed page type is not schema-bound so it carries a minimal, flexible frontmatter):

```text
sources/<repo-short>/<operator>.md      # e.g. sources/ops-transformer/flash_attention_score.md
```
```yaml
id: <repo-short>-<operator>            # e.g. ops-transformer-flash-attention-score
title: CANN <repo> <Operator> source snapshot
type: source
repo: <org/name>
path: <op dir relative to repo root>
commit: <40-char sha>
retrieved_at: YYYY-MM-DD
license: CANN Open Software License Agreement Version 2.0
fixture: true                            # true = excerpt for analysis, not full-repo copy
```
Body: repo/operator location, pinned commit, source-file inventory with SHA-256, key code-region summaries (formula, tiling, pipeline, dispatch, dtype/layout), derived references (which wiki/artifact cite it), reproducibility note. **No cross-source synthesis, no tutorial.**

### Wiki page(s)

Place by content type: per-operator semantics/tiling/abstraction → `wiki/kernels/`; reusable methods (online softmax, tail mask, FP32 accumulation) → `wiki/techniques/`; error/debug/agent-selection → `wiki/patterns/`; Triton DSL semantics → `wiki/languages/`; arch constraints → `wiki/hardware/`. `id` prefix is enforced: `kernel-`, `technique-`, `pattern-`, `lang-`, `hw-`.

`wiki-kernel` required frontmatter (from `schemas.yaml`):
```yaml
id: kernel-<operator>
title: ...
type: kernel
tags: [...]                  # free-form, NOT vocab-checked, but keep CANN-aligned
techniques: [...]             # from tags.yaml#techniques
hardware_features: [...]      # from tags.yaml#hardware_features
architectures: [...]          # EXACT chip models from tags.yaml#architectures
kernel_types: [<OPERATOR_TYPE>]   # from tags.yaml#kernel_types
operators: [<operator>]
symptoms: [...]              # from tags.yaml#symptoms
confidence: verified|source-reported|inferred|experimental   # static analysis ⇒ usually inferred
reproducibility: concept|pseudocode|snippet|runnable|benchmarked  # ≥ snippet per schema
artifact_dir: artifacts/<bundle>
related: [technique-..., ...]
sources: [<source-page-id>]  # MUST resolve to an existing source page id
performance_claims: []       # [] unless you have baseline+env+shapes+repeats+result
```
`wiki-technique` requires additionally `prerequisites`; `wiki-pattern` requires `symptoms` + `candidate_techniques`; `wiki-hardware` requires `aliases`; `wiki-language` requires `reproducibility`. Always check `schemas.yaml`.

### Artifact bundle

`artifacts/<bundle>/` with `PROVENANCE.yaml` (scaffolder creates both). PROVENANCE records `origin_url`, `upstream_repo`, `upstream_sha`/`upstream_commit`, `license`, `retrieved_at`, `asset_mode` (`verbatim|extracted|derived` — derived material uses `derived` or `extracted`), `derived_from`, and `files[]` with `{local_path, role, mode, sha256}` for every file. Schema-canonical `role` values: `upstream-file|extracted-block|derived-source|approach-notes|bench-record`; `mode`: `verbatim|extracted|derived`. After editing any bundle file, recompute sha256:

```bash
python .claude/skills/triton-kb-cann-ingest/scripts/scaffold.py update-provenance \
  --root "$KERNELWIKI_ROOT" --bundle <bundle>
```

## S9 · Validate & query (report §12)

```bash
cd "$KERNELWIKI_ROOT"
python kernelwiki/scripts/validate.py
python kernelwiki/scripts/generate-indices.py
python kernelwiki/scripts/query.py <OPERATOR_NAME> --compact
python kernelwiki/scripts/query.py <OPERATOR_TYPE> tiling --compact
python kernelwiki/scripts/query.py <OPERATOR_TYPE> mask --compact
python kernelwiki/scripts/query.py <OPERATOR_TYPE> triton ascend --compact
```

`validate.py` enforces (so fix before finishing): unique non-empty `id`; no `source_pages`/`languages` fields; all schema-required fields present per page type; every `sources:` entry resolves to a real source page id; `architectures/hardware_features/techniques/kernel_types/symptoms` values are in `tags.yaml`; every referenced `artifact_dir` has `PROVENANCE.yaml`; every PROVENANCE `files` entry exists and its sha256 matches. If the env can't run the scripts, don't fix the env — just record "not run, reason, commands, expected checks".

## S10 · Final report (report §1–§14)

Fill `reports/<operator>-ingest-report.md` from [references/report-template.md](references/report-template.md). It must include: task params (mark each field source-confirmed / inferred / needs_verification), source inventory, semantics, stage table, tiling table, core-partition, template+dispatch count, pipeline analysis, migration table, Triton skeleton, written KB page list, validate/query results, unverified items, and the final modified-files list.

## Governance red lines (`static_ingest`)

- No demo, no kernel run, no correctness check, no profiler/benchmark. Never claim "verified"/"passed"/"benchmarked".
- `confidence` for static analysis is almost always `inferred` (or `source-reported` if the source states it). Set wiki `reproducibility` to a `tags.yaml` vocab value (`concept`/`pseudocode`/`snippet`/`runnable`/`benchmarked`): use `pseudocode` or `snippet` for skeletons. Source pages carry no `reproducibility` field (see their minimal frontmatter above) - do not invent a `source` value, it is not in the vocab.
- `performance_claims: []` unless you have baseline + env + shapes + repeats + measured result — you will not, in this mode.
- Every claim traces to a source page or source path+lines. Inferences are labeled `inferred`; unverifiable items `needs_verification`.
- Do not transfer CANN-specific mechanism (MTE/BufferID/queue/event/manual Cube·Vector sync/ND↔NZ moves) as Triton-Ascend implementation guidance — abstract them.
- Never edit the controlled-vocab YAMLs or `schemas.yaml`.
- Do not bypass review: this skill writes candidate content + validates. If the user wants governed publishing, hand off to `triton-kb-source-builder`.

## Priority

1. static semantic accuracy → 2. source traceability → 3. KernelWiki spec correctness → 4. reasonable Triton-Ascend abstraction → 5. retrievable/reusable by later agents → 6. performance notes. Never trade accuracy or traceability for speed.
