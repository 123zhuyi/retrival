# Migration rubric: CANN -> Triton-Ascend

Classify every knowledge item from the CANN source into exactly one bucket. The bucket decides how it is written into the KB. When an item spans buckets, split it: e.g. "online softmax algorithm" (transferable) vs. "FlashSoftmax UB rescaling buffer layout" (CANN-specific).

## 1. Directly transferable

Mathematical semantics and correctness rules that are architecture-independent. Carry these into `wiki/kernels/` and `wiki/techniques/` verbatim (paraphrased, cited).

- Operator formula (`y=abs(x)`, RMSNorm, `softmax(QK^T)V`, GEMM).
- Boundary / tail rules (last block smaller than tile -> mask).
- Mask / predicate logic (causal mask, atten_mask select-to-negative-infinity, padding mask).
- Online softmax / online reduction (running max+sum, `exp(max_prev - max_cur)`, rescale accumulator).
- FP32 accumulation for BF16/FP8 (cast up, compute, cast back).
- Numerical-stability algorithms (complex abs `sqrt(max^2+min^2)` with normalize-to-[1,2), subnormal handling).
- Shape/dtype promotion rules (COMPLEX64 -> FLOAT, etc.).
- Variants taxonomy (forward / backward / inference / incremental / sparse).

## 2. Rewrite to Triton-Ascend semantics

CANN scheduling decisions that must be *mapped*, not copied, onto Triton primitives. Record as a mapping table in the artifact bundle.

| CANN concept | Triton-Ascend abstraction |
|---|---|
| `blockNum` / core id | `grid` size + `tl.program_id(axis)` |
| `blockDim`, AIC/AIV core split | program grid shape; AIV-only ops -> 1D grid |
| `block_m/block_n/block_k`, C-Tiling/V-Tiling base blocks | `tl.constexpr` meta (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) |
| tiling key -> template selection | `if constexpr` / dispatch in launcher, or separate `@triton.jit` kernels |
| `EleBaseTilingData` UB partition | on-chip tile size chosen by `tl.constexpr`; UB managed by compiler |
| tail block (`blockNum * BLOCK < total`) | `mask = offsets < n_elements` on load/store |
| workspace (mm temp, 100MB empty) | intermediate `tl` tensor, or drop (Triton manages temps) |
| `KERNEL_TYPE_AIV_ONLY` | single program family on AIV |
| `TilingPrepare` platform query (coreNum/ubSize) | launcher picks `BLOCK_*` from shape; runtime query not needed in skeleton |
| arch32 multi-template vs arch35 single-template | note as a tiling-key/branch decision; Triton picks `BLOCK_*` per shape |

## 3. CANN-specific, do NOT transfer

Low-level hardware/runtime mechanisms. Mention them only as "what CANN does and why we abstract it away" - never as Triton implementation steps.

- MTE (Memory Transfer Engine) manual `DataCopy` / `CopyGMToUB` / `CopyUBToGM`.
- BufferID synchronization, `SetFlag`/`WaitFlag`, TQue/TBuf event ordering.
- Manual Cube(AIC)/Vector(AIV) pipeline staging, `IterateBmm1`/`ProcessVec1` handoff timing, `nRatio` CV data-ratio tuning.
- ND <-> NZ / ZN physical format conversion and the associated `DataCopyPad` offsets.
- TBuf `Get<TPosition>` placement (UB/L1/L0A/L0B/L0C), `TID`/`PIPE_ID` enums.
- `ascendcPlatform` runtime capacity queries used to size CANN buffers.
- Binary hash / `*_binary.json` / `simplified_key.ini` selection (compile-time registry, not algorithm).

## 4. Needs later verification

Anything the static source cannot settle, or that depends on a runtime/hardware environment. Mark `needs_verification` in the report; set the wiki page `confidence: inferred` (or `experimental`) and do not assert it.

- Exact tiling parameter *values* (e.g. "S1.i=128") - present as source-reported, do not claim optimal.
- UB/L1/L0C *byte budgets* and occupancy.
- Precision boundaries for FP8 / BF16 paths without a runtime test.
- Specific `aclnn*` API contract details only visible in runtime headers.
- Layout offset arithmetic for TND/BSH packing that needs a tensor to confirm.
- Performance: never assert a speedup without baseline+env+shapes+repeats+result (absent in `static_ingest`).

## Classification table (drop into report §9 / artifact `migration-table.md`)

```markdown
| Knowledge item | CANN source (path Lxx) | Triton-Ascend handling | Bucket | Confidence | Needs verification |
|---|---|---|---|---|---|
| online softmax rescale | FA算子设计介绍.md L11-18 | running max/sum + exp rescale in K loop | transferable | inferred | no |
| arch32 6-template dispatch | flash_attention_score.cpp L380-460 | choose BLOCK_M/N by shape; one kernel | rewrite | inferred | yes (param values) |
| MTE double-buffer ping-pong | FA算子设计介绍.md L40-55 | tile reuse / fused load; not manual MTE | CANN-specific | inferred | no |
| S1.i=128 optimal on arch35 | tiling table | constexpr BLOCK_M; value source-reported | needs verification | inferred | yes |
```
