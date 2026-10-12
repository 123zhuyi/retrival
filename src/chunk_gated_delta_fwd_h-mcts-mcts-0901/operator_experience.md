# Operator Experience: chunk_gated_delta_fwd_h (mcts/mcts-0901)

## Failures / Avoid
<!-- Sub-agent appends here after each round -->
- (R5) multibuffer=False on the fused chunk-recurrence kernel REGRESSES -5.9% geomean (6.8053x vs
  7.2335x anchor, verify 8/8 both). The serial in-kernel chunk loop relies on the compiler's
  double-buffered load/compute overlap to hide next-chunk operand fetches behind the current chunk's
  Cube work; disabling it makes serial-heavy shapes (varlen T=32768: 256-512 chunks/program) pay the
  full load latency per step. Keep multibuffer=True (with unit_flag=True) — this is now the second
  independent negative probe (cf. num_warps=8 in R4) confirming the launch dict is at its optimum.
- (R2) Do NOT shrink BV for large chunk_size purely for UB conservatism. Parent heuristic
  `BV=64 if BT<=64 else 32` left 1.46-1.72x on cs=128 shapes (BV=32). Sweep the ACTUAL kernel
  empirically before trusting tile-size heuristics. BV=128 never beats BV=64 for K=V=128 on 910b1.
- (R2) Triton-Ascend num_warps/num_stages autotune is UNSUPPORTED (validated fixed at 4). Manual
  triton.Config BV autotune runs on dot kernels but reproduces BV=64 with ~7s/shape overhead — prefer
  hand-tuned constexpr tile.

## Successful Patterns
<!-- Sub-agent appends here after each round -->
- (R5) Independent 3rd root-child re-anchor reproduces the champion EXACTLY: byte-identical 6402eb
  golden rebuild + champion recipe gives 7.2335x geomean (verify 8/8, benchmark 8/8) in a fresh
  work_dir — a clean reproducibility anchor under the virtual root (R2 7.33x / R4 7.1132x / R5 7.2335x
  all within shared-NPU drift of each other). AST Type-3 fix that keeps GQA clean: compute the
  hv->key-head map in-kernel (k_head from pid: dense `(i_bh//HV)*HK+(i_bh%HV)//HR`, varlen `i_h//HR`)
  and pass k pre-flattened [B*HK | HK(varlen), T, K] — no host torch.index_select/torch compute.
- (R2) Fused chunk-recurrence GDN forward-H: S[K,BV] fp32 register-resident, serial in-kernel chunk
  loop, host pre-flatten to [BH,T,D], seq-in-grid varlen. **Universal V-tile BV=min(64,V)** is fastest
  on every tested shape (dense+varlen, cs 64/128). Keep multibuffer=True/unit_flag=True and fp32-ieee
  dot. Round 2 geomean 7.33x (verify 8/8, benchmark 8/8), target 5x reached.

## Notes

## Round 3 — root-expansion fused chunk-recurrence reconstruction
- direction: fused chunk-recurrence GDN fwd-H (independent reconstruction vs 6402eb golden)
- best_speedup: 6.95x geomean (verify 8/8, benchmark 8/8), target 5x reached
- success: reconstructing the proven recipe from scratch reproduces ~6.9x on the first
  kernel; no need to inherit parent code. constexpr K/V (K=V=128 in all cases) adds ~1%.
- avoid: do NOT rely on the global_baseline torch golden (0c4c01) — broken varlen/NaN.
  Re-anchor to the 6402eb golden (SAFE_CLAMP + per-seq batch_idx=0 + post-update state store).
- reusable: in-kernel serial prefix over cu_seqlens computes packed chunk offsets for
  varlen WITHOUT host python loops (keeps AST Type-3 clean); SAFE_CLAMP must be a tl.constexpr;
  scalar loop-carried accumulators should be int32 to avoid int32->int64 type errors.

## Round 4 — MCTS continuation of R3 champion (num_warps probe + scan re-confirm)
- direction: re-anchor/optimize the R3 champion sha 0d543b7e against the 6402eb golden
- best_speedup: 7.1132x geomean (verify 8/8, benchmark 8/8), target 5x reached (improvement_made false)
- success: byte-identical champion re-verifies/re-benchmarks reproducibly in a fresh work_dir
  (7.1132x vs R3 6.95x, within ~±2% shared-NPU drift). R4 re-anchor is a clean reproducibility anchor.
- avoid: num_warps=8 on this serial chunk-recurrence kernel REGRESSES (-4.6% geomean; worst
  -8.5%..-9.7% on varlen T=32768 heavy cases 7/8). More warps per program only adds sync/barrier
  cost — each program is ONE serial chunk chain with no intra-program parallelism to absorb warps.
  num_warps=4 is optimal (R2/R3/R4 all confirm).
- avoid: do NOT re-litigate the serial-recurrence split each round. T/chunk axis is strict serial
  (S depends on previous chunk); grid = num_seq*HV x num_vt(2) is maximal-safe. Per-chunk-step
  cost ~0.43-1.2us is uniform across dense+varlen -> recurrence-bound, not a pipe imbalance.
- avoid: BV=32 to add V-tiles is a false lever for weak cases 3/4 (B=1,HV=48 -> 96 programs).
  It halves per-program dot-N and adds redundant S traffic; R2 sweep already proved BV=64 optimal.
- reusable: a "re-anchor + one untested single-variable probe + scan re-confirm" round is the
  correct MCTS exploitation step when the parent already exceeds target — it makes the node
  independently reproducible and records a new negative data point (num_warps=8) cheaply.

## Round 5 — MCTS 3rd root child (re-anchor + multibuffer probe)
- direction: independent re-anchor of the champion fused chunk-recurrence recipe vs 6402eb golden;
  Phase-4 probe multibuffer=False
- best_speedup: 7.2335x geomean (verify 8/8, benchmark 8/8), target 5x reached (improvement_made false)
- success: byte-identical 6402eb rebuild + champion recipe verifies 8/8 and benchmarks 7.2335x in a
  fresh work_dir — the R2/R3/R4 champion is now triply reproducible under root (R3 6.95 / R4 7.11 /
  R5 7.23, ~±2% shared-NPU drift). multibuffer=True re-confirmed optimal via negative probe.
- avoid: multibuffer=False regresses -5.9% (see Failures/Avoid). No remaining launch-flag lever;
  the launch dict (multibuffer=True, unit_flag=True, nw=4) is at its optimum across R2/R4/R5 probes.
- avoid: global_baseline 0c4c01 remains broken (varlen/NaN) — always re-anchor to the documented
  6402eb golden for root-expansion rounds (mode=auto reconstruction is the R3/R5 precedent).
- reusable: weak shapes (B=1,HV=48,T=8192) at ~4.9x are parallel-program-count bound (96 programs,
  serial T/chunk axis unsplittable); varlen heavy shapes ~5.7-6.2x are serial-chunk-count bound.
  These two bounds are the framework ceiling — 5x target is met but pushing past ~7.3x geomean would
  require breaking the serial-recurrence math, which the framework's frozen-golden contract forbids.
