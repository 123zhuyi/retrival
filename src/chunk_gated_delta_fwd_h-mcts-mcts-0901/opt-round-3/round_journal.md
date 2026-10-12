# Round 3 Journal — chunk_gated_delta_fwd_h (MCTS)

## MCTS Context
- parent_id: root (virtual root; mean_reward 1.866, visit_count 2, pending_expansion true)
- expansion motivation: fresh branch off root; global_baseline torch golden (0c4c01) is broken (varlen crash / dense NaN), so freeze re-anchored to the authoritative 6402eb golden that prior verified rounds converged to.

## Design Rationale
- Target direction: fused chunk-recurrence GDN fwd-H kernel — S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, grid (num_seq*HV, 2), GQA head-group select (khead = hv//ratio), in-kernel varlen chunk-offset prefix, BLOCK_L=chunk_size, launch multibuffer=True/unit_flag=True, fp32-ieee dots.
- Why chosen: proven recipe in operator_experience (R2 geomean 7.33x). BV=64 universal best on 910b1. No parent Triton kernel visible from root, so independent reconstruction against 6402eb semantics.

## Iteration Log

### Iter 1 (coding)
- Attempt: initial kernel (runtime K/V masks, host varlen offsets via python loop + list.append).
- Result: AST precheck Type-3 (for-loop + list.append in forward); kernel compile errors (non-constexpr global SAFE_CLAMP; int32/int64 loop-carried boh).
- Fix: removed host python varlen loop (in-kernel serial prefix over cu_seqlens), constexpr SAFE_CLAMP param, int32 scalar loop accumulator. AST valid; verify 8/8.
- Baseline perf: geomean 6.8655 (all >=4.71x), target 5x reached at iter_0.

### Opt-Iter 1 (latency optimization)
- Attempt: optimization point 1 (入参静态化) — declare K,V as tl.constexpr (all cases K=V=128), folding inner-dim masks.
- Result: geomean 6.9508 (verify 8/8). Marginal +1.2% (within noise; fewer freq drifts in opt run; case 8 dip to 5.54 likely device-6 1800->800 drift at sample #79).
- Decision: keep (speedup > baseline per Phase 4.7.4).

### Opt-Iter 2 (point scan)
- Points 2-31: no clean hit. Grid is maximal (chunk axis is serial recurrence, cannot split); autotune unsupported (R2); kernel-split gate (speedup<2.0) not met; scalar g_last inherent. Block-size scaling confirms BV=64 optimum.
- Decision: no more optimization points; target_reached -> exit gate.

## Final Outcome
- best_speedup: 6.9508
- effective_metric_source: kernel (_gdn_chunk_fwd_h_kernel profiler op latency)
- outcome: kept (opt_iter_0 optimized_code.py; sha 0d543b7e)
- verify: 8/8
