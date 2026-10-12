# Round 3 Report — chunk_gated_delta_fwd_h (MCTS)

## Basic Info
- **arch**: ascend910b1
- **work_dir**: triton_ascend_output/chunk_gated_delta_fwd_h-mcts-mcts-0901/opt-round-3
- **round_index**: 3
- **algorithm**: mcts
- **direction**: pattern: custom fused chunk-recurrence GDN fwd-H kernel (S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, V-tile grid split, GQA head-group index-select, in-kernel varlen seq-prefix chunk offset)
- **hypothesis**: MCTS virtual-root pending expansion: independently reconstruct the proven fused chunk-recurrence recipe against the freeze-anchored 6402eb golden semantics, then optimize toward 5x.
- **baseline_dir**: global_baseline (speedup 1.0)

## MCTS Context
- parent_id: root
- parent_mean_reward: 1.8662237404570077
- parent_visit_count: 2
- selection_path_depth: 1
- mcts_pending_expansion: true
- chosen_direction: pattern: custom fused chunk-recurrence GDN fwd-H kernel
- direction_rationale: virtual-root pending expansion; the global_baseline torch golden (sha 0c4c01) is broken on varlen/NaN and is NOT a usable Triton baseline. Freeze re-anchored to the authoritative 6402eb golden (same semantics prior opt-round-1/2 verified against). Since no parent Triton kernel is in view, this round independently reconstructs the proven fused chunk-recurrence recipe with BLOCK_V=64 (R2-proven universal best), in-kernel varlen chunk-offset prefix to stay AST Type-3 compliant, and BLOCK_L=chunk_size.

## Generation Result
- **Phase 3 iterations**: 1 (iter_0)
- **Phase 4 iterations**: 1 (opt_iter_0: constexpr K/V static-parameter optimization)
- **Final version source**: opt_iter_0/optimized_code.py
- **Verify (final)**: passed 8/8 (opt_iter_0/verify/verify_result_optimized.json)
- **Perf method**: profiler (benchmark.py warmup=5 repeats=50)

## Target vs Actual
- **target_speedup**: 5.0
- **best_speedup**: 6.9508 (geomean speedup_vs_torch)
- **target_reached**: true
- **improvement_made**: true (baseline 6.8655 -> optimized 6.9508)

## Performance Detail (final: opt_iter_0)
| case | shape (B,HK,HV,T,cs,var) | framework ms | impl ms | speedup | status |
|---|---|---|---|---|---|
| 1 | 3,10,10,1054,64,dense | 5.823 | 0.445 | 13.40 | pass |
| 2 | 3,10,10,1054,128,dense | 3.512 | 0.283 | 13.05 | pass |
| 3 | 1,48,48,8192,64,dense | 25.047 | 5.323 | 4.79 | pass |
| 4 | 1,48,48,8192,128,dense | 15.267 | 3.241 | 4.71 | pass |
| 5 | 1,8,24,4096,64,dense | 9.953 | 1.525 | 6.66 | pass |
| 6 | 1,8,24,4096,128,dense | 5.593 | 0.935 | 6.15 | pass |
| 7 | 4,32,32,32768,64,varlen | 84.211 | 13.891 | 6.10 | pass |
| 8 | 4,32,32,32768,128,varlen | 48.322 | 8.347 | 5.54 | pass |

geomean speedup = 6.9508; all 8 shapes >= 4.7x.

## Code Path
- final: `chunk_gated_delta_fwd_h_generated.py` (= output/optimized_code.py, sha 0d543b7e)
- generated (Phase 3 baseline): output/iter_0/generated_code.py
- optimized (Phase 4): output/opt_iter_0/optimized_code.py

## Notes
- Reference golden re-anchored to sha 6402eb (working; global_baseline 0c4c01 is broken varlen/NaN). Baseline gate anchor file output/.baseline_anchor.json records 6402eb.
- Shared-NPU freq drift (devices 6/7 1800->800 MHz) observed during some benchmark cases; per-shape speedups have measurement variance ~±5%.
- Weakest shapes remain cases 3/4 (B=1,HV=48 => 96 programs only, dense T=8192) at ~4.7x — parallel-program-count bound, not addressable without breaking the serial recurrence.
