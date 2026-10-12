# Round 4 Report — chunk_gated_delta_fwd_h (MCTS)

## Basic Info
- **arch**: ascend910b1
- **work_dir**: triton_ascend_output/chunk_gated_delta_fwd_h-mcts-mcts-0901/opt-round-4
- **round_index**: 4
- **algorithm**: mcts
- **direction**: pattern: custom fused chunk-recurrence GDN fwd-H kernel (S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, V-tile grid split, GQA head-group index-select, in-kernel varlen seq-prefix chunk offset) — R4 MCTS continuation of the R3 champion (0d543b7e)
- **hypothesis**: MCTS expands from node opt-round-3 (mean 1.9389); re-verify the R3 champion in a fresh work_dir and probe remaining launch/tile levers
- **baseline_dir**: opt-round-3 (global baseline speedup 1.0)

## MCTS Context
- parent_id: opt-round-3
- parent_mean_reward: 1.9388567608659624
- parent_visit_count: 1
- selection_path_depth: 2
- mcts_pending_expansion: true
- chosen_direction: pattern: custom fused chunk-recurrence GDN fwd-H kernel (R3-champion continuation)
- direction_rationale: MCTS selects the high-reward opt-round-3 node. Parent already reached 6.95x; the residual weakness is parallel-program-count bound on the B=1,HV=48 dense cases (96 programs). R4 independently re-verifies the 7.11x champion against the re-anchored 6402eb golden, then probes the one launch variable R3 never measured (num_warps=8) and re-confirms no safe split exists for the strict serial chunk recurrence.

## Generation Result
- **Phase 3 iterations**: 1 (iter_0 — R3 champion re-anchored and re-verified in R4 work_dir)
- **Phase 4 iterations**: 1 (opt_iter_0: num_warps 4->8 probe, regressed, reverted)
- **Final version source**: iter_0/generated_code.py (sha 0d543b7e, num_warps=4)
- **Verify (final)**: passed 8/8 (iter_0/verify/verify_result.json)
- **Perf method**: profiler (benchmark.py warmup=5 repeats=50)

## Target vs Actual
- **target_speedup**: 5.0
- **best_speedup**: 7.1132 (geomean speedup_vs_torch)
- **target_reached**: true
- **improvement_made**: false (num_warps=8 probe regressed; final = anchor)

## Performance Detail (final: iter_0 anchor)
| case | shape (desc) | status | framework ms | impl ms | speedup |
|---|---|---|---|---|---|
| 1 | 3x10x1054x128,3x10x1054x128,3x10x1054x128,3x10x1054… | pass | 5.9361 | 0.4429 | 13.4028 |
| 2 | 3x10x1054x128,3x10x1054x128,3x10x1054x128,3x10x1054… | pass | 3.6082 | 0.2759 | 13.0779 |
| 3 | 1x48x8192x128,1x48x8192x128,1x48x8192x128,1x48x8192… | pass | 25.4043 | 5.2094 | 4.8766 |
| 4 | 1x48x8192x128,1x48x8192x128,1x48x8192x128,1x48x8192… | pass | 15.2205 | 3.2134 | 4.7366 |
| 5 | 1x8x4096x128,1x24x4096x128,1x24x4096x128,1x24x4096… | pass | 10.0659 | 1.5010 | 6.7061 |
| 6 | 1x8x4096x128,1x24x4096x128,1x24x4096x128,1x24x4096… | pass | 5.7225 | 0.9312 | 6.1453 |
| 7 | 1x32x32768x128,1x32x32768x128,1x32x32768x128,1x32x32768… | pass | 93.8879 | 14.0356 | 6.6893 |
| 8 | 1x32x32768x128,1x32x32768x128,1x32x32768x128,1x32x32768… | pass | 49.1162 | 8.3640 | 5.8723 |

geomean speedup = 7.1132; all 8 shapes >= 4.7x.

## Phase 4 Experiment Detail
- **opt_iter_0 (num_warps 4->8)**: verify 8/8, geomean **6.79x** (anchor 7.1132x, -4.6%; a parallel first run measured 6.85x, within run noise). Worst on varlen serial-heavy cases 7/8 (-8.5%/-9.7%). Reverted to num_warps=4. This was the only launch/tile variable never measured on this exact kernel (R3 asserted nw=4 by experience).
- **Lever re-analysis**: R3 exhaustive 31-point scan re-confirmed on byte-identical champion. The T/chunk axis is a strict serial recurrence (grid = num_seq*HV x num_vt is maximal-safe); BLOCK_L=chunk_size and BLOCK_V=64 are empirical optima; GQA head-merge only helps 2/8 cases and adds register pressure; per-chunk-step cost is uniform ~0.43-1.2us across dense+varlen → recurrence-bound, not a pipe imbalance.

## Code Path
- final: `chunk_gated_delta_fwd_h_generated.py` (= output/iter_0/generated_code.py, sha 0d543b7e)
- optimized (Phase 4 probe, reverted): output/opt_iter_0/optimized_code.py

## Notes
- Baseline golden re-anchored to sha 6402eb in R4 work_dir (freeze mode=user, sha check passed). global_baseline 0c4c01 remains broken varlen/NaN and was NOT used.
- Shared-NPU freq drift (devices 4/5/6/7 1800->800 MHz) observed during some benchmark samples; per-shape speedups have ~±5% variance. Anchor and probe were interleaved under the same drift conditions.
- Weakest shapes remain cases 3/4 (B=1,HV=48 => 96 programs, dense T=8192) at ~4.7-4.9x — parallel-program-count bound, not addressable without breaking the serial recurrence.
