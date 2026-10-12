# Round 5 Report — chunk_gated_delta_fwd_h (MCTS root expansion)

## Basic Info
- arch: ascend910b1 (device 0)
- work_dir: triton_ascend_output/chunk_gated_delta_fwd_h-mcts-mcts-0901/opt-round-5
- round_index: 5
- algorithm: mcts
- direction: pattern: custom fused chunk-recurrence GDN fwd-H kernel (S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, V-tile grid split, in-kernel GQA key-head map, in-kernel varlen seq-prefix chunk offset) — R5 root-expansion re-anchor vs 6402eb golden; Phase-4 probe multibuffer=False
- hypothesis: MCTS selects the virtual root for a new expansion. R5 independently re-anchors the documented champion recipe against the freeze-anchored 6402eb golden, then probes ONE genuinely untested launch variable (multibuffer False).

## MCTS Context
- parent_id: root
- parent_mean_reward: 1.9083
- parent_visit_count: 4
- selection_path_depth: 1 (root only)
- pending_expansion: true
- chosen_direction: independent re-anchor of the champion fused chunk-recurrence recipe + multibuffer probe
- direction_rationale: Root's two prior children (R1-chain 7.33x, R3-chain 7.11x) both exceed the 5x target with the same algorithm family. Per operator_experience.md, the correct MCTS exploitation step is "re-anchor + one untested single-variable probe" to create a reproducible third root child and record a NEW negative data point. R4 already probed num_warps; the launch dict's `multibuffer` flag was the only never-tested variable.

## Result
- target_speedup: 5.0
- best_speedup: **7.2335x** (geomean vs global baseline torch reference)
- target_reached: **true**
- improvement_made: false (Phase-4 multibuffer probe regressed; reverted to anchor)
- verify: 8/8 passed (anchor iter_0 and optimized opt_iter_0 both 8/8)
- benchmark: 8/8 measured
- final code: chunk_gated_delta_fwd_h_generated.py (anchor, multibuffer=True, unit_flag=True, nw=4)

## Baseline / Golden note
- Freeze anchor sha: 6402eb836da7... (= the documented working golden; the run-root global_baseline copy 0c4c01 is the known-broken varlen/NaN variant and was NOT used). Baseline task rebuilt byte-identical to 6402eb in this fresh work_dir. Source_mode=auto (root-expansion reconstruction; no in-manifest user source to point at, matching the R3 precedent).
- All 8 task shapes have use_initial_state=False, output_final_state=False, K=V=128 bf16.

## Generation Result
- Phase 3 iterations: 1 (iter_0). AST Type-3 gate required moving the GQA `torch.index_select` from host into the kernel (k_head computed from pid); fixed before first verify.
- Phase 4 iterations: 1 (opt_iter_0).

## Performance detail (anchor, iter_0 perf_result.json; repeats=20)
| case | shape (B,HV,T,cs,varlen) | speedup_vs_torch |
|---|---|---|
| 1 | 3,10,1054,64 | 13.8199 |
| 2 | 3,10,1054,128 | 14.3627 |
| 3 | 1,48,8192,64 | 4.9024 |
| 4 | 1,48,8192,128 | 4.9527 |
| 5 | 1,24,4096,64 (GQA) | 6.7714 |
| 6 | 1,24,4096,128 (GQA) | 6.4967 |
| 7 | 1,32,32768,64 varlen | 6.1826 |
| 8 | 1,32,32768,128 varlen | 5.7181 |
- geomean 7.2335x, avg latency framework 14.5203 ms / impl 2.0074 ms

## Key fixes / findings
1. Re-anchored golden to the working 6402eb reference; verify 8/8 clean in a fresh work_dir (independent 3rd root child reproduction).
2. Moved GQA key-head map fully into the kernel (k_head = (i_bh//HV)*HK+(i_bh%HV)//HR dense; i_h//HR varlen) to satisfy the AST Type-3 gate — no host torch compute.
3. Phase-4 multibuffer=False probe: verify 8/8 but benchmark 6.8053x < anchor 7.2335x (-5.9%) → reverted. multibuffer=True confirmed optimal.

## Bottlenecks observed
- Weak shapes are cases 3/4 (B=1,HV=48,T=8192): only 96 programs → parallel-program-count bound ~4.9x (serial T/chunk recurrence can't split safely).
- varlen heavy cases (T=32768, cs=64/128) at 5.7-6.2x are serial-chunk-count bound (256-512 chunks/program).
- multibuffer=False adds latency on serial-heavy shapes (no double-buffered prefetch of next chunk while current chunk computes).

## Code path
- chunk_gated_delta_fwd_h_generated.py
