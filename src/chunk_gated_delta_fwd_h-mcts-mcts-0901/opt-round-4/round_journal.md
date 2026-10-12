# Round 4 Journal — chunk_gated_delta_fwd_h (MCTS)

## MCTS Context
- parent_id: opt-round-3 (mean_reward 1.9389, visit_count 1)
- expansion motivation: MCTS expands from the high-reward opt-round-3 node. Parent already
  reached target (6.95x); residual weakness is parallel-program-count bound on B=1,HV=48 dense
  cases. R4 re-verifies the R3 champion (sha 0d543b7e) in a fresh work_dir and probes the one
  launch variable R3 never measured (num_warps), then re-confirms no safe recurrence split exists.
- contrast vs selection path: root (visit 3, mean 1.89) vs opt-round-3 (visit 1, mean 1.94);
  this node is a depth-2 exploitation of the highest-mean child of root.

## Design Rationale
- Target direction: R4 continuation of the R3 fused chunk-recurrence GDN fwd-H kernel —
  S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, grid (num_seq*HV,2),
  GQA head-group select, in-kernel varlen seq-prefix, fp32-ieee dots, num_warps=4.
- Why chosen: only known-good recipe (R2 7.33x, R3 6.95x, both verify 8/8). MCTS exploitation
  should not invent a new algorithm family when the champion already exceeds 5x; instead it
  should (a) re-anchor so the R4 node is independently reproducible, and (b) test the single
  untested cheap variable (num_warps=8) plus re-confirm the exhaustive scan.

## Iteration Log

### Iter 1 (coding / re-anchor)
- Attempt: byte-copy R3 champion (0d543b7e) into R4 work_dir; freeze baseline against 6402eb.
- Result: AST clean; verify 8/8; benchmark geomean 7.1132x (all >= 4.7x). Target 5x reached.

### Opt-Iter 1 (latency probe: num_warps 4->8)
- Attempt: only launch-parameter change not measured in R3 (R3 asserted nw=4 by experience).
- Result: verify 8/8 (numerics identical) but geomean 6.79x (recorded; a parallel first run 6.85x — REGRESSION vs 7.1132, -4.6%).
  Per-case -0.7%..-9.7%; worst on varlen serial-heavy cases 7/8 (-9.7%/-8.5%): more warps per
  program adds sync/barrier cost without adding parallel work (one serial chain per program).
- Decision: REVERT to num_warps=4. Records num_warps=8 as a Pareto-avoid option.

### Lever re-analysis (after probe)
- Re-confirmed R3 exhaustive 31-point scan on the byte-identical champion: none cleanly hit.
  Per-chunk-step cost is uniform (~0.43-1.2 us) across dense AND varlen -> serial-recurrence
  bound, not memory/cube imbalance. BV=32 would add V-tiles but halve dot-N per chunk and add
  redundant S traffic (R2 measured slower). GQA head-merge only helps cases 5/6 and adds register
  pressure; not worth it. Grid = num_seq*HV x num_vt is maximal safe parallelism.

## Final Outcome
- best_speedup: 7.1132 (geomean), verify 8/8, target_reached true, improvement_made false.
- effective_metric_source: kernel (_gdn_chunk_fwd_h_kernel profiler op latency).
- outcome: kept Phase 3 anchor (sha 0d543b7e, num_warps=4); num_warps=8 probe abandoned.
