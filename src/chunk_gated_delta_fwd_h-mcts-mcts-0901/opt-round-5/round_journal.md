# Round 5 Journal

## Design Rationale
- Parent: root (visit 4, mean_reward 1.908). Third root child. Target direction: independently re-anchor the champion fused chunk-recurrence recipe against the freeze-anchored 6402eb golden (NOT the broken 0c4c01 global_baseline copy), then run ONE genuinely untested launch probe.
- Why chosen: root exploitation. R4 covered num_warps; multibuffer was never A/B'd. A reproducible third root child plus a new negative data point is the cheap, correct MCTS move. Recipe (from operator_experience + run-root final): S[K=128,BV=64] fp32 register-resident, serial chunk loop, V-tile grid split (NVT=2), in-kernel GQA key-head map, in-kernel varlen seq-prefix chunk offset, multibuffer/unit_flag launch, fp32-ieee dots, nw=4.
- Algorithm sketch: single fused kernel, grid (n_seq|B)*HV*NVT, pid→(bh, vt), per-program serial chunk loop.

## Iteration Log

### Iter 0 (coding)
- Attempt: reconstructed champion from run-root final.py (legit read-only lineage artifact) + applied BV=min(64,V). AST precheck flagged host `torch.index_select` (GQA) as Type-3 → moved key-head map into kernel: k_head = dense (i_bh//HV)*HK+(i_bh%HV)//HR, varlen i_h//HR; pass k flattened [B*HK|HK, T, K].
- Result: verify 8/8 (iter_0). benchmark geomean 7.2335x (repeats 20), target 5x REACHED.

### Opt-Iter 0 (latency optimization)
- Attempt: single-variable probe multibuffer=False (only launch change; nw=4, unit_flag=True held).
- Result: verify 8/8 but benchmark 6.8053x < 7.2335x anchor (-5.9%). Regression consistent across suite; 1 freq-drift sample on device 7 late (case 8) does not change direction.
- Decision: REVERT to anchor (multibuffer=True). Record negative data point. improvement_made=false; target_reached=true.

## Final Outcome
- best_speedup: 7.2335x
- effective_metric_source: kernel
- outcome: kept anchor (multibuffer=True). Champion-equivalent reproducible under root; new negative data point (multibuffer=False) recorded.
