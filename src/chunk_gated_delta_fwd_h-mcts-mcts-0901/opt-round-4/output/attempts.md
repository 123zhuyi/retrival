# Round 4 optimization attempts — chunk_gated_delta_fwd_h (MCTS)

Phase 3 (iter_0) re-anchored and re-verified the R3 champion (sha 0d543b7e) in this round's
work_dir against the freeze-anchored 6402eb golden. Phase 4 (opt_iter_0) ran one additional
single-variable experiment (num_warps 4->8) and re-confirmed the exhaustive R3 31-point scan on
the byte-identical champion. This file records every point evaluated.

## Phase 3 attempt (direction = R4 MCTS continuation of R3 champion)

- [已尝试, 成功] Re-verified R3 champion (sha 0d543b7e): S[K=128,BV=64] fp32 register-resident,
  serial in-kernel chunk loop, grid (num_seq*HV, 2), GQA head-group select, in-kernel varlen
  seq-prefix chunk offsets, multibuffer=True/unit_flag=True, num_warps=4, fp32-ieee dots.
  verify 8/8, benchmark 8/8, geomean 7.1132x (target 5x reached at iter_0).

## Phase 4 single-variable experiment (opt_iter_0)

- [已尝试, 失败] Launch-parameter probe — num_warps 4 -> 8. verify 8/8 (numerics identical),
  benchmark geomean 6.79x (recorded; a parallel first run measured 6.85x — REGRESSION vs 7.1132x
  anchor, -4.6%). Worst on varlen serial-heavy cases 7/8 (-9.7%/-8.5%): more warps per program
  adds sync/barrier cost without adding parallel work (each program is one serial chunk chain).
  Decision: REVERT to num_warps=4. Confirms R3/R2 experience: nw=4 is optimal here.

## Phase 4 full point evaluation (R3 31-point scan re-confirmed on identical champion)

- [已命中, 已在 R3 应用] Point 1 入参静态化 — K,V already `tl.constexpr` (all 8 cases K=V=128);
  inner-dim K/V masks already fold away. No further static params available this round.
- [未命中] Point 2 Tiling 优化 — V-tile already split (BV=64); K/V tiles are row-contiguous
  [BLOCK_L,128] blocks; no strided access on a reduction axis. BLOCK_L = chunk_size is the
  recurrence boundary and cannot be enlarged.
- [未命中] Point 3 分核优化 — grid = num_seq*HV x num_vt(2). The T/chunk axis is a strict
  serial recurrence (S depends on previous chunk), so it cannot be split across programs without
  breaking correctness. Current grid is the maximal safe parallelism.
- [未命中] Point 4 离散访存 — inner loop uses affine row offsets; varlen bos/eos are per-program
  scalar loads (<=4 seqs), not an inner-loop gather.
- [未命中] Point 5 Scalar→Vector — scalar g_last per chunk is inherent to the recurrence
  boundary (single exp per chunk); no scalar reduction/accumulator in the hot loop.
- [未命中] Point 6 标量降级 — no i64 CMP/div inside the vectorized chunk loop; the varlen prefix
  loop is scalar over <=4 sequences once per program.
- [未命中] Point 7 Pass 消除合并 — S recurrence is already a single fused pass per chunk; no
  repeated traversal of the same data.
- [未命中] Point 8 维度合并 — inputs kept contiguous [.., T, D]; K/V chunk tiles are already
  dense row blocks (no 3-layer strided nesting to merge).
- [未命中] Point 9 Libdevice — exp uses native tl.exp; no hand-written math.
- [未命中] Point 10 循环不变量 — per-chunk w/k/u/g loaded once each iteration; nothing
  loop-invariant is re-loaded.
- [未命中] Point 11 Load 重排序 — multibuffer=True lets the compiler overlap loads with compute;
  no manual reorder available.
- [未命中] Point 12 Grid 多路径 — single kernel covers all 8 shapes verify 8/8; both small
  (60-program) and large (1024-program) shapes use the same optimal tile. A host dispatch split
  adds branching with no per-path algorithmic difference.
- [未命中] Point 13 Autotune — Triton-Ascend num_warps/num_stages autotune unsupported (R2
  validated fixed at 4); hand-tuned BV=64/BLOCK_K=128 is the empirical optimum.
- [未命中] Point 14 混合策略 — no shape/dtype-dependent kernel switch warranted; every case uses
  the same optimal tile config.
- [未命中] Point 15 dim-merge large block — not a normalization stat+apply operator.
- [未命中] Point 16 连续拷贝聚合 — not a pure memory-copy operator.
- [未命中] Point 17 冗余边界 — tail-chunk masks + other=0.0 are required for varlen / partial
  last chunks; K/V masks now compile away after Point 1.
- [未命中] Point 18 Kernel 分裂 — every case speedup_vs_torch >= 4.71 (gate < 2.0 not met), and
  the serial chunk recurrence cannot be split into independent kernels.
- [未命中] Point 19 Cube/MTE3 解耦 — no atomic scatter; single output writes per chunk.
- [未命中] Point 20 Host concat — no composite dot `a·c+b·d` split across segments.
- [未命中] Point 21 Workspace 解耦 — no multi-pass gather conflict; state is register-resident.
- [未命中] Point 22 Latency-bound tile merge — per-chunk dot M=BLOCK_L(64/128) already a full
  cube micro-tile; serial chunk iterations are the irreducible recurrence. BV=32 would add
  V-tiles but halve per-program dot-N and add redundant S traffic (R2 measured slower).
- [未命中] Point 23 Device gather 连续化 — no device-side random gather (GQA handled by affine
  head id, host index_select not needed).
- [未命中] Point 24 Chained matmul dtype — both dots are fp32-ieee accumulation into the fp32
  register state (required for precision); no fp16/bfloat16 downgrade safe.
- [未命中] Point 25 输出预初始化 — outputs fully written by the kernel (torch.empty); no
  default-value if/where fill.
- [未命中] Point 26 Interpolate — not an interpolate op.
- [未命中] Point 27 Pooling — not a pooling op.
- [未命中] Point 28 Matmul transpose — dots use tl.trans(k2) on the loaded tile, not a host
  transpose; layout already optimal.
- [未命中] Point 29 CV 融合 — this is a pure serial Cube recurrence, not a Cube+Vector FA-style
  scatter; no cross-scope ping-pong opportunity beyond multibuffer already on.
- [未命中] Point 30 Attention/FA 专用 — not an FA op (no QK^T softmax PV chain).
- [未命中] Point 31 IR 分析 — skipped: target reached (7.11x >= 5.0x), no ordinary lever remains,
  and per-chunk-step cost is uniform (~0.43-1.2 us) across dense+varlen -> recurrence-bound, not
  a pipe imbalance that IR/simulator could name. ir_has_more_suggestions=false.

## GQA head-merge exploration (design-level, not implemented)

- [已评估, 不实施] Fold >1 value-head per program sharing one key head (ratio>1 cases 5/6 only).
  Each value-head has a distinct w/u/v stream, so merging heads only saves the shared-k load
  while adding register pressure and a second serial chain per program. Cases 5/6 already at
  6.7x/6.1x (above the 5x target). Geomean-weighted, not worth the complexity/numerical risk.

## Outcome

- best_speedup: 7.1132 (geomean), verify 8/8, target 5x reached.
- improvement_made: false (the single experiment regressed; revert to anchor).
- final code = Phase 3 baseline sha 0d543b7e (chunk_gated_delta_fwd_h_generated.py, num_warps=4).
