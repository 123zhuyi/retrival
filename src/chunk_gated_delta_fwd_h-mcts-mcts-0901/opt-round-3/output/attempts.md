# Round 3 optimization attempts

Phase 3 (iter_0) applied the MCTS direction (fused chunk-recurrence GDN fwd-H kernel,
independent reconstruction against the 6402eb golden). Phase 4 (opt_iter_1) ran the
full triton-latency-optimizer scan of optimization points 1-31. This file records
every point evaluated.

## Phase 3 attempt (direction = fused chunk-recurrence GDN fwd-H)

- [已尝试, 成功] Fused chunk-recurrence kernel: S[K=128,BV=64] fp32 register-resident,
  serial in-kernel chunk loop, grid (num_seq*HV, 2), GQA head-group select, in-kernel
  varlen seq-prefix chunk offsets, multibuffer=True/unit_flag=True, fp32-ieee dots.
  verify 8/8, benchmark 8/8, geomean 6.87x (target 5x already reached at iter_0).

## Phase 4 full 31-point scan (opt_iter_1)

- [已命中, 成功] Point 1 入参静态化 — K,V declared `tl.constexpr` (all 8 cases K=V=128),
  so the inner-dim K/V masks fold away on the hot S store / u load / v_new store.
  verify 8/8; geomean 6.8655 -> 6.9508.
- [未命中] Point 2 Tiling — V-tile already split (BV=64); K/V tiles are row-contiguous
  [BLOCK_L,128] blocks; no strided access on a reduction axis.
- [未命中] Point 3 分核 — grid = num_seq*HV x num_vt(2). The T/chunk axis is a strict
  serial recurrence (S depends on previous chunk), so it cannot be split across programs
  without breaking correctness. Current grid is the maximal safe parallelism.
- [未命中] Point 4 离散访存 — inner loop uses affine row offsets; varlen bos/eos are
  per-program scalar loads (<=4 seqs), not an inner-loop gather.
- [未命中] Point 5 Scalar→Vector — scalar g_last per chunk is inherent to the recurrence
  boundary (single exp per chunk); no scalar reduction/accumulator in the hot loop.
- [未命中] Point 6 标量降级 — no i64 CMP/div inside the vectorized chunk loop; the varlen
  prefix loop is scalar over <=4 sequences once per program.
- [未命中] Point 7 Pass 消除合并 — S recurrence is already a single fused pass per chunk;
  no repeated traversal of the same data.
- [未命中] Point 8 维度合并 — inputs kept contiguous [.., T, D]; K/V chunk tiles are
  already dense row blocks (no 3-layer strided nesting to merge).
- [未命中] Point 9 Libdevice — exp uses native tl.exp; no hand-written math.
- [未命中] Point 10 循环不变量 — per-chunk w/k/u/g loaded once each iteration; nothing
  loop-invariant is re-loaded.
- [未命中] Point 11 Load 重排序 — multibuffer=True lets the compiler overlap loads with
  compute; no manual reorder available.
- [未命中] Point 12 Grid 多路径 — single kernel covers all 8 shapes (dense+varlen, cs
  64/128, GQA) verify 8/8; no runtime dispatch needed.
- [未命中] Point 13 Autotune — Triton-Ascend num_warps/num_stages autotune unsupported
  (R2 validated fixed at 4); hand-tuned BV=64/BLOCK_K=128 is the empirical optimum.
- [未命中] Point 14 混合策略 — no shape/dtype-dependent kernel switch warranted; every
  case uses the same optimal tile config.
- [未命中] Point 15 dim-merge large block — not a normalization stat+apply operator.
- [未命中] Point 16 连续拷贝聚合 — not a pure memory-copy operator.
- [未命中] Point 17 冗余边界 — tail-chunk masks + other=0.0 are required for varlen /
  partial last chunks; K/V masks now compile away after Point 1.
- [未命中] Point 18 Kernel 分裂 — every case speedup_vs_torch >= 4.71 (gate < 2.0 not met).
- [未命中] Point 19 Cube/MTE3 解耦 — no atomic scatter; single output writes per chunk.
- [未命中] Point 20 Host concat — no composite dot `a·c+b·d` split across segments.
- [未命中] Point 21 Workspace 解耦 — no multi-pass gather conflict; state is register-resident.
- [未命中] Point 22 Latency-bound tile merge — per-chunk dot M=BLOCK_L(64/128) already a
  full cube micro-tile; serial chunk iterations are the irreducible recurrence.
- [未命中] Point 23 Device gather 连续化 — no device-side random gather (GQA handled by
  affine head id, host index_select not needed).
- [未命中] Point 24 Chained matmul dtype — both dots are fp32-ieee accumulation into the
  fp32 register state (required for precision); no fp16/bfloat16 downgrade safe.
- [未命中] Point 25 输出预初始化 — outputs fully written by the kernel (torch.empty);
  no default-value if/where fill.
- [未命中] Point 26 Interpolate — not an interpolate op.
- [未命中] Point 27 Pooling — not a pooling op.
- [未命中] Point 28 Matmul transpose — dots use tl.trans(k2) on the loaded tile, not a
  host transpose; layout already optimal.
- [未命中] Point 29 CV 融合 — this is a pure serial Cube recurrence, not a Cube+Vector
  FA-style scatter; no cross-scope ping-pong opportunity beyond multibuffer already on.
- [未命中] Point 30 Attention/FA 专用 — not an FA op (no QK^T softmax PV chain).
- [未命中] Point 31 IR 分析 — skipped: target already reached (6.95x >= 5.0x) and no
  remaining ordinary point yields a real lever; ir_has_more_suggestions=false.

## Outcome

- best_speedup: 6.9508 (geomean), verify 8/8, target 5x reached.
- final code = output/opt_iter_0/optimized_code.py (chunk_gated_delta_fwd_h_generated.py).
