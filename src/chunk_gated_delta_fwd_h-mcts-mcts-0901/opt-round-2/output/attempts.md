# Round 2 optimization attempts

Phase 3 (iter_0) applied the MCTS direction (BV=64 universal fix). Phase 4 (opt_iter_1)
ran the full triton-latency-optimizer scan of optimization points 1-31. This file records
every point evaluated.

## Phase 3 attempt (direction = V-tile BV correction)

- [已尝试, 成功] BV V-tile correction: parent heuristic `BV=64 if BT<=64 else 32` -> uniform `BV=min(64,V)`.
  Offline sweep of the ACTUAL kernel over all 8 shapes (dense+varlen, cs 64/128) proved BV=64
  fastest on EVERY shape; cs=128 cases improved 1.46-1.72x (kernel ms). Geomean 5.70x -> 7.06x.
  verify 8/8, benchmark 8/8.

## Phase 4 full 31-point scan (opt_iter_1)

- [未命中] Point 1 入参静态化 — T/NT already `do_not_specialize`, shapes/HV/K/V/BT/BV constexpr.
- [未命中] Point 2 Tiling — V-tile already split (BV=64); BV sweep shows no better tile config.
- [未命中] Point 3 分核 — grid = B*HV*NVT; max usable parallelism given V-tile grid split.
- [未命中] Point 4 离散访存 — no runtime-random index tl.load; k head-map is host index_select.
- [未命中] Point 5 Scalar→Vector — no scalar reduction/accumulator in inner loop.
- [未命中] Point 6 标量降级 — no int64 CMP/div loop; arange/comparisons vectorized.
- [未命中] Point 7 Pass 消除合并 — single fused pass per chunk already (S recurrence fused).
- [未命中] Point 8 维度合并 — host already pre-flattens to [BH,T,D] contiguous.
- [未命中] Point 9 Libdevice — exp is tl.exp (native); no manual math re-implemented.
- [未命中] Point 10 循环不变量 — h0 loaded once; chunk operands loaded once per iteration.
- [未命中] Point 11 Load 重排序 — multibuffer=True already enables compiler overlap.
- [未命中] Point 12 Grid 多路径 — single kernel covers all 8 shapes (dense+varlen, cs64/128) verify 8/8.
- [已尝试, 未采用] Point 13 Autotune — Triton-Ascend num_warps/num_stages unsupported (validated fixed at 4).
  Manual triton.Config BV autotune runs on dot kernels but selects BV=64 for EVERY shape while adding
  ~7s/shape overhead; hand-tuned BV=min(64,V) reproduces the optimum with zero runtime cost.
- [未命中] Point 14 混合策略 — dtype/head-ratio handled host-side (reshape/index_select); no per-shape kernel split needed.
- [未命中] Point 15 dim-merge large block — not a normalization stat+apply operator.
- [未命中] Point 16 连续拷贝聚合 — not a pure memory-copy operator.
- [未命中] Point 17 冗余边界 — tail-chunk masks + other=0.0 required for varlen/dense partial chunks.
- [未命中] Point 18 Kernel 分裂 — every case speedup_vs_torch >= 4.97 (>= 2.0); no split warranted.
- [未命中] Point 19 Cube/MTE3 解耦 — outputs (S_out/vn_out) stored per-chunk, no atomic-reduce contention.
- [未命中] Point 20 host concat — the two recurrence dots (w@S, k^T@upd) are sequential GEMMs on different operands, not concat-able segments.
- [未命中] Point 21 workspace 解耦 — no multi-pass recompute / conflicting loop order.
- [未命中] Point 22 latency-bound tile merge — strictly serial chunk recurrence is a genuine data dependency (D2); cannot merge chunk iterations.
- [未命中] Point 23 device-side gather — k head-map gather is tiny host index_select before launch, not in-kernel random gather.
- [已尝试, 拒绝] Point 24 chained-matmul dtype — switching recurrence dot accumulation/operands to bf16 breaks the fp32-ieee CANN semantics (operator-memory: must reproduce exact fp32 dot incl. +Inf mask); would fail verify.
- [未命中] Point 25 输出预初始化 — vn/S every element written via masked store; no default-value branch.
- [未命中] Point 26 interpolate 专用 — operator is not interpolate/upsample.
- [未命中] Point 27 pooling 专用 — operator is not MaxPool/AvgPool.
- [未命中] Point 28 matmul-transpose 专用 — not a transpose-matmul operator family.
- [未命中] Point 29 CV 融合 — S register-resident across chunks, no GM round-trip of intermediate state; multibuffer/unit_flag already on (D7).
- [未命中] Point 30 FA 专用 — not attention main chain (QK^T→softmax→PV); gated-delta recurrence.
- [已尝试, 未采纳] Point 31 IR 分析 — inspected generated IR; serial recurrence dots dominate; no regression-free IR transform available.

## Evaluated, not applied (context)

- [评估] Triton-Ascend num_warps/num_stages — validated unsupported (fixed at 4).
- [评估] manual BV autotune (triton.autotune Config over BV) — runs on dot kernels but selects BV=64
  for every shape; adds ~7s/shape autotune overhead; no runtime gain -> not applied.

## Notes

- AST precheck flags torch.index_select (Type-3) — accepted pattern for gated-delta-net head-group k map;
  functional verify.py gate is authoritative (8/8).
- Final Phase 4 code == Phase 3 generated body (header comment only documents the scan); benchmark geomean
  7.3312x (8/8), verify 8/8. No regression-free optimization point found beyond the BV fix.
