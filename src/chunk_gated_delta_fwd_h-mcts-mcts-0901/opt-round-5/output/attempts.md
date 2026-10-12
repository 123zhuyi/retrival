# Attempts Log — Round 5

Round 5 (MCTS root-expansion re-anchor). Anchor = frozen 6402eb golden-equivalent fused chunk-recurrence kernel; Phase 3 iter_0 verify 8/8, benchmark geomean 7.2335x (target 5x reached). Phase 4 ran one real optimization iteration (opt_iter_0: multibuffer=False probe) and reverted on regression. Below is the ordered Phase-4 optimization-point check log against the anchor kernel.

## Optimization Point Sweep (Phase 4)

- [ ] #1 入参静态化优化: 未命中 — kernel already declares shape/block constexprs (HV,HK,HR,K,V,BT,BV,NVT,IS_VARLEN,USE_H0,STORE_HT); T/NT are runtime via do_not_specialize (required for varlen seq dispatch). No un-staticized fixed param remains.
- [ ] #2 Tiling 优化: 未命中 — block tiling already optimal (BV=min(64,V), BT=chunk_size, S[K=128,BV=64] register-resident). Larger BT is forbidden by task chunk_size; BV cannot exceed V=128.
- [ ] #3 分核优化: 已尝试 — weak shapes (cases 3/4: B=1,HV=48,T=8192) emit only 96 programs; serial T/chunk recurrence cannot be split without changing math. Grid split NVT=2 already doubles parallelism (V-tile). No safe additional split exists.
- [ ] #4 离散访存优化: 未命中 — all loads/stores are dense affine (row-contiguous) within a head; only the seq-prefix chunk base (boh) is computed once per program (scalar), not per-element gather.
- [ ] #5 Scalar 转 Vector 优化: 未命中 — per-chunk scalars (g_last, e_last, d broadcast) are inherent to the recurrence update; g_vec is a vector load; no scalar loop over elements.
- [ ] #6 避免向量 API 标量降级: 未命中 — tl.dot routes to Cube (fp32-ieee), elementwise ops are vector-wide; no cumsum/cumprod/reduce chain that the compiler would scalarize.
- [ ] #7 Pass 消除合并优化: 未命中 — single fused pass already writes v_new and S in one chunk loop; no second traversal over the same data to merge.
- [ ] #8 维度合并优化: 未命中 — (T,K)/(T,V) are 2D tiles, not 3+ nested contiguous loops; head dims are hoisted into the grid, not looped.
- [ ] #9 Libdevice 函数使用: 未命中 — only tl.exp / tl.trans are used (native); no hand-rolled math functions.
- [ ] #10 循环不变量外提: 未命中 — g_last/e_last reload per chunk is O(1) scalar and depends on chunk index; S/K/V block pointers advance by constant strides (already hoisted as affine). No inner-loop reload of an invariant vector.
- [ ] #11 Load 指令重排序: 未命中 — the 5 chunk operand loads (w,u,k,g,vn target) are independent and issued together at loop top; S update depends only on prior S. No load blocked behind a dependent store.
- [ ] #12 Grid 形状与多路径特化: 未命中 — single kernel already covers dense/GQA/varlen via constexpr IS_VARLEN + in-kernel head map; grid = (n_seq|B)*HV*NVT. Host dispatch is minimal and already shape-aware.
- [ ] #13 Autotune 自动调优: 未命中 — launch config manually tuned (nw=4 probed by R4 sibling child); @triton.autotune would re-benchmark per shape at runtime and adds launch overhead to a 2ms total kernel; not used per experience_file guidance.
- [ ] #14 混合策略自动选择: 未命中 — all 8 cases share one fused path; cases differ only in B/HV/T/variancy, handled by constexpr specialization already present.
- [ ] #15 维度合并与大 BLOCK 累加: 未命中 — not a normalization multi-kernel; no separate stats kernel with scalar accumulator.
- [ ] #16 连续拷贝聚合优化: 未命中 — not a pure-copy op; outputs (S_out, vn_out) are produced by compute, not block copies.
- [ ] #17 消除冗余的边界运算: 未命中 — boundary mask (mask_t) is required for tail chunks in varlen/last chunk; no provably redundant tl.where chain.
- [ ] #18 Kernel 分裂优化: 未命中 — total_cases=8 and geomean speedup 7.2335x >> 2.0; rule condition (speedup<2.0) not met; splitting a serial recurrence across kernels would force GM round-trip of S each chunk (strictly worse).
- [ ] #19 Cube/MTE3 分阶段批量解耦优化: 未命中 — no atomic-scatter alternation with Cube accumulation; S is register-resident and stored once per chunk (state) — no MTE3-blocking pattern.
- [ ] #20 Host 侧张量维度拼接优化: 未命中 — no composite `a·c + b·d` multi-dot over separate segments needing host concat.
- [ ] #21 Workspace 物化解耦优化: 未命中 — v_new is already materialized to vn_out (task output) rather than recomputed; no multi-pass gather conflict.
- [ ] #22 Latency-Bound 循环维度 Tile 合并: 未命中 — kernel is NOT latency-bound: geomean 7.23x, cube util bounded by serial chunk count; dot dims (K=128, BV=64) are at microblock-friendly sizes and cannot be enlarged without changing S tile semantics.
- [ ] #23 Device-side Gather 连续化: 未命中 — no random-index gather; seq-prefix offset is a scalar per program.
- [ ] #24 Matmul 链中间 buffer dtype 优化: 未命中 — single-level dot chain (w@S then kᵀ@upd), no串联 matmul with intermediate buffer read by a second dot.
- [ ] #25 输出预初始化: 未命中 — every output element is written exactly once by a program (no default-value zero fill / conditional overwrite).
- [ ] #26 Ascend Interpolate 专用优化: 未命中 — operator is not interpolate/upsample.
- [ ] #27 Ascend Pooling 专用优化: 未命中 — operator is not pooling.
- [ ] #28 Ascend Matmul Transpose 专用优化: 未命中 — not a MatmulBothTrans/MatmulTransA/TransB/BMM/Linear family op; dots are in-kernel recurrence primitives, not the operator's identity.
- [ ] #29 CV 融合优化: 已尝试 — Cube(CU: w@S, kᵀ@upd) 与 Vector(AIV: exp/g clamp/decay) 在同 chunk 循环内交替，属 CV 融合 / Batch 流水线适用面。已实际执行单变量 probe: launch `multibuffer=False`（关闭编译器 double-buffer 预取重叠），nw=4 / unit_flag=True 保持。verify 8/8 通过但 benchmark 6.8053x < anchor 7.2335x（-5.9%）→ 回归 → 回退到 multibuffer=True。negative data point recorded: 本 kernel 的 load-compute 重叠依赖 multibuffer，关闭后 serial-heavy shapes（varlen 256-512 chunk/program）延迟上升。
- [ ] #30 Attention/FA 类算子专用优化: 未命中 — 本算子为 gated delta 串行状态递推，非 FA 的 `QKᵀ→softmax→PV` online-softmax 主链；无 kv_lo 收缩 / BLOCK_N 放大等 FA 专用杠杆可施加。S 递推不可改写（数学固定）。
- [ ] #31 IR 分析优化: 已尝试 — 已按 IR/编译视角核查：fp32-ieee dot 强制 Cube 低精度路径禁用（精度必须）；关键链 S=S*e_last + kᵀ@(vn*d) 无标量降级；serial chunk loop 是数学硬依赖，IR 无可用重组。IR 分析本轮给出结论：无进一步可执行编译级优化点；llvm/triton-compiler 开关（除 multibuffer/unit_flag/nw）已被 R4 覆盖。

## Genuine Phase-3/4 iteration records

- [x] Iter 0 (Phase 3 codegen): independently reconstruct champion fused chunk-recurrence kernel (S[K=128,BV=64] fp32 register-resident, serial in-kernel chunk loop, V-tile grid split NVT=2, in-kernel GQA key-head map, in-kernel varlen seq-prefix chunk offset). Fix: AST Type-3 gate → GQA key-head map moved into kernel (k_head from pid), k passed pre-flattened [B*HK|HK, T, K]. Result: verify 8/8, benchmark geomean 7.2335x (repeats=20). Target 5x reached.
- [x] Opt-Iter 0 (Phase 4): single-variable launch probe `multibuffer=False` (kept nw=4, unit_flag=True, code otherwise byte-identical). verify 8/8 but benchmark 6.8053x < anchor 7.2335x (-5.9%) → regression → REVERTED to multibuffer=True. improvement_made=false; target_reached=true.
