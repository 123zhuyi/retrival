# Matmul operator optimization experience

## Experience: 4_GroupedMatmulSwigluQuant

- **Updated**: 2026-08-22 07:31 UTC
- **Best round**: 5
- **Best speedup**: 0.8011
- **Target reached**: None

### Operator experience

# Operator Experience: 4_GroupedMatmulSwigluQuant (fusion-search/0821-4)

## Failures / Avoid
<!-- Sub-agent appends here after each round -->

## Successful Patterns
<!-- Sub-agent appends here after each round -->

## Notes

## Round 1 (fusion-search 0821-4, generation 1) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **route-1 (G001 BLOCK_M=64/BLOCK_K=512 + G007 staged_sw16) 是 gen-1 best 0.4466x**（dev6 同设备 msprof, 60/60 verify PASS）。BLOCK_M=64, BLOCK_K=512, HALF_BLOCK_N=128, num_stages=2, num_warps=16。
- **所有 fused 单 kernel 均 <1.0x**：torch_npu 算子链（npu_grouped_matmul + npu_swiglu + npu_dynamic_quant）本身已高度优化，CANN 融合算子快于 Triton 手写 kernel。
- route-2 (G001+G004+G006, BLOCK_M=32) 0.2003x：BLOCK_M=32 太小，cube 填充差。
- route-3 (G002+G007, BLOCK_M=128/BLOCK_K=128) 0.0722x：BLOCK_M=128 过大，UB 溢出/低并行。
- route-4 (G001+G003 N-split) 0.2658x：N-split 让每个 split 重读完整 X tile，X HBM 流量翻倍。
- route-6 (G002+G004) 0.3455x：framework/host D2H sync 开销巨大（task 21930us vs framework 7577us）。
- **重要差异**：0819 的 0.77x 是对 Triton baseline（BLOCK_M=32）测的，NOT torch_npu reference —— 与本次 0.4466x 不可直接比较。

### 瓶颈（gen-1 观察到）
1. **HBM/MTE2 data-load bound 绝对主瓶颈**（aic_mte2 0.81-0.90, aic_mac 0.19-0.29）。
2. **两遍 sw16 HBM staging 往返**（pass-1 写 [M,H] fp16 + pass-2 读回）是 #1 可消除流量。
3. **small-M expert（E=2, m_e~256）cube 欠利用**（grid=4, cube_util 4.6-8.6）。
4. **torch_npu reference chain 在 60/60 case 上都快于 fused Triton kernel**。

### 约束与坑
- **HALF_BLOCK_N=256 在 BLOCK_M=64 下超出 192KB UB**（fp32 dequant + int32 acc 工作集）→ MLIRCompilationError / 'requires 2365440 bits while 1572864 available'。必须 HALF_BLOCK_N=128。
- **EPI_BLOCK_N=256 同样 UB 溢出**，需 EPI_BLOCK_N=128。
- **num_warps 16->8 中性偏差**（0.4466->0.4423）；num_stages 1->2 小增益（-1.16%）。
- **msprof 设备竞争**：并发 route 必须钉在同一物理设备（dev3/6）串行采集，跨 route 绝对数字只比同设备。

### gen-2 探索方向
- sw16 HBM 往返消除（G005 recompute 在 gen-1 未验证；G012 in-UB sw16 先验不可行）。
- split-K / 把 epilogue 融合进最后 split-K program（先验 G014/G026 路径）。
- adaptive BLOCK_M（G008 route-5 超时未验证）需在 gen-2 用 bounded sampling 完成。

## Round 2 (fusion-search 0821-4, generation 2) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **route-4 (G001+G006+G010 flattened grouped single-launch grid) 是 gen-2 best 0.5435x**（308.8us, -37% vs gen-1 490.2us, 60/60 verify）。BLOCK_M=64, BLOCK_K=512, HALF_BLOCK_N=128, num_stages=3, num_warps=16。
- **G010 = 单 launch 拍平 total-M grid + pointer-gather expert lookup**，替换 per-expert launch loop。isolated delta +13.3% speedup / -35% task_duration。launch_count=1 确认。
- num_stages 扫描: ns2=0.5009 → ns3=0.5336 (+6.5%) → ns4=0.5301 (-0.65%, UB 压力)。**flattened kernel 上 ns3 最优**。
- 负结果（已验证）: G009 split-K epilogue 0.2232x；G005 recompute 0.2647x；G008 thr=256 3.3x 回退 (仅 thr=32 净正 +0.9%)；BLOCK_K=256 0.1656x。
- route-2 的 1.0087x 是对**自己的 Triton baseline**（回退 op_summary 指标，不可比）；vs torch 0.6132 因共享机竞争自标不可靠。

### 瓶颈（gen-2 确认）
1. **HBM/MTE2 data-load bound 仍是绝对主瓶颈**（aic_mte2 ~0.86）。
2. **两遍 sw16 [M,H] HBM staging 往返仍是 #1 可消除流量**（G009/G005 替代方案均回退 → 两遍 staging 目前最优）。
3. small-M expert（m_e~256）grid 仍欠填充。
4. **torch_npu reference chain 在大多数 shape 上仍快于手写 Triton**（CANN 算子链高度优化）。

### gen-3 探索方向
- sw16 往返的第三种消除: K-outer multi-N-chunk accumulation（route-4 报告提出）或 pass-1 仅算 max_abs（int8 结果直接在 pass-1 用 scale 重算，见 G005 变体）。
- 把 G010 flattened grid 与 G008 adaptive BLOCK_M 组合（route-4 报告提出 adaptive BLOCK_M for small-M core fill）。
- 在 flattened kernel 上重试 split-K 以提升 small-M core 占用（G009 变体）。

## Round 3 (fusion-search 0821-4, generation 3) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **无 gen-3 route 超越 gen-2 champion**。全局最优保持 route-4 (G010+G001+G006, BLOCK_M=64/BLOCK_K=512/HALF_BLOCK_N=128/ns3/nw16) 0.5435x (308.8us)。
- route-1 (elite floor, ns3/nw8) = 0.5355x (317.1us)，champion 噪声内 -1.5%。**精英保留机制有效保证性能不下滑**。

### 关键负结果（gen-3 全部拒绝，避免 gen-4 重试）
1. **G011 K-outer multi-N-chunk accumulation 拒绝**：x 冗余读（16x）不是结构 #1 瓶颈——champion 在 16x reload 下仍达 68% cube util。K-outer 的 acc-tile UB 代价迫使 BLOCK_M 64→32，W HBM 流量翻倍（route-3 391.06us/0.4512x；route-6 unroll=4/BK=256 1110us/0.3544x）。
2. **G012 fused gate+up 单宽 dot 拒绝（decisive regression）**：wide [BK,2*HBN] i8 W tile + non-affine gather + permute/split 无法装入 192KB UB（BM 被迫 16, HBN 64）→ AI-core 2.0x 恶化 (17.61 vs 8.81us) + host-time x15.7 (7320 vs 466us)。**two-narrow-dot 是 Ascend910b3 grouped i8 matmul 的正确形状**。
3. **G008 adaptive BLOCK_M 两档拆分（thr=32）在 flattened grid 上 -22% regression**（406.44us vs 316.74us same-cond）。连续两代验证为负 → 正式 discard。
4. **G001 BLOCK_M=128/HALF_BLOCK_N=64 变异拒绝**：halving W 流量被 doubling A-tile 流量（A 128x512 重读 per hn-iter）抵消，MTE2-bound (aic_mte2 0.893) 下 1515us/0.5154x。**MTE2-bound kernel 上任何增加 A/W HBM 流量的 tile 变异都直接回归**。

### 瓶颈（gen-3 确认）
1. **HBM/MTE2 data-load bound 绝对主导**（aic_mte2 0.86-0.89）；flattened kernel 68% cube util。
2. **两遍 sw16 [M,H] HBM staging 往返仍是 #1 可消除流量**；但 recompute(G005)/split-K(G009)/K-outer(G011)/wide-dot(G012) 四类消除方案在 MTE2-bound 上全部验证回退 → 消除方向需更精密 UB 预算设计，简单结构重构不可行。
3. small-M expert cube 欠填充、torch_npu reference 更快，同前。

### gen-4 探索方向（供参考）
- 针对 sw16 往返做**更精密 UB 预算的局部消除**（如 pass-1 仅算 max_abs、int8 结果 pass-1 内用 scale 重算并跳过 pass-2 的 W 重读），而非全局结构重构。
- num_warps/num_stages 在 champion tile 上已无收益 → 微调已耗尽。
- 若继续，建议探索 W-tile schedule（route-3 提出的 cutting aic_mte2 W HBM traffic）或数据布局变换（如 W 预转置/重排以改善 gather 连续性）。

## Round 4 (fusion-search 0821-4, generation 4) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **无 gen-4 route 超越 gen-2 champion**。全局最优保持 route-4 (G010+G001+G006+G007, BLOCK_M=64/BLOCK_K=512/HALF_BLOCK_N=128/ns3/nw16) 0.5435x (308.8us)。
- route-1 (elite floor) = 0.5166x，champion 噪声内。**精英保留机制有效保证性能不下滑**。
- route-5 (G017 回退 two-narrow-dot) = 0.5206x，同样 champion 噪声内。

### 关键负结果（gen-4 全部拒绝，避免 gen-5 重试）
1. **G013 M-superblock W-reuse（3 变体全拒绝）**：W 确实是 #1 MTE2 HBM 流量（≈2×A），但 int32 superblock acc（BM_TOTAL=128/HBN=128 时 128KB+）**破坏 champion 的 ns=3 double-buffer** 并迫使 HBN=64 → hn-chunk epilogue 翻倍 + A 流量翻倍。W-HBM 削减无法补偿（route-2 canonical 0.2578x / route-3 aggressive 0.1912x / route-4 conditional 0.2426x）。**任何 W-reuse 设计必须保持 HBN=128（硬结构杠杆）。**
2. **G017 wide-dot-via-join 不可行**：`tl.join`+`reshape`+`tl.split` 在 [BK,2*HBN] joined W tile 上 192KB UB 全溢出；连续宽 load+trans+split 变体在 torch_npu Triton 3.2 上 `ConvertLinalgRToBinary` 失败。**two-narrow-dot 是 Ascend910b3 grouped i8 matmul 唯一可行 dot 形状**（与 gen-3 G012 一致）。
3. **elite ns4/nw8 均 aicore execution timeout (507014)**（共享机 cases 23-60）→ **ns3/nw16 是该编码的 stability ceiling**。

### 瓶颈（gen-4 确认）
1. **HBM/MTE2 data-load bound 绝对主导**（aic_mte2 ~0.86）；W re-read ~2×A 是 #1 可减流量，但 UB 预算内所有 W-reuse 设计都打破 ns3 pipeline 或迫使 HBN=64。
2. **sw16 [M,H] 两遍 staging 往返**仍是 #1 可消除流量；recompute(G005)/split-K(G009)/K-outer(G011)/wide-dot(G012/G017) 全部验证回退 → 消除方向需更精密 UB 预算设计。
3. small-M expert cube 欠填充、torch_npu reference 更快，同前。

### gen-5 探索方向（供参考）
- 若继续，唯一可能突破：**数据布局变换**（W 预转置/重排以改善 MTE2 gather 连续性）或**更精密 sw16 局部消除**（pass-1 仅算 max_abs + int8 结果 pass-1 内 scale 重算，跳过 pass-2 W 重读）。纯 tile/结构重构在 192KB UB + ns3 约束下已收敛。

## Round 5 (fusion-search 0821-4, generation 5) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **route-2 (G001+G006+G010+G018 W data-layout transform) 是 gen-5 best 0.8011x**（kernel 202.8us, 60/60 verify, A/B same-window 独立验证）。
- **这是 gen-2 以来第一个结构性突破**：全局最优 0.5435x → **0.8011x (+47.4%)**。
- 关键：**A/B same-window 独立验证揭示 route-2 自报 1.537x 是对受竞争污染的 in-window baseline（544.7us）**；同窗对 champion 实测为 0.5220x，G018 实测 0.8011x。诚实指标下 G018 仍 +53.5% 真实增益。

### 关键正结果（G018, PROMOTED to active）
1. **W [N,K] 预转置（host 侧缓存）+ in-kernel 连续 [HBN,BK] load + tl.trans**：把 W load 从 128B-discrete segments（stride N）变成 BLOCK_K=512B contiguous bursts。**aic_mte2 0.878→0.584**，kernel task 309.8us→202.8us（-35%）。
2. **BLOCK_K=512 对 G018 pattern 是结构性**：bk256（608.6us）regress，因为 contiguous burst 缩到 256B。
3. G018 与既有 champion 编码（G001/G006/G010）正交可叠加，不破坏 HBN=128 / ns3 / two-narrow-dot。
4. G018 后剩余瓶颈转为 **aic_mte1 0.585 / aic_mte2 0.584 共主导**，cube_util ~50%。

### 关键负结果（gen-5 拒绝）
1. **G019 eviction_policy 在 torch_npu Triton 3.2 上不支持** → contiguity-hint fallback 正确性安全但性能中性（0.5316x）。discard。
2. **G020 int8 chunk-local-scale sw16 staging 失败**：精度退化（56/60 max），perf 0.07x，route 超时。discard。两遍 fp16 sw16 staging 仍是 F1 内最优。
3. route-4 (G018+G019 crossover) worker mid-round_3 死亡，无结果（共享机 worker 稳定性风险）。

### gen-6 探索方向（供参考）
- G018 后 aic_mte1（L2/UB）与 aic_mte2 共瓶颈 → 下一方向应瞄准 **MTE1/UB 压力**（如 sw16 staging 的 UB 内局部化、A tile 的 L2 reuse）或 **G018 × sw16 局部消除组合**。
- G018 的 host 侧 W 预转置缓存已验证；可探索 **W 按 expert 打包** 或 **int8→fp16 预转换上移** 以进一步减 MTE2。
- cube_util ~50%，small-M expert 仍欠填充 → 结合 G018 的 flattened grid 上重试 adaptive-BLOCK_M（G008 thr=32）可能不再回归。

## Round 8 (fusion-search 0821-4, generation 8) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **无 gen-8 route 超越 gen-5 champion**。全局最优保持 0.8011x (202.8us A/B verified, sha cef024ef)。
- route-1 (elite floor) = 1.536x contended (354.5us)，champion 字节级复现。**精英保留机制有效保证性能不下滑**。

### 关键负结果（gen-8 全部拒绝，避免 gen-9 重试）
1. **G029 W-stationary M-sub-block 拒绝**：BM_SUB=64 在 192KB UB 溢出（1853440>1572864 bits）；BM_SUB=32 编译通过但 64 行/program = 与 champion 相同 W HBM 流量 + 32 行小 dot 降 cube 效率。4 个 sweep 配置全部回归 0.72x-0.844x。**核心洞见：kernel 是 cube-efficiency-bound，不是纯 MTE2-bound —— 减半 W HBM 不改善端到端。**
2. **G030 A-tile hn-pair reuse 拒绝**：BM=32 使 per-expert M-block 数翻倍 → W 经 MTE1/MTE2 多拉 ~2x（aic_mte1 0.55→0.60，cube under-fill aic_mac 0.415→0.313）。BK=256 破坏 G018 W MTE2 bursts（aic_mte2 0.727）—— **BLOCK_K=512 对 G018 W layout 是结构性**。
3. **G031 HBN=64/ns4 拒绝**：结构性 +39-40% 回归，ns4/3/2 无关（498.5/496.4/495.8us）。更小 dot [64,512]x[512,64] mac 0.435→0.314，N-chunk epilogue 翻倍主导。**HBN=128 是硬结构杠杆**（与 gen-4 G013 结论一致）。
4. **G032 explicit cross-hn W prefetch 编译不可行**：loop-carried 2D W tile → `ConvertLinalgRToBinary` LLVM error + scf.for type mismatch。torch_npu Triton 3.2 不支持手动 cross-loop 2D prefetch buffer。

### 瓶颈（gen-8 确认）
1. **cube-efficiency-bound**：减 W HBM（G029）无效，证明瓶颈在 cube 填充/小 dot 效率，而非单纯 HBM 流量。
2. **W-tile HBM reload per hn-chunk + two-pass sw16 staging roundtrip** 仍是 F1 内残余结构成本；W layout/swizzle/L2-reuse 是唯一未探 MTE2 杠杆。
3. small-M expert cube 欠填充、torch_npu reference 更快，同前。共享机绝对窗口漂移 ~1.7x，same-window A/B 是唯一有效比较。

### gen-9 探索方向（供参考）
- 若继续：**W layout/swizzle / L2-reuse within G018 [N,K] K-contiguous 约束**（唯一未探 MTE2 杠杆）或 **precision-safe single-pass sw16 elimination**（recompute/split-K/int8-staging 全失败，需全新精度安全思路）。
- gen-5 以来连续 4 代（gen-5/6/7/8）0.8011x 平台期 → 建议 Orchestrator C2 停止判定。

## Round 9 (fusion-search 0821-4, generation 9) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **无 gen-9 route 超越 gen-5 champion**。全局最优保持 0.8011x (202.8us A/B verified, sha cef024ef)。
- route-1 (elite floor) = 1.537x contended (354.35us)，champion 字节级复现。**精英保留机制有效保证性能不下滑**。
- route-2/3 最终 kernel 均为 champion 字节级复现（cef024ef）。

### 关键负结果（gen-9 全部拒绝，避免 gen-10 重试）
1. **G033 W chunk-major layout 拒绝**：host 侧把 gate/up 行相邻放置 [num_chunks, 2*HBN, Kp]，kernel 两个窄 load 形成单个 128KB 连续 HBM 区域。Same-window A/B：median +1.3% worse，**aic_mte2 未降（0.556→0.570/0.553）**。**硬件不合并两个窄 tl.load 为单突发** —— 两个 load 仍是独立 MTE2 descriptor，按 per-row 512B K-contiguous burst 抓取（与 G018 相同）。与 gen-8 G029 一致：**W 流量布局类杠杆在 cube-efficiency-bound kernel 上不改善端到端**。
2. **G033 × G021 crossover 中性/负面**：cache_modifier 在 torch_npu Triton 3.2 上禁用 AutoBlockify（kernel warning）→ hints 中性到负面（W-only 0.967x worst, best 0.996x drift-immune）。128KB 连续 granule 未产生 L2 persistence 收益。

### 瓶颈（gen-9 确认）
1. **cube-efficiency-bound 结构性封顶**：aic_mac ~0.42, mte1/mte2 ~0.56 co-limited。W-layout/MTE2 杠杆已耗尽（G018 有效、G025/G029/G033 无效）。
2. **HBN=128 是硬结构杠杆**（256 溢出 192KB UB）；HBN=64 结构性回归。
3. two-pass sw16 staging roundtrip（~6% HBM）残留；single-pass 全部已知变体（G005/G009/G020）已在 gen-2/5 失败。
4. small-M expert cube 欠填充、torch_npu reference 更快，同前。

### gen-10 探索方向（供参考）
- **W-layout 方向已彻底耗尽**（G018 + G025 + G033）。precision-safe single-pass sw16 elimination 的全部已知变体失败，需全新精度安全思路（如 fp32 chunk-max 高精度累积）。
- gen-5 以来连续 5 代（gen-5/6/7/8/9）0.8011x 平台期 → **建议 Orchestrator C2 停止判定**。


## Round 10 (fusion-search 0821-4, generation 10) 关键经验

### 编码与性能（vs torch_npu reference baseline, 目标 2.0x）
- **无 gen-10 route 超越 gen-5 champion**。全局最优保持 0.8011x (202.8us A/B verified, sha cef024ef)。
- route-1 (elite) / route-2 (G034) / route-3 (G035) 最终 kernel 均为 champion 字节级复现（cef024ef）。精英保留机制有效保证性能不下滑。

### 关键负结果（gen-10）
1. **G034 num_warps=32 拒绝（中性）**：full-60 same-window A/B (n=480 each) nw16 203.65us vs nw32 203.50us vs nw16 rerun 203.63us → mean -0.07%, median +0.08%, cube_util 35.65 vs 35.01, **aic_mac 0.423 完全不变**。co-bound kernel 非 warp-overlap limited —— 更多 in-flight warps 不改善 cube/mte overlap。
2. **G035 num_stages=2 拒绝（中性）**：post-G018 champion 上 fresh A/B retest。3 个独立 same-window 窗口 mean ratio 1.0034 / 1.0008 / 1.0003，**aic_mte2 完全一致（0.554-0.560），cube_util ~35% 不变**。"shallower pipeline 减少 UB pressure" 假设不成立 —— kernel 是 cube-efficiency-bound，pipeline depth 不是 limiter。与 gen-2 (pre-G018 ns 非杠杆) 和 gen-7 (G025 ns/nw sweep 非杠杆) 一致。
3. **ns/nw 参数空间正式关闭**：num_stages 与 num_warps 在 gen-10 两个方向上的最后一个未测参数均验证中性 → launch 参数不是该 cube-bound kernel 的杠杆。

### 瓶颈（gen-10 确认）
1. **cube-efficiency-bound 结构性封顶**：aic_mac ~0.42, mte1/mte2 ~0.56 co-limited。W-layout/MTE2 杠杆已彻底耗尽（G018 有效、G025/G029/G033/G035 无效）。
2. HBN=128 是硬结构杠杆；two-pass sw16 staging roundtrip 残留；small-M expert cube 欠填充同前。

### gen-11 探索方向（供参考）
- **ns/nw / W-layout 方向已彻底耗尽**（G018+G025+G029+G033+G034+G035）。唯一残余结构成本是 two-pass sw16 staging roundtrip（~6% HBM），全部已知 single-pass 变体失败，需全新精度安全思路（如 fp32 chunk-max 高精度累积）。
- gen-5 以来连续 6 代（gen-5/6/7/8/9/10）0.8011x 平台期 → **建议 Orchestrator C2 停止判定**。

