# 5_LightningIndexer 迭代日志

目标: 精度 54/54 全过前提下 speedup_vs_torch 最大化（插件 fusion-search，target_speedup=5 vs baseline）。
环境: Ascend 910B3（Ascend910_9372，20 AIC）/ triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。
本 run 由 `triton-optimizer-multi-agent` 插件驱动：Round 1 = fusion-search 融合发现（B1/B2/C0/D0，内部并行 3 route）；
Round 2 = 遗传进化（G0-G7，内部并行 3 route）。每轮子 CLI 独立执行 Phase 0-8，主 orchestrator 只做调度与 C1/C2 判定。

## 时间线总览

```
R1（融合发现）      R2（遗传进化）
0.3178 ──────────→ 3.9558 ★ 破 5x baseline（12.02x）
（route-2 冠军）    （route-2 排序压缩）
```

数据来源：R1/R2 的 `perf_result.json`（54 case 几何平均）、`gene_pool.json` fitness、`round_result.json`、`report.md`。

---

## Round 1 — 融合发现（一次性 B1/B2 + 并行 3 route 初始基因池）

### 基线（warm-start opt2x，Phase 3 初版）

**speedup_vs_torch = 0.3178**（route-2，impl 1.348ms / fw 0.4285ms，54/54）。
基线是纯 Triton bitonic-topk（1.3485ms），topk 是已知墙（[[triton-5-lightningindexer-topk-wall]]：
torch.topk ~17µs 是唯一快路径但 validator 禁用；tl.sort 编译崩 / gather-bitonic ~1ms 地板）。

### Phase 1/2 — B1 原子分解 + B2 自动融合分析

选融合方案 **F1**：fused Triton score 链（dot+relu+head-sum+mask）+ CANN `npu_sort_v2` K 阈值 oracle
+ Triton threshold-select 尾处理。B2 判定 score 链可整段融合、top-K 必须借 NPU sort。

### Phase 2.5/C0 — 初始基因池（8 基因 G001-G008）

| gene | 类别 | 说明 |
|---|---|---|
| G001 | structural | sort-threshold-select 三段结构（npu_sort_v2 tau oracle） |
| G002 | compute | `tl.dot` bf16/fp32-acc + relu + w head-sum 融合 |
| G003 | tiling | BLOCK_J=256（1024 编译失败后的上限） |
| G004 | tiling | 宽块 BLOCK_J=1024（预期不可行，留作对照） |
| G005 | memory | tl.cumsum 单遍前缀排名 epilogue |
| G006 | memory | two-pass select（概念，未实施） |
| G007 | vectorization | packed (score, int32(bf16)) 一次算两用 |
| G008 | pipeline | num_stages=2 双缓冲 |

### Phase 3 — dispatch_parallel 并行 3 route

- **route-1**：F1 x G001+G002+G003+G005（cumsum 前缀排名 epilogue）
- **route-2**：F1 x G001+G002+G004+G008（宽块 + 流水）
- **route-3**：F1 x G001+G002+G003+G005（同上探索）

3 route 全部 **3600s timeout** 被 dispatch_parallel 杀掉（route-1 中途死于 **API 429 额度上限**，
第一次 re-dispatch 后 3 route 各跑满 3600s）。但 route-2 在 iter_1 留了完整 perf+verify：
**speedup_vs_torch = 0.3178，54/54**。

### Round 1 关键修复（进基因池 / 经验）

1. `npu_sort_v2` 必须喂 **2D `[rows, S2]` view**（1D 布局不可靠）——G001 关键约束。
2. BLOCK_J=1024 **编译失败**（UB hivm-plan-memory）→ 256（G003/G004 实测差异）。
3. sparse_mode=3 mask 语义 `j <= i + (as2-as1)` 验证 54/54。
4. bf16 score value 输出走 **RNE 位模拟** `(vb+32767+lsb)&-65536`。

### Round 1 瓶颈画像（决定 Round 2 方向）

- score 核 44µs（不是瓶颈）；
- **select epilogue 是墙**：route-1 串行 chunk-scan 在 S2=32768 时 **10.1ms**，
  `aiv_scalar_ratio=0.999` 纯标量 —— profiling 驱动发现，催生 G009。

### Round 1 判定

`round_result.json`：best_speedup=1.0004（vs baseline，仅追平）／0.3178（vs torch），
target_reached=false，54/54。C1/C2 判定 → **继续 Round 2**（<10 rounds，<5x）。

---

## Round 2 — 遗传进化（G0-G7，3 route 并行）

### G0 — 加载基因池（9 基因：继承 8 + 新增 G009）

**G009 = profiling 驱动新基因**：`PARALLEL CHUNK-SELECT EPILOGUE`，把 K3 select 从 grid=(rows,)
串行 NCHUNK 循环改成 grid=(rows, nchunks)，攻击 round-1 的 aiv_scalar=0.999 串行扫描。

### G1-G3 — 基因选择 / 交叉 / 变异

- **route-1**（elite）：G001+G002+G004+G008 —— round-1 冠军组合复验，锚定 speedup。
- **route-2**（crossover）：G001+G002+**G003**（交叉 tiling）+G008+**G009**（并行 select 新基因）。
- **route-3**（exploration）：G001+G002+G004+G008+**G009** —— 隔离验证 G009 效果。

### G4 — profiling 驱动基因新增（G009 进化为排序压缩）

route-2 内部迭代把 G009 从"并行 chunk-select"**进化为 sort-compaction select**：

```
count kernel（每 chunk 并行 gt/eq 计数）
→ build_masked kernel（score>tau 写 rank 位 + LARGE2 填充）
→ 第二次 npu_sort_v2（升序压缩，前 K 即升序 top-K）
→ finalize（清无效行 + value 位提取）
```

消除串行 `tl.cumsum` carry 链，impl 1.348ms → **0.1122ms**。

### G5 — 并行派发 + G6 — 收集结果

| route | 基因 | impl_ms | speedup_vs_torch | verify | 状态 |
|---|---|---|---|---|---|
| route-1 | elite | 1.3529 | 0.3168 | 54/54 | success |
| route-2 | crossover+sort-compaction | **0.1122** | **3.9558** | 54/54 | **success（冠军）** |
| route-3 | exploration（scatter select） | 4.229 | n/a | 54/54 | partial（**4.2ms 抛弃**） |

route-2/route-3 又撞 **3600s dispatch timeout**；final `round_result.json` 从 route-2 完整的
round_1 perf+verify 产物恢复（route-2 的 3.9558 落在 `opt-round-2/output/iter_1/`）。

### Round 2 关键结果

- `round_result.json`：best_speedup=**12.02**（vs baseline）／3.9558（vs torch），**target_reached=true**，54/54。
- route-3 证伪：并行 scatter + 两次 int32 `tl.cumsum` 定位（BLOCK_SEL=2048）仍标量（aiv_scalar=0.997），
  4.2ms —— **同样绕串行，排序压缩 vs 并行 cumsum 差 37 倍**。
- 剩余墙：大 S2（16384/32768）tail 0.3–1.8ms（第二次 npu_sort_v2 主导）；build_masked 占 impl 39%。

### Round 2 判定

target_reached=true → 进入停止门禁 → `check_stop_decision.py` can_stop → **停止，选择 opt-round-2 为全局最优**。
复制 `5_LightningIndexer_final.py`，出最终报告。

---

## 已排除方向（合并全表，全部有实测证据）

| 方向 | 结论 | 证据 |
|---|---|---|
| 串行 chunk-select epilogue（tl.cumsum 前缀） | ⛔ | route-1：10.1ms（S2=32768），aiv_scalar=0.999 |
| 并行 chunk-scatter select（两次 int32 cumsum） | ⛔ | route-3：4.2ms，int32 cumsum 仍标量 |
| BLOCK_J=1024 宽块 | ⛔ | G004：UB hivm-plan-memory 编译失败 |
| `torch.cumsum` host 前缀表 | ⛔ | validator AST 预检禁用 |
| `torch.topk` | ⛔ | validator 禁用（唯一已知快路径但不可用） |
| 纯 Triton bitonic-topk | ⛔ | ~1.35ms 地板（warm-start 基线即此） |

## 环境与口径备注

- 每轮子 CLI 由 `dispatch_round.py` 拉起的独立 `claude -p` 执行；内部 route 由 `dispatch_parallel.py` 并行派发。
- **API 429（5h 额度上限）曾中断 route**：route-1 首跑死于 429，re-dispatch 后恢复；与 0812/53 同款问题。
- **调度 infra 事故**：orchestrator 早期 9 连败是 `.claude/template` copytree 冲突（`FileExistsError`），
  由 supervisor 清残留目录解决；非算子问题。
- benchmark 口径：`speedup_vs_torch` 逐 case 几何平均；warmup=5/repeats=50；54/54 硬闸门。
- 冻结锚 `44291baa…9529` = 0811/5 镜像锚，torch 参考从未被改动。
