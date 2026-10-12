# 5_LightningIndexer 性能优化方向与收益分析

> 面向"下一个做 LightningIndexer / Sparse 索引型 top-K 算子的人"。每条标注**实测收益**与**适用条件**。
> 实测环境：Ascend 910B3（Ascend910_9372，20 AIC）、triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。
> 本算子由 `triton-optimizer-multi-agent` 插件以 **fusion-search** 算法驱动（Round 1 一次性 B1/B2 融合分析 +
> 并行 route 初始基因池；Round 2 遗传进化 G0-G7），所有数字来自官方 `verify.py` / `benchmark.py`，
> 阈值未改、未用 `--verify_not_required`。
> ⚠️ 数据来源分级：R1/R2 的数字盘上 `triton_ascend_output/…/perf_result.json` 可验证；
> 单条 route 内部迭代的部分数字仅存于 route 子目录 `sub_agent.log` 与 `report.md`，下文逐条标注。

## 0. 算子形态（先算账）

```
score[b,i,j] = Σ_h relu(q[b,i,h,:] · k[b,j,:]) · w[b,i,h]        # 打分：Q(BS×D) @ K^T + relu + w 加权 head-sum
topk_idx[b,i,:] = top-K 个 j（按 score 升序，sparse_mode=3 时 j ≤ i + (as2-as1)）
out_idx[b,i,1,K] = 升序 top-K 索引；return_value=True 时另出 out_val = bf16(score 取指定位)
```

- 54 个 case：B ∈ [1,8]，S1 ∈ [64,512]，S2 ∈ [256,32768]（8 档），N1=**固定 64 head**，D=128，
  bf16 46 个 / fp16 8 个；`sparse_count=2048`（K=min(2048, S2)），`sparse_mode=3`。
- **形态判定：这是"score 计算极轻 + top-K 极重"的 Sparse-索引型算子**，与 attention 家族（FA/GQA 的
  "KV 长循环"形态）完全不同——**attention 那套 KV 握手经验在这里基本不适用**。
- **三个决定方向的事实**：
  1. **score 核不是瓶颈**：fused 后仅 38µs（占 impl 22%）。Q 只有 64×128、K 也小，`tl.dot` cube 路径
     轻松喂饱，瓶颈从来不在打分。
  2. **top-K selection 是绝对主墙**：torch 参考里 `aclnnSort_SortAiCpu_Sort` 占 1466µs（CPU sort），
     impl 侧一开始也卡在 select 尾处理。**K=2048 是常数**，S2 大时等于"从 32768 里选 2048 个再排序"。
  3. **`tl.cumsum` 在 Ascend 上极慢**（与 Sparse-块跳过型 100/101 号同族结论）：select 尾处理的
     串行 `tl.cumsum` 前缀链在 S2=32768 时 ~ms 级、aiv_scalar=0.999——**选 K 位之前的"计数→定位"步骤
     必须绕开 cumsum**（本算子用"排序压缩"绕开，见 §2）。

## 1. 收益排序（本算子实测，两条线合并）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | **top-K 选位改"排序压缩"**：count kernel（每 chunk 的 gt/eq 计数）+ build_masked（score→rank 位、未选中填 LARGE2）+ 二次 `npu_sort_v2` 压缩 + finalize | 0.3178 → **3.9558（约 12.4x，route-2 内部迭代）** | select 尾处理出现 `tl.cumsum` 前缀链且 aiv_scalar≈0.99 时 |
| 2 | **`npu_sort_v2` 作 K 阈值 oracle（G001）**：升序排序取 `sorted[S2-K]` 当 tau，省掉全量 top-k | 0.1603 → 0.3178（R1 route-2） | 需要"第 K 大的值"而非完整有序输出时 |
| 3 | **score 链融合成单 kernel（G002）**：`tl.dot` bf16/fp32-acc + relu + w head-sum + mask 全在核内 | 0.2262 → 0.3178（R1）| 参考实现把 score 拆成多 kernel / 逐 token 标量点积时 |
| 4 | **BLOCK_J=256 tiling（G003）** | 0.3178 → 2.8644（fitness） | BLOCK_J=1024 编译失败（UB hivm-plan-memory）时——**上限先探再定** |
| 5 | **packed (score, bf16-bitcast) 一次算两用（G007）**：K1 同时出 score 值 + int32 位图，K3 复用不重算 | 并入 3.9558 | `return_value=True` 需同时出索引与值 |
| 6 | **并行 count kernel 替代 host 前缀表（G009 演进）** | 并入 3.9558 | validator 禁 `torch.cumsum`，host 侧前缀表不可用时 |
| 7 | **num_stages=2 双缓冲（G008）** | 并入 2.8644 | score 核 K-loop（S2 分块）MTE2 预取与 cube 重叠时 |
| 8 | **G004 宽块 BLOCK_J=1024** | **0.3171（负资产）** | 只在 elite 路线保留；**证明"块越大越好"在此算子不成立** |

## 2. 有增益方向详解

### 2.1 ★ 头号方向：top-K 选位用"排序压缩"绕开 `tl.cumsum`（route-2 内部迭代）

**起点（R1 的 select epilogue）**：grid=(rows,) 串行扫 NCHUNK 个 chunk，每个 chunk 内 `tl.cumsum`
算 gt/eq 前缀定位。S2=32768 / BLOCK_SEL=2048 → 16 chunk 串行，`aiv_scalar_ratio=0.999`
（纯标量），实测 **~120ms**（route-1）~ **4.2ms**（route-3 scatter 版）——**不可用**。

**终点（route-2 演化出的 sort-compaction）**，把"选 K 位"变成 4 步，全部走 cube / 库级 sort：

```
K1 score 核        → score_buf [rows, S2]                      38µs
K2 npu_sort_v2     → sorted_vals（升序，取 tau=sorted[S2-K]）  22µs
K3a count kernel   → gt_cnt/eq_cnt [rows, NCHUNK]（grid=(rows,nchunks) 并行）  5µs
K3b build_masked   → masked[rows,S2]：score>tau 写 rank 位，score==tau 按 chunk 前缀补位，
                     未选中写 LARGE2=2^21                             68µs（次墙）
K3c npu_sort_v2    → sorted_masked（升序 → 前 K 个即升序 top-K 索引） 16µs
K4  finalize       → out_idx/out_val（清无效行、value 位提取）          4µs
```

**为什么能绕过 cumsum**：把"选中的 j 的 rank 位" + "未选中填超大数"写成 masked 数组，**第二次
`npu_sort_v2` 升序压缩后，前 K 个天然就是升序 top-K 索引**——排序同时完成"选 K + 排序 + 去重"，
不再需要逐元素前缀和。`aiv_scalar` 从 0.999 降到正常，select 从性能墙变成快路径。

> **可迁移判据**：forward 里 select/紧凑化出现"串行 NCHUNK 循环 + `tl.cumsum` 前缀定位"且
> aiv_scalar≈0.99 时，优先想"用一次排序把选位 + 排序合并"。**排序压缩 vs 并行 cumsum 对比**：
> route-2（排序压缩）0.112ms vs route-3（并行 chunk-scatter + 两次 int32 cumsum）4.2ms——
> **同样绕过串行的两个方案相差 37 倍**，区别在于 route-3 的 cumsum 仍是标量（int32 走 vector）。

### 2.2 `npu_sort_v2` 双角色复用（G001 阈值 oracle + G009 压缩）

同一个 `torch_npu.npu_sort_v2` 在最终实现里用**两次**：
- **K2 阈值 oracle**：升序排序 → `tau = sorted[S2-K]`（第 K 大的值），供 count/build 判断 gt/eq。
- **K3c 压缩排序**：对 masked 数组升序，把"选中的 rank 位"紧凑到前 K 位。

⚠️ **必须喂 2D `[rows, S2]` view**（R1 关键修复）：`score_buf.view(rows, S2)`，**1D 布局不可靠**
（npu_sort_v2 对 1D 输入的行为不确定）。这是 R1 阶段踩过坑的教训。

### 2.3 score 链融合（G002）——小但必要的地基

参考/基线把 score 拆成 dot → relu → w 加权 → mask 多步（甚至逐 token 标量点积），
fusion-search 的 B2 分析发现这些能融成单 kernel：

```python
qk = tl.dot(q, tl.trans(kt))            # [N1, BLOCK_J] fp32 acc（bf16 输入）
s  = tl.sum(tl.maximum(qk, 0.0) * w[:, None], axis=0)   # relu + w head-sum 融合
s  = tl.where(valid, s, float('-inf'))  # OOB 列 -inf
```

`BLOCK_J=256`（G003）：**1024 编译失败**（UB hivm-plan-memory），256 是探测后的上限。
`num_stages=2`（G008）让 S2 分块的 MTE2 预取与 cube 重叠。

## 3. 先算账：为什么墙在 top-K 而不在 score

探针实测：score 核（Q 64×128 @ K 128 @ BLOCK_J 256）= **38µs**，cube 占用率低（小 M）。
torch 参考的 `aclnnSort_SortAiCpu_Sort` 单个就 **1466µs** —— **CPU sort 比整个 impl 慢 13 倍**。

即使 score 核优化到极限，top-K 的"从 S2 选 2048 再升序"成本不变。因此：

- **天花板估算 = sort 成本 + select 成本**，不是 score 成本。
- R1 推算"select epilogue 串行扫描"天花板 ~1ms → 被 G009 排序压缩**证伪**：把 select 从扫描改排序后
  select 侧几乎免费，只剩两次 sort（38µs）才是真墙。**教训：天花板估算必须把"可消除的算法结构"
  先归零再算**——这里可消除的是"逐元素前缀和"，不是排序本身。

## 4. 已证伪方向（比 §2 更值钱）

| 方向 | 结果 | 证据 |
|---|---|---|
| **串行 chunk-select epilogue**（grid=(rows,) 扫 NCHUNK + tl.cumsum） | **~120ms**（route-1 10.1ms 为整 select；S2=32768 单 chunk 扫描） | R1：`aiv_scalar_ratio=0.999`，纯标量链 |
| **并行 chunk-scatter select**（grid=(rows,nchunks) + 两次 int32 tl.cumsum 定位） | **4.2ms，抛弃** | route-3：int32 cumsum 仍走标量（aiv_scalar=0.997），跨 chunk 前缀仍需串行 |
| **BLOCK_J=1024 宽块** | **编译失败**（UB hivm-plan-memory） | G004：1024 不可行，256 是上限 |
| **host 前缀表 `torch.cumsum`** | **validator 禁用** | AST 预检禁 `torch.cumsum`（与 Sparse-块跳过型 100/101 号同条），必须绕开 |
| **`torch.topk`** | **validator 禁用**，且本项目已知唯一快路径（~17µs） | [[triton-5-lightningindexer-topk-wall]]：pure-Triton topk 不可行（tl.sort 编译崩 / radix SSA bug / gather-bitonic ~1ms 地板） |
| **纯 Triton bitonic-topk baseline** | **~1.35ms 地板**（warm-start opt2x 基线） | 0811/5 审计：topk kernel ~3814µs 是墙 |

## 5. 与家族经验冲突对照

| 条目 | 家族结论 | 本算子实测 | 冲突原因 |
|---|---|---|---|
| attention 主成本 | 80/81/99：KV 循环 CV 握手主导（4–6µs/页） | **不适用**：score 核仅 38µs，主墙是 top-K | 形态差异：Sparse-索引型，无 attention 输出 |
| `tl.cumsum` 慢 | Sparse-块跳过型 100/101：前缀和写成矩阵乘（56x） | **一致且更彻底**：排序压缩直接绕过 cumsum | 同族：`tl.cumsum` 在 Ascend 上标量化 |
| sort 选择 | 93 号：`aclnnSort` 落 AiCpu 极贵 | **用 `npu_sort_v2` 走 NPU 侧（SortV22 22µs）**，与 torch 参考的 AiCpu sort 形成对照 | 参考实现用 aclnnSort（AiCpu），实现侧用 npu_sort_v2（NPU） |
| 块越大越好 | 99/100：加大 BM/BN 摊薄访存 | **BLOCK_J=1024 编译失败** | UB 预算不同；宽块在此算子 UB 顶不住 |
| 单 token program | 99：逐 token 是数量级级坏味道 | 本算子天然 64 head×S1 行，已按行融合 | 兼容：Q 行粒度就是自然的 program 粒度 |

---

## ★ 测量口径（报数前必读）

- `speedup_vs_torch` = 54 case **逐 case 几何平均**；`passed_cases == total_cases == 54` 是硬闸门。
- **`framework.avg_latency_ms` 是 torch 参考（含 AiCpu sort 1466µs）**；`implementation.avg_latency_ms`
  是 Triton 实现（0.112ms）。两者是同一份 torch 参考的对比，非替换。
- impl 侧 `SortV21/SortV22`（npu_sort_v2 的 NPU 段）与 framework 侧的 `aclnnSort_SortAiCpu_Sort`
  **是两个不同的算子**——**不要拿 framework 的 1466µs 当实现侧的墙**。
- 历史审计最优 0.1603x（0811/5 镜像，54/54）为**另一份实现**；本 run 从 warm-start opt2x
  baseline（1.3485ms）起步，`speedup_vs_baseline=12.02x` 是相对该基线的口径。
