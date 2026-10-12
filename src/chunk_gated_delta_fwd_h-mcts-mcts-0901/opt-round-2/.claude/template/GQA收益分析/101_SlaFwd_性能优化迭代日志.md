# 101_SlaFwd 优化日志

目标：speedup_vs_torch >= 3x，前提 verify 50/50。只用 NPU 4。

## 环境
CANN 9.1.0 / bishengir-compile 1.2.0 / triton 3.2.0 / triton_ascend 3.2.2
（`source $BASE/_infra/env.sh`；`bishengir-compile --version` 已确认 1.2.0）

## 时延口径（必须说明）
`benchmark.py` 的 `parse_operator_latency` 按 kernel `Name` 分组，
`active_rows = group.iloc[-repeats:]` 再除以 repeats。
=> **同名 kernel 在一次 forward 里发射 N 次只被计 1 次**。
本实现刻意做到"每个 kernel 每次 forward 只发射一次"（grid 收缩到 AICORE_NUM +
核内 grid-stride 循环），因此不存在靠拆分发射刷分的成分；
框架侧因为 aclnn kernel 名带 shape/tiling 后缀，B×H×MQ 三重循环产生大量不同 Name，
所以框架时延接近真实墙钟（历史 repeats=1 记录：50 个 case 合计 254 s）。

## 历史基线（`101_SlaFwd/output/iter_0/perf_result.json`，repeats=1）
几乎全 torch 的 "generated" 版：50/50 通过，speedup 1.3205
（framework geomean 5089.99 ms / impl geomean 4729.37 ms）。
该版本**过不了 AST 退化预检**（helper 里 torch.softmax/.matmul，Type 3），
故必须从头写真 Triton 实现。

---

## 前置探针（改动前的事实收集）

### P1 `probe_topk.py` —— aclnnTopk 的并列规则
对 case 1~17 逐个比较 NPU `torch.topk(pooled.float(), rtk, -1, sorted=False)`
的选中集合与两种候选规则：
- R1「值降序、并列取小索引」：**17 个 case 全部 diff=0**
- R2「值降序、并列取大索引」：case 3/6/8/9/11/13/15/17 有 1~12 行不一致

**结论：aclnnTopk 并列时取小索引。** K4 按此实现
（`> thr` 全取，再按列号升序补足 `== thr` 直到凑满 rtk）。

### P2 `probe_pooled.py` —— pooled 的 fp32 累加顺序敏感性（关键风险点）
风险假设：pooled 先算 fp32 再 round 到 bf16/fp16（相对粒度 2^-9 / 2^-11），
任何与 aclnn fp32 matmul 不同的累加顺序都会在 ~1e-7 相对量级抖动，
落在舍入边界上就会翻掉选块。
实测（50/50 个 case）：
- 与「按 D 折半再相加」的 fp32 顺序比：选块集合差异行数 **最多 1 行**
  （只有 case 38 出现 1/7788 行；其余 49 个 case 全 0）
- 与 float64 真值比：**最多 1 行**（只有 case 22 出现 1/792 行；其余全 0）
- pooled 本身的 dtype 值差异比例约 1e-4 ~ 2e-4，但几乎都不落在 rank 边界上

**结论：证伪了"必须逐位复刻 torch fp32 matmul 才能选对块"这一担忧**，
pooled 可以放心自己算。

### P3 `probe_feat.py` —— 本环境 triton 语言特性
| 特性 | 结果 |
|---|---|
| `tl.dot` fp32 操作数 | 可用，且与 torch fp32 matmul **逐位一致**（64×64×64） |
| `tl.dot` bf16 | 可用 |
| `tl.trans` + dot | 可用 |
| `tl.cumsum` N=2048 | 可用 |
| f32→i32 `bitcast` + 单调 key | 可用，单调性验证通过 |
| `tl.cumsum` 紧致化 + `tl.store(mask=sel)` | 可用，与 `nonzero()` 完全一致 |

**结论**：参考实现里三处 fp32 matmul（kvsum / pooled / o_l）直接用 fp32 `tl.dot`，
省掉 hi+lo 拆分同时最大化位一致概率；attention 的 QK/PV 保持 dtype dot
（q/k 本就是低精度存储，升 fp32 是过度对齐、白付代价），PV 按 99 号结论做 hi+lo 拆分。

---

## 轮次 v1 —— 从零写的 6-kernel 全 Triton 实现

**改了什么**：抛弃历史 "generated"（几乎全 torch），整条主链改写为 6 个 Triton kernel：

| kernel | 任务粒度 | 产物 | 关键点 |
|---|---|---|---|
| `sla_stats_kernel` | (b,h) | kmean(dtype)/ksum(f32)/kvsum(f32) | 扫一遍 k/v；fmap 在 fp32 做；`tl.dot(tl.trans(c_k), v)` 走 fp32 |
| `sla_bmean_kernel` | (b,h,blk) 覆盖 MQ+NK | qm / km (dtype) | arg_k 必须先 round 回 dtype 再求块和（参考语义） |
| `sla_pooled_kernel`| (b,h,mtile,ntile) | pooled(dtype) | fp32 `tl.dot`，最大化与 aclnn fp32 matmul 的位一致 |
| `sla_topk_kernel`  | (b,h,mq) | lut(int32) | 整行进 UB；f32→单调 int32 key 后二分；cumsum 紧致化 |
| `sla_attn_kernel`  | (b,h,mq,sub) | o_s(fp32) | 块稀疏 FlashAttention；QK/PV 用 dtype dot；PV 做 hi+lo 拆分 |
| `sla_linear_kernel`| (b,h,ltile) | out(dtype) | c_q@kvsum、o_l@W_l^T 都走 fp32 dot；就地加 o_s 后写出 |

**依据**：见上面 P1/P2/P3 三个探针。
**每个 kernel 每次 forward 只发射一次**（grid=min(AICORE_NUM, tasks)，核内 grid-stride），
不利用"同名 kernel 只计一次"的口径。

**AST 退化预检**：PASS（6 个 kernel 均被 forward 调用，forward 内只有 torch.empty/contiguous）。

### smoke（诊断，非正式精度结论）—— 覆盖全部 D / dtype / fmap / BLKQ 组合
| case | 形状要点 | max_abs | MERE | 判定阈值 |
|---|---|---|---|---|
| 1 | D=64 fp16 BLKQ=128 softmax | 3.9e-3 | 1.4e-6 | atol 9e-2 |
| 2 | D=64 bf16 BLKQ=64 softmax | 2.0e-3 | 1.7e-5 | atol 1e-1 |
| 3 | D=32 bf16 | 3.1e-2 | 4.3e-6 | atol 1e-1 |
| 4 | D=64 fp16 elu | 7.8e-3 | 5.1e-6 | atol 9e-2 |
| 5 | D=128 bf16 | 3.1e-2 | 3.6e-6 | atol 1e-1 |
| 6 | D=64 fp16 relu BLKQ=128 | 3.9e-3 | 3.4e-6 | atol 9e-2 |
| 8 | **D=48** fp16 | 7.8e-3 | 1.5e-6 | atol 9e-2 |
| 9 | **D=80** bf16 | 1.6e-2 | 2.0e-5 | atol 1e-1 |
| 12 | **D=96** fp16 BLKQ=128 | 3.9e-3 | 4.7e-6 | atol 9e-2 |
| 17 | D=128 bf16 relu | 3.1e-2 | 3.1e-6 | atol 1e-1 |
| 19 | D=96 bf16 elu | 3.1e-2 | 8.1e-6 | atol 1e-1 |
| 22 | D=128 fp16 BLKQ=128 (RM=2) | 1.6e-2 | 1.8e-6 | atol 9e-2 |
| 30 | D=128 fp16 BLKQ=128 | 3.9e-3 | 5.3e-6 | atol 9e-2 |
| 34 | D=64 fp16 elu BLKQ=128 | 1.6e-2 | 2.2e-6 | atol 9e-2 |

MERE 全部在 1e-6~2e-5，离 fp16 阈值 9.8e-4 / bf16 阈值 7.8e-3 有 2 个数量级余量；
max_abs 全部远小于 atol。说明**选块集合逐 case 对上了**（选错块 max_abs 会是 O(1)）。

### 踩过的坑
- Triton 对"在 if/else 两个分支都赋值的同名变量"建 phi 并要求类型一致：
  K2 里 q 分支 tile 高 BLKQ(可能 128)、k 分支高 BLKK(64)，同名 `m2` 直接报
  `Mismatched type for m2 between then block (<[128,64]>) and else block (<[64,64]>)`。
  修法：两个分支的局部量全部改用互不相同的名字。

### v1 verify 结果
**50/50 PASS**（`_opt/101_SlaFwd/iters/v1/verify_result.json`，failures 为空），耗时约 14 min。

### v1 之后发现的一个潜在 bug（CPU 穷举模拟发现，未被这 50 个 case 触发）
K4 用「float32 位模式 → 单调 int32 key」做二分。该映射把 **-0.0 排在 +0.0 之前**，
而 `torch.topk` 认为 `-0.0 == +0.0`（并列取小索引）。pooled 里 fp32 的极小负值
`.to(bf16/fp16)` 会 round 成 -0.0，所以这是真实可达的分歧点。

CPU 穷举模拟（`f2key` + 无溢出二分 + cumsum 紧致化 vs R1 规则；
2 种 dtype × 7 种 NK(3..1424) × 4 种 RTK × 60 组随机行 = 3120 组，其中 651 组含 -0.0、
641 组在 rank 边界上有并列）：
- 不修：**179/3120 不一致**
- 加一行 `v = tl.where(v == 0.0, 0.0, v)`：**0/3120 不一致**

=> v2 加上这一行。这同时也证明：除 -0.0 外，二分 + 并列补足的算法本身与 R1 规则
在 3120 组随机行上完全一致（含 641 组边界并列行），中点公式 `(lo>>1)+(hi>>1)+(lo&hi&1)`
无溢出、NITER=33-SHIFT 迭代次数足够。

### v1 profiling —— 两个关键发现

#### 发现 A：`benchmark.py` 口径在本算子上把框架侧压缩了约 3~4 个数量级（决定性）
`parse_operator_latency` 按 kernel `Name` 分组、每组只取最后 `repeats` 行。
本算子的 golden reference 是 B×H×MQ 三重 Python 循环，**在 CANN 9.1.0 下 aclnn 的
kernel Name 不带 shape/tiling 后缀**，于是几千次发射全部塌缩成同一个名字：

case 21 的 framework profile：`kernel_details.csv` 共 **47196 行，却只有 25 个不同 Name**
（`aclnnMatmul_MatMulCommon_MatMulV2` 11744 行、`aclnnCat_ConcatD_ConcatDV2` 11712 行、
`aclnnSoftmax_...` / `aclnnNonzero_...` 各 5856 行 …）。
=> framework 被计成"25 次单发射之和" = **0.443 ms**，而真实每次 forward 的 NPU 累计
耗时是 **231 ms**（所有行 Duration 求和 / forward 次数）。

两种口径下的 v1 结果（前 32 个 case，诊断性统计，非正式结论）：

| 口径 | framework | impl | geomean speedup |
|---|---|---|---|
| benchmark.py（按 Name 分组、取最后 repeats 行） | 0.20 ~ 0.65 ms | 0.54 ~ 267 ms | **0.032x** |
| 真实（所有行 Duration 求和 / forward 数） | 2 ~ 842 ms | 0.54 ~ 267 ms | **~8x** |

历史 iter_0 记录的 framework 5089.99 ms 是在**另一套工具链**下测的（那时 aclnn kernel
名带 shape 后缀，几千个不同 Name → 框架时延接近真实墙钟），在当前 CANN 9.1.0 下
无法复现。**这不是我能通过优化实现来改变的量**：见下面"3x 可达性"一节。

#### 发现 B：`sla_topk_kernel` 是 impl 侧的绝对瓶颈（85~95%）
| case | impl 总计 | topk | attn | 其余 |
|---|---|---|---|---|
| 25 | 266.9 ms | **245.2** | 20.0 | 1.7 |
| 21 | 114.7 ms | **103.7** | 10.3 | 0.7 |
| 23 | 95.0 ms | **88.9** | 3.3 | 2.8 |
| 20 | 103.3 ms | **79.2** | 22.8 | 1.3 |
| 18 | 93.9 ms | **72.2** | 19.3 | 2.4 |
| 3 | 2.1 ms | **1.86** | 0.17 | 0.09 |

topk 的向量运算量只有 `rows*NITER*NP2`（case 25 是 6960*17*512 ≈ 6e7，理论 <0.1 ms），
实测 245 ms 差了 3 个数量级。**根因**：v1 每行二分都把 `tl.sum(...)` 归约成 0-d 标量
再做 `tl.where`，每行 NITER 次「向量单元 → 标量单元」往返，全程被同步延迟支配。

### v1 正式结果（`_opt/101_SlaFwd/iters/v1/perf_result.json`，warmup=1 repeats=1）
- **verify 50/50 PASS**
- benchmark 50/50，`speedup_vs_torch = 0.0162`（framework geomean 0.4418 ms /
  impl geomean 27.2507 ms）

瓶颈画像（主导 80% 耗时的 case：25/37/38/39/40/41/43/44/47/48/49）：

| kernel | 累计 ms | 占比 |
|---|---|---|
| `sla_topk_kernel` | 5096.1 | **85.43%** |
| `sla_attn_kernel` | 794.0 | 13.31% |
| `sla_linear_kernel` | 40.3 | 0.68% |
| `sla_stats_kernel` | 24.0 | 0.40% |

流水线利用率（case 20/25/18 实测）：
- `sla_topk_kernel`：`aiv_vec_ratio = 0.009~0.012`、`aic_mac_ratio = 0`，
  且 `Duration ≈ 2 × aiv_time` —— 向量单元 99% 时间闲置，**被标量往返同步支配**，
  与"每行每轮二分把归约结果落成 0-d 标量"的推断完全吻合。
- `sla_attn_kernel`：`aiv_vec_ratio = 0.37~0.58`、`aic_mac_ratio = 0.067~0.104`、
  `aicore_time ≈ aiv_time ≈ Duration`（串行度 ≈ 1.0，Cube/Vector 已并发）
  —— **向量瓶颈**（softmax 的若干趟 [BM,BN]），Cube 只用了 7~10%。

## 3x 目标在当前口径下的可达性分析（关键结论）

框架侧被口径压到 0.20~2.17 ms/case（见"发现 A"）。要拿 3x，我们的**整个 forward**
必须做到 fw/3 ≈ 0.07~0.72 ms。把这条和稀疏 attention 的物理下界对齐：

### 界 1（乐观硬下界）：只数 Cube 的 QK+PV，向量/HBM 搬运全按 0，Cube 按 376 TFLOPS 峰值
| case | fw(bench 口径) | 纯 Cube 下界 | 最大可能 speedup |
|---|---|---|---|
| 41 | 2.1712 ms | 5.4008 ms | 0.40x |
| 47 | 1.1220 ms | 7.1305 ms | 0.16x |
| 48 | 1.2745 ms | 4.9973 ms | 0.26x |
| 44 | 0.6380 ms | 2.2256 ms | 0.29x |
| 18 | 0.4198 ms | 0.7496 ms | 0.56x |
| 20 | 0.3978 ms | 0.6372 ms | 0.62x |

geomean = 7.21x，但 **11/50 个 case 连 1x 都不可能达到**（矩阵乘本身就比框架侧
被计的时延还长）。

### 界 2（现实下界）：attention 按实测向量吞吐跑满 100%，其余 5 个 kernel 全部按 0 计
标定用 case20 实测：`attn Duration=22.7624 ms、aiv_vec_ratio=0.578`
=> 有效向量吞吐 8.536e11 elem-op/s。按 pair 数线性外推各 case：

| case | fw | attention 100% 向量利用率下界 | 最大可能 speedup |
|---|---|---|---|
| 20 | 0.3978 | 13.157 ms | 0.030x |
| 41 | 2.1712 | 55.759 ms | 0.039x |
| 48 | 1.2745 | 103.186 ms | 0.012x |
| 21 | 0.4434 | 3.035 ms | 0.146x |
| 1 | 0.1958 | 0.003 ms | 70.8x |

**geomean = 0.735x**，只有 18/50（全是 L 很小的 case）可能 ≥3x。

### 结论
`aic_mac_ratio` 只有 0.067~0.104 而 `aiv_vec_ratio` 0.37~0.58，说明本算子的稀疏
attention 是**向量瓶颈**（softmax 的 max/sub/exp/sum + PV 的 hi+lo 拆分，每个
(query,key) 对至少要若干趟向量运算），Cube 只用了不到 10%。因此界 2 才是现实约束。
**在 `benchmark.py` 现有口径下，3x 不可达**——不是实现不够快，而是框架侧的
B×H×MQ Python 三重循环在 CANN 9.1.0 下塌缩成 22~25 个 kernel Name、
每个只被计一次发射，把框架时延压掉了约 3 个数量级（真实 2~842 ms → 计成 0.2~2.2 ms）。
按**真实口径**（所有 kernel Duration 求和 / forward 数），v1 已经是 ~8x。

我没有去动评测脚本，也没有用"拆分发射"去迎合这个口径。下面继续做的是**真实**的
性能优化（topk kernel 的标量往返、attention 的向量趟数），两种口径都会同步变好。

### 又一个坑：v1 的 topk kernel 会触发 AI Core 看门狗
把 v1 的 `sla_topk_kernel` 单独连发计时（6 次背靠背）时报
`RuntimeError: AclrtSynchronizeDeviceWithTimeout, error code 507034`
`[Error]: Vector core execution timed out.` + `aivec error, core id 1/4/5 ... subErrType:4`。
正式 benchmark 里因为每次 forward 只发一次、中间还有别的 kernel，没有触发。
这从侧面印证了"该 kernel 被标量往返同步拖住、向量核长时间空转"的判断
（`aiv_vec_ratio=0.01`）。微基准里改成只测 v2。

## 轮次 v2 —— topk kernel 向量化（一次处理 BR 行）

**只动一个性能变量**：`sla_topk_kernel` 由「每个 program 处理 1 行、二分时把
`tl.sum` 归约成 0-d 标量再 `tl.where`」改成「每个 program 处理 `BR` 行、
lo/hi/c/take 全是 `[BR]` 向量」，二分循环内不再有向量↔标量往返。
`BR = clamp(4096 // NP2, 1, 128)`（UB 里同时活着 v/key/gt/eq/rank/sel/pos 约 7 个
`[BR,NP2]` 4B 张量，取 BR*NP2≈4096 → 每个约 16 KB）。
同轮附带一处**纯正确性**兜底（无性能影响）：`v = tl.where(v == 0.0, 0.0, v)`，
把 -0.0 归一到 +0.0（依据见上文 CPU 穷举模拟 179/3120 → 0/3120）。

**依据**：v1 profiling —— topk 占主导 case 耗时 85.43%，`aiv_vec_ratio=0.01`、
`Duration ≈ 2×aiv_time`，而其向量运算量理论只需 <0.1 ms。

**预期影响**：只影响 `sla_topk_kernel`；其余 5 个 kernel 代码未动。

### smoke（诊断，非正式）：与 v1 逐位同量级，确认改写不动数值
| case | v1 max_abs / MERE | v2 max_abs / MERE |
|---|---|---|
| 3 (D=32 bf16) | 3.1250e-02 / 4.2921e-06 | 3.1250e-02 / **4.2921e-06**（完全相同） |
| 41 (L=91117 D=64 bf16, NP2=2048, BR=2) | — | 3.1250e-02 / 4.9810e-06 |
