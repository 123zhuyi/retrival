# 33_AdaptiveAttention 性能优化方向与收益分析

> 本文是"方向清单 + 每个方向的收益/代价/证据"，与
> `33_AdaptiveAttention_性能优化迭代日志.md`（按轮次的流水账）互补。
> 状态：**进行中**，实测数字随迭代补齐；未测到的写"待测"，不填估算值。

---

## 0. 算子画像：它到底是什么形状的负载

```python
router = nn.Sequential(Linear(D, max(1,D//4)), ReLU, Linear(max(1,D//4), n_opts))  # 两层都有 bias
selected = one_hot(argmax(router(x.mean(dim=1)), -1)).to(x.dtype)
for n_kv in n_kv_heads_options:            # n_opts ∈ {2,3}；50 个 case 里 35 个是 3 路
    完整一路 GQA(q/k/v/out 四个 Linear, bias=False) -> candidates[i]   # [B,S,D]
return (stack(candidates, 1) * selected[:,:,None,None]).sum(1)
```

每一路 GQA 与 `21_GroupedQueryAttention` **逐字相同**，所以 21 号上成立的结论直接可用。

### 0.1 三个决定优化方向的事实

**事实 1：瓶颈在投影 GEMM，不在 attention。**
21 号 v5 的 kernel 分项是 `linear_kernel 128.8us` vs `gqa_attention_kernel 70.7us`。
原因是本算子族 **S 小（3~897）而 D 大（48~1536）**：
投影段 FLOPs = `2·(B·S)·(3D)·D`，attention 段 = `2·B·H·S²·dh·2 = 4·B·S²·D`。
两者之比 = `6D / (4S)`，S < 1.5D 时投影更重 —— 50 个 case 里 **48 个满足**。
⇒ FA 类算子那套"KV 循环握手成本主导"的直觉在这里**不成立**。

**事实 2：one-hot 意味着 (N-1)/N 的计算是纯浪费。**
`selected` 是 one-hot，未被选中的分支乘 0 后对输出贡献恒为 0。
按分支数分布（35 个 3 路 + 15 个 2 路）算几何平均：`3^0.7 × 2^0.3 ≈ 2.66`。
torch 参考侧**没有**这个优化（它老老实实 stack 完再乘 0 求和），所以这是净收益。

**事实 3：router 的 argmax 是"全或无"的精度点。**
argmax 翻转 ⇒ 输出整个换成另一路的结果，不是误差变大。
从 `probe_stage.py` 采到的实际 logits，top1−top2 的间隔在 **0.03 ~ 0.31** 量级
（例：case6 `[-0.112, -0.421]` 间隔 0.31；case7 三路 `[-0.201,-0.094,-0.621]` 间隔 0.107）。
fp16 在这个量级的 ulp 约 1e-4 ⇒ 只要 router 的每一段都在与参考相同的量化格点上落子，
翻转风险很低；但**任何"数学上等价、数值上重排"的 router 优化都要重新对拍 argmax**。

### 0.2 优化链路（含转折点）

```
v1 (继承, 从未跑出结果)
 │  静态排查发现 4 处必然失败项
 ├─ 转折点 A: tl.arange(0, NOPT) 对 NOPT=3 非法 ⇒ 35/50 的 case 根本编不过
 ├─ 转折点 B: 旧编译器 2^18 的 tile 预算作废, 1.2.0 是 2^20
 ▼
v2  verify 0/50, 全部 aicore timeout
 ├─ 转折点 C: 分步探针定位到 out_proj 的 `m // S` + 向量下标 gather
 │            ⇒ 不是精度问题也不是规模问题, 是写法本身让 aicore 挂死
 │            ⇒ 副作用: 把 NPU 3 打成持久损坏, 换到 NPU 11
 ▼
v3  tile 空间改 (b, s_tile, n_tile), 挂死消失
 ├─ 转折点 D: 4/50 —— 但**这一轮是在坏卡上测的, 判无效**
 │            剩余真问题: 15 个 case UB overflow(router 的 W1 tile)
 ▼
v4  one-hot 门控推广到三个 per-branch kernel(未单独验证, 已并入 v5)
 ▼
v5  router 三段并行化 + linear tile 按 dtype 分档 + one-hot 门控   <- 当前
```

---

## 1. 已落地的方向

### 1.1 linear tile 按 dtype 分档 ★ 最大单项

| 项 | 值 |
|---|---|
| 改动 | `BLOCK_M/N/K` 从 `64/128/64` → fp32 `128/256/64`、fp16/bf16 `128/256/128` |
| 增益 | 21 号实测 `linear_kernel` **128.8us → 35.9us（3.6x）**，整体 speedup **1.4518 → 2.2839** |
| 机理 | `linear_kernel` 对每个 M-tile 都要把整份权重 `[K,N]` 重读一遍。M-tile 数 = `ceil(M/BLOCK_M)`。case35（M=1149, N=4416, K=1472, fp32）下 BM=64 ⇒ 18 遍 × 26MB ≈ **468MB** GM 流量 ≈ 468us@1TB/s；BM=128 ⇒ 减半 |
| 代价 | UB 占用上升。fp32 下 BK 只能到 64（BK=128 会 overflow），低精度可以到 128 |
| 适用性 | 本算子的 `qkv_sel_kernel` / `out_sel_kernel` 与 21 的 `linear_kernel` 同体，直接搬 |

### 1.2 one-hot 门控：只算被选中的那一路 ★ 本算子独有

| 项 | 值 |
|---|---|
| 改动 | `qkv_sel_kernel` / `gqa_attention_kernel` / `out_sel_kernel` 三者在进入 K 循环（或 KV 循环）之前 `if tl.load(sel_ptr + b) == BRANCH:` 整块跳过 |
| 前提 | 三个 kernel 的 tile/task 空间里 **b 必须是一个坐标**。attention 本来就是 `(b, h, m_tile)`；两个 GEMM 是为此才从 `M = B*S` 平铺改成 `(b, s_tile, n_tile)` |
| 增益 | 理论 **2.66x**（几何平均）；实测待填 |
| 代价 | 每个 tile 多一次标量 load + 比较（可忽略）；host 侧仍然发射 N 次 kernel，小 case 上发射开销不减 |
| ⚠️ 语义假设 | **跳过 ≠ 乘 0**。仅当未选中分支会产出 NaN/Inf 时两者不等价——参考实现里 `0 * NaN = NaN` 会传播出来，跳过则得到被选中分支的正常值 |
| ✅ 假设已实测验证 | `scratchpad/probe_refnan.py` 在健康卡（NPU 12）上跑完参考实现的全部 50 个 case，**同时统计最终输出与每一路候选值**的 NaN/Inf：`cases with non-finite: 0 -> []`。⇒ 本测试集上**没有任何一路候选值是非有限的**，"跳过"与"乘 0 再求和"**逐位等价**。这条假设不再是风险项，而是已证事实（仅对本测试集成立；若输入分布改变需重扫） |
| 附带 | 未选中 batch 的 `qkv`/`attn` 中间缓冲保持 `torch.empty` 的未初始化内容，但只会被同样被跳过的下游 kernel 读到，不污染输出 |

### 1.3 router 三段并行化

| 项 | 值 |
|---|---|
| 改动 | `grid = B` 的单 kernel → `router_mean_kernel`(沿 (b,d_tile) 并行) + `router_fc1_kernel`(tl.dot，沿 DQ 并行) + `router_argmax_kernel`(grid=B，只读两块小张量) |
| 触发 | v3 有 **15 个 case** 报 `MLIRCompilationError`，原文 `ub overflow, requires 1859584 bits while 1572864 bits available`（232KB vs 192KB） |
| 机理 | 单 kernel 版要在一个 program 里吃下 `[BLOCK_DQ, BLOCK_KD]` 的 W1 tile。缩小 tile 能编译，但 D=1536/BDQ=512 时外层要 **192 次**迭代；更根本的问题是 `grid = B ≤ 4` ⇒ 最多 4 个核在读 `D²/4` 的 W1（D=1536 时 590K 元素 = 1.2MB/核） |
| 代价 | 多两个 kernel 名 ⇒ 按 kernel 名分组的时延口径下多两个分项。三段都很小（mean 读一遍 x；fc1 是 `[B≤4, DQ≤384]` 的 GEMM；argmax 只读 `[B,DQ]` + `[NOPT,DQ]`） |

### 1.4 继承自 21 号的改动

| 改动 | 增益来源 |
|---|---|
| QKV 三合一投影 + 权重 `__init__` 预转置成 `[in,out]` | `nn.Linear` 的 `[out,in]` 布局让 b tile 最内轴 stride = in_features（离散 gather）。MQA 上同类改动 5.94x |
| attention 直接按 `[B,S,H,dh]` 落盘 | 省掉参考实现的 `transpose(1,2).contiguous()` |
| `tl.dot` + online softmax 重写 attention | 21 号基线 attention 里一个 `tl.dot` 都没有，全是 head_dim 长的 python 标量循环 + 三维广播 |
| `grid = min(num_cores, total_tasks)`，`num_aicore=24` | 旧代码硬编码 14。MQA 上 +5.8% |
| PV 段直接用 fp32 `tl.dot` | 本机 `tl.dot` 支持 fp32 操作数且与 torch fp32 matmul `max_abs_err == 0`，**不需要** 80 号那套 `p_hi/p_lo` bitcast 拆分 |
| 合并段不物化 `[B,N,S,D]` | one-hot ⇒ 第 i 路的 out_proj 直接带 `(sel[b]==i)` 的 store 掩码写进同一块输出，每行恰好被写一次 |

---

## 2. 已排除的方向

### 2.1 本算子上直接证伪的

| 方向 | 结论 | 证伪了什么假设 |
|---|---|---|
| `b = m // S` 反推 batch + 按向量下标 gather `sel` | **aicore timeout(507014)**，并把整块卡打成持久损坏 | 证伪"挂死是因为 tile 太大/计算量太大"：把 M 传成 2（只剩一个 64×128 tile 且大半被掩码）**仍然挂**。真因是 runtime 标量除法 + 非仿射 gather 这个**写法** |
| `tl.arange(0, NOPT)`，NOPT ∈ {2,3} | 必然编译失败 | `tl.arange` 要求 2 的幂。**35/50 的 case 是 3 路** |
| 整块 `W1[DQ, D]` 一个 tile | UB 差两个数量级 | 证伪"router 很小所以怎么写都行"：D=1536 时 W1 是 590K 元素，UB 只有 192KB |
| 单 kernel router（缩小 tile 版） | 能编译但结构上不对 | 证伪"缩小 tile 就能解决"：迭代次数爆炸到 192 次，且并行度只有 B≤4 |
| 旧编译器 2^18 tile 预算 | 作废 | 1.2.0 是 2^20（4 倍）；失败模式也从 `hivm.hir.vsel root alloc` 变成老实的 `UB overflow` |
| `__init__` 预生成全部 45 组权重 | CPU 常驻 ≈1.2GB | 改成容量 1 缓存（与参考 `self._cache` 同语义），重建落在 warmup 内 |

### 2.2 21 号（每一路逐字相同）上已实测证伪、不必重跑的

| 方向 | 结论 | 证据 |
|---|---|---|
| PV 段换低精度 `tl.dot` | ❌ | 精度超阈 **1.6 倍**，速度只差 4%，且多数 tile 下 UB overflow |
| 编译参数（9 个逐个试） | ❌ | 全在 ±1% 噪声内；`ops_reorder` 在大 case 上 **−24% 有害**；`enable_mixed_cv` 确认 **950-only** |
| K 预转置消除 `tl.trans` | ❌ | fp32 档更慢 ×0.969，低精度档 UB overflow |
| attention tile 继续调 | ⚠️ 薄 | 128/128 只比各形状最优档慢 3~8%；低精度 dh≤64 档 256/64 能再拿 3~8%，**fp32 档 256/64 直接 UB overflow** |
| attention 循环嵌套压平 | ❌ 已最小 | IR 显示已是 2 层 × Cube/Vector 两份；136 个同步是 online softmax 每个 KV 迭代固有的 Cube↔Vector 往返，`aic_scalar_ratio 0.73` 是**等待被计入标量管线**，不是标量运算过多 |

### 2.3 算账即否、没有实测的

| 方向 | 否决理由 |
|---|---|
| 非融合三段式 attention | 物化 `[B*H,S,S]` 分数矩阵最大 case 需 51MB，往返约 206MB GM 流量 ≈ 206μs |
| 把 `qkv_sel_kernel` 与 `out_sel_kernel` 合并成同名 | 两者代码逐字相同，合并后"测得的"时延凭空少掉一项，真实 wall-clock 一秒不变 —— **口径把戏，刻意不做** |
| 把 mean 折进 fc1（`h = (1/S)Σ_s x[b,s,:]@W1ᵀ`） | 数学上等价但**数值上重排**：参考实现先把 `xm` 舍入到 x.dtype 再做 matmul。相对误差 ~2^-11 传到 logits，而 top1−top2 间隔只有 0.03~0.31 ⇒ argmax 翻转风险不可接受 |

---

## 3. 还没试的方向（按预期收益排序）

| 方向 | 预期 | 风险 |
|---|---|---|
| 低精度 dh≤64 档 attention tile 用 256/64 | 21 号实测 +3~8% | fp32 档必须保持 128/128（256/64 会 UB overflow），要按 dtype 分叉 |
| `qkv_sel_kernel` 的 BLOCK_M 随 S 自适应（S 小于 BLOCK_M 时白算） | S=3~29 的小 case 上 BM=128 会让 s_mask 掩掉 90%+ 的行 | 小 case 本来就快，收益可能被发射开销吃掉 |
| 三路 kernel 发射合并成一次（把 BRANCH 做成 task 坐标） | 省 2N 次发射开销，小 case 上可能显著 | 不同分支的 `n_kv` 不同 ⇒ N 维长度不同，tile 空间要按分支变长；且会把三个分支合成同名（口径问题） |
| `router_mean_kernel` 与 `router_fc1_kernel` 合并 | 省一个 kernel 分项 | 会让 x 的读取乘以 `ceil(DQ/BLOCK_N)` 倍 |

---

## ★ 测量口径变更与最终成绩（`triton_dev_v2`，2026-08-14 重测）

`triton_dev_v2` 的 `6e4dc13c fix(triton-op-verifier): benchmark 时延改为按单次调用计算，
支持 forward 内多次启动 kernel` 变更了时延口径：由整组行数反推发射次数
`L = len(group)/active_count`，取末尾 `L×active_count` 行求和再除以 `active_count`
⇒ **一次 forward 内的多次发射如实全部计入**（旧版按 kernel 名 groupby，只计一次）。

测量条件：NPU 1、`--lock-frequency`（实测无频率漂移）、仓库自带 `verify.py` / `benchmark.py`，
阈值未改、未用 `--verify_not_required`。

| 本算子 | framework | impl | speedup | 精度 |
|---|---|---|---|---|
| 旧口径 | 0.1877 ms | 0.0646 ms | 2.9033 | 50/50 |
| **新口径** | **0.5559 ms** | **0.1386 ms** | **4.0101** | **50/50** |

**impl 从 0.0646 涨到 0.1386（2.1x）**，因为三个 per-branch kernel
（`qkv_sel` / `gqa_attention` / `out_sel`）各按分支数发射 N 次（50 个 case 里 35 个是 3 路），
旧口径只计最后一次。framework 涨得更多（0.1877 → 0.5559，3.0x），因为参考实现把 N 路**全部**算满。
⇒ 分数不降反升，且这恰好**证明门控跳过是真加速**：被跳过的 program 虽被计入发射次数，
但立刻退出、耗时极低。

### 四个算子横向（同一口径、同一测量条件）

| 算子 | framework | impl | speedup |
|---|---|---|---|
| 21_GroupedQueryAttention | 0.1797 ms | 0.0715 ms | **2.5119** |
| 33_AdaptiveAttention | 0.5559 ms | 0.1386 ms | **4.0101** |
| 80_GqaSparseFwd | 2.9083 ms | 0.0488 ms | **59.5558** |
| 81_GqaSparseDecode | 3.1449 ms | 0.0750 ms | **41.9099** |

### 这条对「以后怎么优化」的两点影响

1. **旧口径下的「合并同名 kernel 刷分」「拆成不同名 kernel 会被罚」两条顾虑全部作废**，
   新口径下按真实 wall-clock 设计即可；
2. **旧口径下算出的天花板/缺口必须重算**。成本模型本身（每次 KV 迭代的固定 CV 握手、
   UB 预算、tile 分档）约束的是 **impl 的绝对时间，与口径无关，仍然成立**；
   失效的只是把它换算成 speedup 的那一步。
   ⇒ 本文件前面各节的 speedup 一律是**旧口径**，只用于**方案之间的横向比较**（impl 绝对时间没变），
   **不要与上表的绝对分数混用**。

### 经验已沉淀为可复用 template

本算子族的完整经验已写成 `cannbot-skills_5171` (`triton_dev_v2` 分支) 的
`plugins-official/triton-op-generator/template/gqa_sparse.md`，并接入 AGENTS.md 的
category 判定表（排在 `flash_attention` 之前，从上到下匹配）。
已用 Kimi 驱动 `triton-op-generator` 做端到端验证。

---

## ★★ 时延目标下的补充（2026-08-14，目标改为 impl 几何平均 ≤ 0.03 ms + speedup > 2x）

### 先算账：0.03 ms 等价于「四个算子同时达到 vendor peak」

| 算子 | impl 几何均值 | 理论下界 | 离下界 |
|---|---|---|---|
| 21_GroupedQueryAttention | 0.0715 ms | 0.0259 ms（vendor peak TFLOPS + 3 次 launch） | **1.99x**（逐 case 几何平均） |
| 33_AdaptiveAttention | 0.1386 → **0.0969** | 0.0307 ms | 3.16x |
| 80_GqaSparseFwd | 0.0488 | — | — |
| 81_GqaSparseDecode | 0.0750 | 0.0191 ms（1.18GB KV @1.2TB/s） | 3.9x |

**四者全部打到下界的几何平均 = 0.0294 ms。** 单是 kernel 启动开销的几何平均下界就有 0.005 ms
（实测最小 case 7.0~10.6 µs / 1~3 次发射）。⇒ 0.03 不是"难"，是贴着物理极限。

### 本轮唯一成立的改动：门控 kernel 由「每分支发一次」改为「整体发一次」

**33：impl 0.1386 → 0.0969 ms（1.43x），speedup 4.0101 → 5.7479，精度 50/50。**

新口径（`triton_dev_v2` 的 `6e4dc13c`）如实计入每次发射后暴露出来的问题：
`qkv_sel` / `gqa_attention` / `out_sel` 各按分支发 3 次，其中 2/3 不干活，
但**每次发射仍要把整个 task 空间走一遍**（`for tile in range(pid, num_tiles, num_programs)`
+ 内部标量 load + 比较）。

做法：分支权重打包成 `[nopts, D, n_max]` 连续缓冲 + 两张小表（每分支 kv 宽度、每分支 n_kv_heads），
kernel 内 `br = sel[b]` 取偏移；各分支宽度不等就按最大宽度铺 task 空间、一次标量比较跳过空 tile
（实测空转 n-tile 占 35.3%，但空转 tile 不做 GEMM，代价只有循环记账）。

### ⛔ 本轮被实测否掉的三条（都曾看似显然）

| 改动 | 结果 | 机理 |
|---|---|---|
| 80：PV 由 `p_hi/p_lo` 双原生 dot 改 **fp32 单 dot**（3→2 个 dot） | **慢 1.78 倍**（同时段 A/B：55.33 → 27.97） | fp32 dot 走**非 Cube 原生通路**，且每页多一份 `[BS,BLOCK_D]` f32 转换缓冲（64KB）。这两笔超过省下的一次 CV 握手 |
| 80：PV 改 **fp16 单 dot** | 精度 **48/50**（2 个 MERE 超阈 1.8 倍） | — |
| 21：PV 改 **fp16 单 dot** | 精度 **24/50**（7 个 fp16 精度失败 + 19 个 MLIRCompilationError） | 21 有 17 个 fp16 case，阈值 `2^-10` 比 bf16 的 `2^-7` 严 8 倍；`p` 降 fp16 的 `2^-11` 误差只剩 2 倍余量 |

⇒ **「循环内 dot 个数是成本单位」这条只在 CV 握手主导的形态（81）成立**，
在 80 上结论相反。判别信号：`aic_mac_ratio` 极低（0.02~0.05）且 `serial_ratio` 接近 2 ⇒ 握手主导；
否则算力/转换主导，此时**宁可多一个 dot 也要保原生 dtype**。

### attention kernel 的结构性上限

21 的逐 case 分解显示：**attention 只占算力的 7~14%，却吃掉约一半时间**（效率约为 GEMM 的 1/10）。
但 KV 迭代次数 ∝ `S² / (BLOCK_M × BLOCK_N)`，而 `BLOCK_M × BLOCK_N ≤ 2^20 / BLOCK_DH` 被 UB 钉死
（dh≤64 时 128×128，dh=128 时 128×64）。
⇒ **迭代次数由 UB 预算决定，不是调参能改的**——这解释了为什么 tile 全档扫描只有 3~8% 的空间。

### 测量纪律（今天两次被污染数据误导）

机器被别的会话占满时**绝对时延整体放大 6~10 倍**（framework 2.91 → 17.08 ms）。
`--lock-frequency` 只挡频率漂移，挡不住核/带宽争抢。
「取比值就好」也**不是普遍成立**：

| 算子 | torch framework | 干净 speedup | 重负载 speedup |
|---|---|---|---|
| 80（golden 发上千个小 kernel） | 2.91 ms | 59.56 | 55.33（−7%） |
| 21（golden 只有几个大 GEMM） | 0.18 ms | 2.5119 | 1.7681（**−30%**） |

⇒ 所有版本对比必须**同时段、同一块卡、背靠背连测**；开测前 `npu-smi info` 看 AICore 占用。

### ★★★ host 侧 `permute(...).contiguous()` 是达标与否的分水岭

新口径下这些**纯搬运 kernel 如实计入 impl**，而它们不产生任何计算。

**实测**：同一个算子（21），Kimi 生成版与手工版的 tile 配置**完全相同**
（fp32 `128/256/64`、低精度 `128/256/128`、`BLOCK_Q=BLOCK_KV=128`），QKV 融合投影也做对了，
唯一差别是生成版多了 **4 处 `permute(...).contiguous()`**：

```python
q = qkv_out[:, :D].view(B,S,H,dh).permute(0,2,1,3).contiguous()          # 搬运 1
k = qkv_out[:, D:D+KH*dh].view(...).permute(...).contiguous()            # 搬运 2
v = qkv_out[:, D+KH*dh:].view(...).permute(...).contiguous()             # 搬运 3
attn_flat = attn_out.permute(0,2,1,3).contiguous().view(B*S, D)          # 搬运 4
```

⇒ `impl 0.0715 → 0.1287 ms`（慢 **1.8 倍**），`speedup 2.5119 → 1.4420`，**从达标掉到不达标**。

正确做法：kernel 带 stride 直接读原始 `[M, q|k|v]` 布局（`k_base=D` / `v_base=D+d_kv`），
输出直接 `tl.store` 成 `[B, S, H, dh]`，一次搬运都不要。

**自查**：forward 里每出现一次 `.contiguous()`，都要问「这次搬运能不能用 stride 消掉」——
本类算子里答案几乎总是「能」。
