# 21_GroupedQueryAttention 性能优化方向与收益分析

> 面向"下一个做同族算子的人"。每条都标注**实测收益**与**适用条件**，
> 未实测的一律标为"未验证"，不写成结论。

## 0. 算子形态

```
x [B,S,D] --Wq/Wk/Wv--> q[B,H,S,dh] k,v[B,KH,S,dh]
                        k,v repeat_interleave 到 H 头 (GQA)
scores = q.float() @ k.float().T        # 注意: golden 在 fp32 域做
weights = softmax(scores / sqrt(dh))
out = (weights @ v.float()).to(x.dtype)
return out_proj(out)
```

- 50 个 case：B ∈ [1,4]，S ∈ [3,897]，D ∈ [48,1536]，H ∈ [2,24]，KH ∈ [1,24]；
  dh 多数为 64，另有 {20,24,32,48,56,60,72,78,80}。
- dtype 三档各约 1/3：fp16 / bf16 / fp32。
- **形态特征：小 shape 多、launch 开销占比高**。S ≤ 61 的 case 已能到 5~7x，
  几何平均被 10 个大 case（< 1x）拖住。

## 1. 收益排序（本算子实测）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | **attention 用 `tl.dot` + online softmax 重写** | 0.0016 → 1.4518（**约 900x**） | 基线里没有 `tl.dot` 时。先看源码有没有 `for d in range(head_dim)` 这类标量循环 |
| 2 | **GEMM tile 按 dtype 分档** | 1.4518 → **2.2839（+57%）** | 任何有独立投影 GEMM 的算子。**这是本算子最大的单项调优** |
| 3 | tile 预算按当前编译器重扫 | 含在 #1/#2 内 | **结构性改动后必做**，不要查历史清单 |
| 4 | QKV 三合一 + 权重预转置 `[in,out]` | 含在 #1 内 | 有多个共享输入的 `nn.Linear` 时 |
| 5 | 输出直接按目标布局落盘 | 含在 #1 内 | golden 里有 `transpose(1,2).contiguous()` 时 |
| 6 | attention tile 低精度档 128/128 → 256/64 | +3~8%（仅该档 case） | 低精度且 BDH ≤ 64 |

## 2. 各 dtype 档的 tile 可行域（bishengir 1.2.0 实测）

**这是本算子最重要的一张表**，`sweep_linear.py` / `sweep_attn.py` / `probe_block2.py` 扫出来的。

### 2.1 UB 预算边界

```
fp16/bf16:  128x128x 64 = 2^20  OK        128x128x128 = 2^21  UB overflow
            128x 64x128 = 2^20  OK        256x128x 64 = 2^21  UB overflow
            256x 64x 64 = 2^20  OK        128x256x 64 = 2^21  UB overflow
fp32:       128x128x 64 = 2^20  OK        128x 64x128 = 2^20  UB overflow  <== 注意
             64x 64x128 = 2^19  OK
```

⇒ 预算大致是 `BM*BN*BDH ≤ 2^20`，但 **fp32 下 BDH=128 比同乘积的 BDH=64 更吃 UB**
（操作数字节翻倍），**不能只看乘积，要按 dtype 分档**。

### 2.2 GEMM 最优 tile（相对拍脑袋的 64/128/64）

| 形状 | dtype | 最优 BM/BN/BK | 加速 |
|---|---|---|---|
| M=879 N=4032 K=1344 | fp32 | 128/256/64 | **x3.56** |
| M=766 N=4416 K=1472 | fp32 | 256/128/64 | **x3.46** |
| M=634 N=1536 K=1408 | fp16 | 128/256/128 | **x5.15** |
| M=419 N=2048 K=1536 | fp16 | 256/128/128 | **x4.87** |

低精度吃得下 `BK=128`，fp32 吃不下（`64/256/128`、`128/256/128`、`256/64/128`、`256/128/128` 全 UB overflow）。

### 2.3 attention 最优 tile

| 形状 | dtype | 当前 128/128 | 最优档 | 收益 |
|---|---|---|---|---|
| B3S897 D512 H8 KH2 | fp16 | 465.7us | 256/64 = 452.8us | x1.03 |
| B2S641 D768 H12KH1 | bf16 | 272.0us | 256/64 = 251.7us | x1.08 |
| B1S769 D640 H10KH10 | fp32 | 173.0us | **128/128 已最优**（256/64 UB overflow） | — |
| B2S509 D1024H16KH16 | fp32 | 197.2us | **128/128 已最优** | — |
| B3S463 D1280H16KH1（dh=80） | bf16 | 不可用 | 128/64 = 454.7us | — |

⇒ **attention tile 基本没有空间**（3~8%），不要在这里花时间。

## 3. dtype 契约

- **本机 `tl.dot` 支持 fp32 操作数，且与 torch fp32 matmul `max_abs_err == 0`**。
  golden 里 `softmax(fp32) @ v.float()` 这种真 fp32 段可以直接用 fp32 dot 逐位同构。
- QK 段 q/k 从低精度内存加载 ⇒ **保持原生 dtype + fp32 累加**。升 fp32 不增加任何有效位。
- **不要为了"避开 fp32 dot"去做低精度拆分**（见下节证伪）。
- 精度阈值（`verify.py get_limits`，rel_threshold 同时是 MERE 判据）：
  fp16 `2^-10`、bf16 `2^-7`、fp32 `2^-13`。**fp32 那档比想象的松**。

## 4. 已证伪的方向（省下你的时间）

| 方向 | 结论 | 证据 |
|---|---|---|
| PV 段换低精度 dot（含 `p_hi/p_lo` 双 dot） | **整条路线不成立** | 精度超阈 1.6 倍（bf16 MERE 1.278e-02 > 7.8125e-03）；速度只 x1.04；多数 tile 下 UB overflow |
| 编译参数 | **无收益，一个有害** | 9 个逐个试全在 ±1.6% 内；`ops_reorder` **-24%**；`enable_mixed_cv` 确认 950-only |
| K 预转置消除 `tl.trans` | **更慢或编不过** | fp32 x0.969；低精度 UB overflow |
| 靠 tile 继续压 attention | **没空间** | 全档扫描后最优档只快 3~8% |
| 堆同组 q head 复用 KV | **账上不成立** | tile 行数被 UB 钉死，堆 head 就得减每 head 行数，任务数等比例上升，总访存不变 |

## 5. IR 读数

`work/ir/out/gqa_attention_kernel_last_pass.mlir`：

- **`scf.for` 显示 4 个 ≠ 4 层嵌套**。是 2 层循环 × Cube/Vector 两份
  （`hivm.func_core_type` 66 处、`hivm.part_of_mix` 2 处）。
  iter_args 为 `memref<64x64xf32, ub>` —— `BLOCK_M=128` 被 2 个 vector 子核对半分。
  **读 IR 时别把 mix kernel 的两份代码当成额外的循环嵌套。**
- 同步操作 **136 个**（`set_flag` 37 / `wait_flag` 37 / `pipe_barrier` 26 /
  `sync_block_set` 18 / `sync_block_wait` 18）。
- ⇒ `aic_scalar_ratio 0.73` 是**等待被计入标量管线**，不是标量运算多。
  这一条与 MQA 完全一致，是同族算子的通用读数陷阱。

## 6. 测量口径的坑

`benchmark.py` 的 `parse_operator_latency` 按 **kernel 名** groupby，取最后 `repeats` 行除以
`repeats` ⇒ 得到的是**每次发射的平均耗时**。所以：

- 同一个 kernel 一次调用里发射 N 次，**只按一次计**（本算子 `linear_kernel` 发 2 次，只按均值计一次）；
- 总时延 = Σ(每个**不同名** kernel 的单次平均耗时)；
- ⇒ 把不同名 kernel 合并成同名、或把一次大发射拆成多次小发射，都会让"测得的"时延变好看，
  但那不是真加速。**按真实 wall-clock 优化，报告里如实说明口径。**

`speedup_vs_torch` 是各 case speedup 的**几何平均**，小 case 与大 case 等权。
本算子小 case 已 5~7x，别只盯着大 case。

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
| 旧口径 | 0.1206 ms | 0.0528 ms | 2.2839 | 50/50 |
| **新口径** | **0.1797 ms** | **0.0715 ms** | **2.5119** | **50/50** |

**impl 从 0.0528 涨到 0.0715**，因为本算子的 `linear_kernel` 一次 forward 里确实发射 **2 次**
（QKV 融合投影 + out_proj），旧口径漏计了其中一次；framework 同步从 0.1206 涨到 0.1797。
⇒ 第 1 节里「QKV 三合一」这类**减少 kernel 发射次数**的改动，在旧口径下会让分数变差、
在新口径下才被如实奖励。

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
