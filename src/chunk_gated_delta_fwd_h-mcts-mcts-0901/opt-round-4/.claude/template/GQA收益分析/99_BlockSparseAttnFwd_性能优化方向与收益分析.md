# 99_BlockSparseAttnFwd（sparse-块跳过型）性能优化方向与收益分析

面向"块稀疏 / 掩码跳块"这一类前向 attention 算子。所有结论都在
CANN 9.1.0 + triton_ascend 3.2.2 + 910B2C 上实测过，标注了实测收益。

## 一、先按收益排序的检查清单

| # | 方向 | 本算子实测收益 | 适用判据 |
|---|---|---|---|
| 1 | **把"1 行 query / program"改成"BLOCK_M 行 / program"** | **26.25×**（几何平均） | grid 里出现 `total_q` 或 `seq_len` 这一维就一定要查 |
| 2 | **手写点积改 `tl.dot`** | 含在上条内 | kernel 里出现 `tl.sum(a[None,:]*b, axis=1)` 即命中 |
| 3 | **索引/掩码准备搬到 CPU 张量** | framework 口径下不计入，但省掉 NPU 上的 aclnn 算子 | 见第三节 |
| 4 | **tile 取到掩码块粒度（128）** | 使掩码行不必再拆，K/V 复用最大 | 受 UB 预算约束，必须实测扫 |
| 5 | 掩码合并成一个 `[BM]` 向量 | UB 省 3 个 `[BM,BN]` int32 临时量 | 编不过报 `ub overflow` 时先查这个 |
| 6 | CV 重排（dot 提前） | 串行度 1.66 未变，收益不显著 | 见第五节 |

## 二、方向 1/2：为什么"逐 token program"是这类算子的头号问题

块稀疏 attention 的掩码粒度是 128×128。基线按 `grid=(total_q, H)` 逐 token 发 program，
于是同一个 128 行掩码块里的 128 个 query，**每一个都要把整块 K/V 搬一遍**，
K/V 的 GM→UB 流量被放大了 128 倍；同时 M=1 的"矩阵乘"退化成向量点积，Cube 闲置。

改成 `(batch, q_tile=128, head)` 之后：
- K/V 每块只搬一次，访存降到 1/128；
- `tl.dot` 的 M=128、K=D、N=128，是 Cube 的正常工作形状；
- program 数从 22.6 万降到 `min(24, total_tasks)`，调度开销可忽略。

实测 impl 平均时延 2.2833 ms → 0.0868 ms，几何平均 speedup 2.7285 → 71.63。
50 个 case 全部提升，最小 9.74×、最大 77.17×。

**判据**：只要 kernel 的 grid 里出现 `total_q` / `seq_len` 这一维，且掩码是块粒度的，
就一定能这样合并。合并的上限是掩码块粒度（这里是 128），再大就要跨掩码行、得拆表。

## 三、方向 3：索引准备放 CPU 还是 NPU

块稀疏算子普遍需要一张"哪些列块被选中"的表。三种放法：

| 放法 | 是否进 benchmark 的 operators | 备注 |
|---|---|---|
| NPU 上用 torch 算（`repeat_interleave`/`argsort`/`nonzero`…） | **进**，而且常常很贵（`aclnnSort` 会落到 AiCpu） | 最差 |
| CPU 张量上算完再 `.to(device)` | **不进**（CPU 算子不产生 NPU kernel） | 本算子采用 |
| 专门写一个 Triton kernel | 进，但通常很便宜 | 表很大时用（如 93 号的 CSR 展开） |

选择依据是**表的大小**：本算子的表是 `[B, NROW, H, NCOL]`，B≤4、NROW≤6、H≤128、NCOL≤11，
只有几万个 bool，CPU 上向量化构造是微秒级。93 号那种 `nnz` 到百万级的就必须上 Triton kernel。

⚠️ 同时这也是 AST 退化预检的要求：`forward()` 里禁止 Python for/while 和绝大多数
`torch.*`；可达 helper 里只禁"主链计算算子"（`matmul`/`einsum`/`softmax`/**`cumsum`**/
`masked_fill`…），索引整形类（`argsort`/`repeat_interleave`/`sum`/`clamp`）是允许的。
**`cumsum` 被禁**这条很容易踩到——前缀和可以用 `torch.searchsorted` 或直接
`argsort` 紧致化绕开。

## 四、方向 4/5：tile 可行域由 UB 预算决定，且随编译器版本变

同一份代码：

| 编译器 | `--enable-ubuf-saving` | D=128 可编 tile | D=64 可编 tile |
|---|---|---|---|
| bishengir 0.1.0（CANN 8.5.1 自带） | **不支持**，传了直接 `Unknown command line argument` | 只有 32×32 | 到 64×32 |
| bishengir 1.2.0（triton_ascend 3.2.2 自带） | 支持 | **128×128** | **128×128** |

差了 16 倍的 tile 面积。**任何"最优 tile"的历史结论在换编译器后全部作废，必须重扫。**
扫法：`compile_probe.py` 只跑一个 case，看编译报错里的
`ub overflow, requires X bits while 1572864 bits available`，按 X 反推砍哪个中间张量。

省 UB 的两个通用手法（本算子实测有效）：
1. 把 causal 上界和序列边界**压成同一个 `[BM]` 向量**再做一次 2D 比较。
   原写法 `n_valid[None,:] & (offs_n[None,:] <= offs_m[:,None]+d)` 会物化 3 个
   `[BM,BN]` 的 int32 广播临时量。
2. 整行全掩码的处理用"masked score 填 −3e38 + m 下钳到 −1e30"，
   `exp(−3e38 −(−1e30))` 精确为 0，于是**不需要**第二次 `tl.where(keep, p, 0)`，
   直接省掉一个 `[BM,BN]` fp32 张量；末尾用 `acc / max(l, 1e-30)` 再省一个 `[BM,D]`。

## 五、方向 6：CV 流水的实测边界

优化后的瓶颈画像：`aic_mac_ratio` 只有 **0.0986**，`aic_scalar_ratio` **0.4854**，
串行度 `(aic+aiv)/Duration = 1.6610`（>1.5 即 Cube-Vector 串行）。

- 高 `aic_scalar_ratio` **不等于**标量运算多。MQA 上按这个 ratio 三次判定
  "标量运算过多"全部落空（`tl.make_block_ptr` 无收益、整数比较转 fp32 直接 0/50 崩、
  批处理 2 个 KV tile 回归 −14%），dump IR 后发现是 117 个同步操作的等待被计进了标量管线。
  **要下"标量退化"结论前必须先 dump IR 确认**，不能只看 ratio。
- 已做的 CV 重排（Diff 7，把 `tl.dot(p,v)` 提前到 `alpha`/`p_sum` 之前）是**单次迭代内部**
  的重排、数值等价；串行度仍是 1.66，说明这一步没打开 CV 重叠。
- 剩余空间的判断：`aic_mac_ratio` 低 + 串行度高 → 瓶颈在**每个小块的调度与同步**，
  不在访存也不在算力。下一步应该按 IR 的循环嵌套层数与 `set_flag`/`wait_flag`/
  `pipe_barrier` 数量去优化，而不是继续扫 tile。

## 六、已排除的方向及其证据

| 方向 | 结论 | 证据 |
|---|---|---|
| `enable_ubuf_saving` 开关 | 在 bishengir 0.1.0 上**不可用**（编译直接失败），1.2.0 上可用且是 128×128 能编过的前提 | 见第四节表 |
| 更小的 tile（32×32 / 64×32） | 在 1.2.0 上没必要，且会把掩码行拆成 4 份、K/V 重复搬运 4 倍 | `_logs/scan99b_*.log` |
| 把 q/k 升 fp32 再 `tl.dot` | 过度对齐。q/k 本来就是 bf16/fp16 存储，低精度操作数的乘积在 fp32 累加器里本来就精确 | — |
| 把 p 单段降 bf16 喂 `tl.dot` | **不行**。p 是真 fp32 量，误差被 `Σp|v|/|Σpv|` 放大约 20 倍，实测 rel 2~4%（阈值 7.8e-3） | 见迭代日志轮 B |

## 七、精度：这类算子唯一需要拆分的量

`tl.dot` 的 dtype 契约在本算子上的具体落点：

```python
# ❌ q/k 升 fp32 —— 输入本来就只有 8/11 bit，一位有效数字都不增加
s = tl.dot(q.to(tl.float32), k.to(tl.float32))
# ✅ 操作数保持原生 dtype，累加器 fp32
s = tl.dot(q, k)

# ❌ p 单段降精度 —— p 是真 fp32 量
pv = tl.dot(p.to(DTYPE), v)
# ✅ hi + lo 二段拆分，有效位翻倍，代价是 PV 的 Cube 工作量 ×2
ph = p.to(DTYPE)
pl = (p - ph.to(tl.float32)).to(DTYPE)
pv = tl.dot(ph, v) + tl.dot(pl, v)
```

修完 MERE 从 0.43 → 1e-6，`max_abs_diff` 3.9e-3（正好是最终 bf16 舍入本身）。

## 八、测量口径（写报告时必须交代）

`benchmark.py` 的 `parse_operator_latency` 按 kernel `Name` 分组、取最后 `repeats` 行
再除以 `repeats`，**同名 kernel 一次 forward 内发射多次只计一次**。
本算子实测 impl 侧 `launch/call = 1.0`、framework 侧 `237~425`，
所以 speedup 的分母准、分子保守。做这类"参考实现是 Python 循环"的算子时，
必须把 `launch/call` 一起报出来，否则 speedup 数字无法解释。
