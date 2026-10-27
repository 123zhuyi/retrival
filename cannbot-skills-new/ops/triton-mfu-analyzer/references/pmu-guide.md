# PMU 是什么，以及怎么读它

## 一句话

**PMU = Performance Monitoring Unit（性能监控单元）**，是烧在 AI Core 硅片上的一组
硬件计数器。它在 kernel 执行时**由硬件自动累加**，记录各条执行流水各自"忙了多少周期"。

它不是采样，不是估算，也不是软件插桩——是硬件事实。这是它和墙钟计时、
`torch.profiler` 等软件侧手段的本质区别。

## 为什么 MFU 之外还需要它

MFU 只告诉你**"离峰值算力有多远"**，是一个标量：

```
mfu = 理论 FLOPs / kernel 耗时 / 芯片峰值算力
```

MFU = 3% 时，它无法回答**"那 97% 的时间在干什么"**。可能是：

- cube 在等数据搬进来（访存受限）
- cube 在等标量单元算地址（标量受限）
- 根本没用 cube，整个算子跑在 vector 上
- 流水都没排满，在等依赖

这四种情况的优化手段完全不同，而 MFU 这一个数字区分不了。
**PMU 计数器就是用来做这个区分的。**

## 昇腾 AI Core 的执行流水

一个 AI Core 内部是多条**可并行**的流水线，各司其职：

### AIC（Cube 核）—— 负责矩阵乘

| 计数器 | 流水 | 干什么 |
|---|---|---|
| `aic_mac_ratio` | MAC（cube） | **矩阵乘本身**。这条就是"算力" |
| `aic_mte2_ratio` | MTE2 | **GM → L1** 搬运（把数据从 HBM 搬进片上缓存） |
| `aic_mte1_ratio` | MTE1 | **L1 → L0** 搬运（送进 cube 的输入缓冲） |
| `aic_fixpipe_ratio` | FixPipe | 结果的量化/搬出 |
| `aic_scalar_ratio` | Scalar | 标量指令：**地址计算、循环控制、分支判断** |

### AIV（Vector 核）—— 负责逐元素运算

| 计数器 | 流水 | 干什么 |
|---|---|---|
| `aiv_vec_ratio` | Vector | 向量计算（softmax、激活、归一化…） |
| `aiv_mte2_ratio` | MTE2 | 读入 |
| `aiv_mte3_ratio` | MTE3 | 写出 |
| `aiv_scalar_ratio` | Scalar | 标量指令 |

## ratio 的确切含义

```
aic_<pipe>_ratio = aic_<pipe>_time / aicore_time
aiv_<pipe>_ratio = aiv_<pipe>_time / aiv_time
```

即**该流水的忙时占核心活跃总时长的比例**，是**占空比**。

三个必须记住的性质：

1. **各流水并行，所以所有 ratio 之和可以远大于 1。**
   搬运和计算本来就该重叠，`mte2 0.9 + mac 0.8` 是好事不是矛盾。
2. **ratio 高 ≠ 效率高。** 它衡量"这条流水有多少时间在发指令"，
   不衡量"每条指令干了多少活"。
   实测反例：某 decode attention 在 `head_dim=64` 时 `aiv_mte2 = 0.766`、
   达成带宽 265 GB/s；`head_dim=256` 时 `aiv_mte2` 只有 0.594、
   达成带宽却是 **648 GB/s**。原因是 D 决定访存的连续段长度
   （256×2B=512B vs 64×2B=128B），窄事务把指令条数翻了 4 倍——
   流水更"忙"，有效带宽反而掉一半以上。
   **判访存 bound 必须 ratio 与达成带宽一起看**：
   ratio 高 + 带宽高 = 真撞 HBM 墙；ratio 高 + 带宽低 = 访存**模式**差，是可优化项。
3. **ratio 恒 ≤ 1；出现 >1 说明计数不自洽**（通常是 MIX kernel 的 `aicore_time`
   相对各核实际忙时被低估）。此时该 kernel 的 PMU 数据不可用，
   应标注"计数异常"而不是拿它当一条饱和的流水。

## 怎么采

```bash
msprof --output=<dir> --ai-core=on --task-time=on \
       --aic-metrics=PipeUtilization --ascendcl=on \
       python3 <runner.py>
```

- `--aic-metrics=PipeUtilization` 才会输出上面这组 ratio；
  换成 `ArithmeticUtilization` 输出的是另一组（`aic_cube_fops`、`aiv_vector_fops`、
  各 dtype 的 mac 占比），可用来和自己算的 FLOPs 交叉验证。
- 产物在 `<dir>/PROF_*/mindstudio_profiler_output/op_summary_*.csv`，一行一次 kernel 启动。
- **开 PMU 后 kernel 耗时会放大 2~16 倍**（计数器读出有开销）。
  所以：**ratio 是核内相对量，可以用；绝对耗时要在不开 PMU 的采集里另测。**

## 怎么判 bound

**判据**：某条流水的 ratio ≥ 阈值（一般取 0.7~0.85）才算该流水是瓶颈。
没有任何一条到线 → **依赖/延迟 bound**（流水没排满，在等依赖，不是任何单一资源打满）。

| 最高的那条 | 结论 | 优化方向 |
|---|---|---|
| `aic_mac` | 算力 bound | 已到头，只能改算法降 FLOPs |
| `aic_mte2` | 访存 bound（GM→L1） | 提高数据复用、改 tiling、增大算术强度 |
| `aic_mte1` | L1→L0 bound | 调 L1 tiling |
| `aic_scalar` / `aiv_scalar` | 标量 bound | 减少地址计算与分支，简化间接寻址 |
| `aiv_vec` | 向量 bound | 逐元素段是瓶颈，考虑融合 |
| 都没到线 | 依赖/延迟 bound | 提高并行度、软流水、消除串行依赖 |

### 还要看"核活跃"

```
核活跃 = Σ min( max(aicore_time, aiv_time), Task Duration ) / Σ Task Duration
```

取 `max` 而非求和：AIC 与 AIV 是并行的两套单元，时间高度重叠，相加会重复计数；
`min(..., Task Duration)` 是因为计数窗口与任务窗口不完全对齐，偶尔会略微超出。

**必须用 `max` 而不是只看 `aicore_time`**：纯 vector kernel 的 `aicore_time` 天然为 0，
只除 `aicore_time` 会把它误判成"核外空转 100%"。

核活跃低（< 0.8）意味着瓶颈在核外——下发、同步、尾部效应。
但**低核活跃要先排除采集争用**：实测同一 shape 同一 kernel，
16 路并发 msprof 下核活跃 52.6%、单次 666 ms，
4 路并发下核活跃 99.2%、单次 339.5 ms，而 `aiv_time` 两边都是 336.6 ms 纹丝不动——
多出来的时间全落在"核不活跃"里，是采集开销而非算子特性。

## 常见误读

| 误读 | 实际 |
|---|---|
| ratio 之和 > 1 是数据错了 | 流水并行，重叠是正常且期望的 |
| `aic_mte2` 高就是撞了 HBM 墙 | 还要看达成带宽；带宽低说明是访存模式差 |
| `aicore_time = 0` 说明核在空转 | 纯 vector kernel 本就不用 cube，要看 `aiv_time` |
| MFU 低就是算力没打满 | 先看是哪条流水到线；很多时候 cube 只是在等标量或搬运 |
| 开 PMU 测出的耗时可以直接用 | PMU 会放大 2~16 倍，绝对耗时要另测 |
