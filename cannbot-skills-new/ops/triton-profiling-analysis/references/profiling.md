# Profiling 解读（msprof）与 IR 交叉验证

> 完整的采集流程（环境、命令、覆盖门禁）归 `triton-simulator-optimizer` 所有，本文只覆盖
> **"拿到 CSV 后怎么看"**以及**如何用 IR 解释 profiling 数字**。

## 1. 两种 msprof 模式的选择

| 模式 | 命令 | 产出 | 适用 |
|---|---|---|---|
| simulator | `msprof op simulator --application=... --kernel-name=<名> --launch-count=1 --soc-version=<SOC>` | 逐指令 `instr_exe.csv` + 按源码行 `code_exe.csv`（每核目录一份） | 诊断瓶颈类型与热行（唯一能做指令级归因的方式） |
| board | `msprof op --application=... --aic-metrics=PipeUtilization,MemoryUB` | kernel 级时间 + pipe 概览 | 真实 NPU、秒级；无指令级，只做粗确认 |

**simulator 采集要点**（细节见 simulator-optimizer/references/msprof-simulator.md）：
- 脚本只跑**单次 forward + 最小 shape**（整应用所有 kernel 都被逐指令仿真，不只 --kernel-name 那个）
- `--kernel-name` 是**大小写敏感子串匹配**；CANN 算子名如 `FlashAttentionScore_`
- 子进程退出码非 0（如 134）不一定失败，以 CSV 是否落盘为准
- 产物用 `find prof_out -name "*_instr_exe.csv"` 定位，勿假设层级
- 忙核 vs 闲核：任务数 < 核数时部分核只有 SCALAR 开销（~5K cycles 的启动壳），分析只看忙核

## 2. instr_exe.csv 读法（瓶颈类型）

按 `cycles` 降序，按 pipe 聚合占比：

| pipe | 含义 | 异常信号 |
|---|---|---|
| `CUBE`（MMAD） | 矩阵乘真实算力 | 计算密集 kernel 应 >50%；**远低于此 = cube 在干别的** |
| `VECTOR` | vector 计算（exp/max/sum...） | 正常计算 |
| `FIXP` | L0C→UB（dot 结果搬给 vector） | 高 = cube/vector 交接频繁 |
| `FLOWCTRL` | set/wait flag 空等 | **高 = 跨 pipe/跨核等待（WAIT_FLAG_DEVI）** |
| `MTE2` / `MTE3` | GM↔片上 搬入/搬出 | 高 = 访存密集或中间结果走了 GM |
| `MTE1` | L1→L0（喂 cube） | cube 供数路径 |
| `SCALAR` | 标量指令 | calls≈元素数 → 标量降级；一次性高占比（DC_PRELOAD 等）可能是单跑冷启动 artifact，剔除后重算占比 |
| `BAR` | barrier | 高 = 同步密集 |

解析脚本：`triton-simulator-optimizer/scripts/parse_prof.py <prof_out>`。

## 3. code_exe.csv 读法（瓶颈位置）

`code` 列 = `源文件:行号`（开 `TRITON_DISABLE_LINE_INFO=0` 时直达 .py 源码行）。
**多行 call_count 相近 = 同一循环体的相邻语句**。定位到行后读 ±5 行看硬件语义。

## 4. IR 交叉验证方法论（本 skill 的核心增量）

profiling 告诉你"什么 pipe 热"，IR 告诉你"为什么"。两边必须互洽：

```
FLOWCTRL 高（cube 空等）
  → 查 npuir.mlir（GraphSyncSolver dump）里 sync_block_set/wait 的位置与密度
  → 查 cv-pipelining 是否生效（before/after diff）
MTE3 高（vector 写 GM）
  → 查 IR 里 fixpipe 的 outs 地址空间 / alloc_workspace → 中间结果走 GM 实锤
FIXP 高
  → IR 里每个 mmadL1 后是否紧跟 fixpipe 落 GM（而非 UB 直通）
某编译选项无收益
  → before/after diff 该选项控制的 pass；前后一致 = pass 拒转 = 结构性原因
```

**互洽即高置信；矛盾时以 IR 为准**（simulator 占比可能被超长指令拉偏，且小 shape 有冷启动噪声）。

## 5. 案例：FA 单流 vs 双流（2026-09-04 实测，可作为解读模板）

同 shape（1×1×2048×64, BM=BN=128）稳态 simulator 对比：

| 指标（忙核 core0） | 单流 | 双流 | IR 解释 |
|---|---|---|---|
| 每核总 cycles | 474,250 | 316,523（-33%） | 双流每个 work item 装 2 tile，交接次数减半 |
| Cube FLOWCTRL | 13.4% | 10.1% | 跨核握手减少（sync_block 38→22） |
| Vector MTE3 | 18.4% | 9.5% | P 写回 GM 次数减半（宽度翻倍） |
| Cube MMAD | 16.6% | 15.2% | 计算量相同 |

注：两版 CVPipelining **均生效**（循环都拆成 work-item 子循环）；差距来自 stage 交接密度。
结论模板：**"流水更优"= IR 层交接密度下降 + simulator 层总 cycles/等待下降，两者互证。**

⚠️ 本案例的反面教训：曾两次误判——① 用"双流64 ≈ 单流128"误判"流水无收益"（两种机制恰好等效）；
② 手动重放缺 `--enable-hfusion-compile` flags + 文本切分 bug 误判"pass 拒转"。
**pass 是否生效一律用 `scripts/diff_pass.sh` 复核，不要手写文本切分。**

## 6. 解读纪律

- 先看**相对占比结构**，再看绝对 cycles；跨 kernel 比绝对 cycles 前先确认两边工作量相同
- 小 shape（迭代数 ≤2）流水首尾效应主导，占比不代表稳态——需要稳态就加大迭代数（如 16 次 KV 迭代）
- 一次性 SCALAR 高峰（DC_PRELOAD/LD_XD 等，calls 几十、单次几百 cycles）是单跑冷启动 artifact，
  剔除后重算占比再下结论
- 禁止只看 cube 核——vector 核的两张表解释"cube 在等什么"
