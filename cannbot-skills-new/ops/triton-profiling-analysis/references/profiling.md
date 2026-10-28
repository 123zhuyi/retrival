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

## Memory.csv 数据通路字段表（2026-10-26 定版：通路级验证已完成）

> 完整审计流程（理想量→实测→逐变量归账→冗余判定→效率比）见 [channel-traffic-audit.md](channel-traffic-audit.md)。

### 通路 → 字段映射（取数方法：逐子核行求和，单次发射确认后为该发射真实流量）

| 通路 | 字段（cube/AIC 行） | 字段（vec/AIV 行） | 通路级验证 |
|---|---|---|---|
| GM→L1 读 | `GM_to_L1_datas(KB)` | —（vec 无此列） | **1.007×** vs last_pass 读点清单模型（c1: 580 vs 576KB/核） |
| L1→GM 写 | `L1_to_GM_datas(KB)(estimate)` | — | estimate 口径，无代码对应项（52KB，悬置） |
| L0C→GM 直写 | `L0C_to_GM_datas(KB)` | — | **1.000×** vs IR fixpipe f32@gm 模型（480 vs 480KB/核） |
| L0C→L1 | `L0C_to_L1_datas(KB)` | — | 计数器存在；本 kernel = 0（片上直达通路未用） |
| GM→UB 读 | — | `GM_to_UB_datas(KB)` | **1.035×** vs IR 读回模型（289.8 vs 280KB/核） |
| UB→GM 写 | — | `UB_to_GM_datas(KB)` | 逻辑量 ×2.0 吻合（行计费假设，见下） |
| 主存读（接口侧视角） | `read_main_memory_datas(KB)` | 同名列 | 与 GM→L1 / GM→UB 互证一致（<2%） |
| 主存写（接口侧视角） | `write_main_memory_datas(KB)` | 同名列 | 与 L0C→GM / UB→GM 互证一致（<2%） |

### 跨通路守恒检验（判断采集可信的快速门）

- **发送 ≈ 接收**：cube `L0C_to_GM` 合计 ≈ vec `GM_to_UB` 中间结果份额（c1: 7.68 vs 7.96MB，差 2.4%）
- **侧和 ≈ 总账**：两侧 read_main 之和 ≈ 总读、write_main 之和 ≈ 总写
- 任一不成立 → 采集口径有问题，先查发射次数（OpBasicInfo 行数 ×1 确认）再谈数据

### 已定案的语义细节

- 主存读写 = **NPU 板载 HBM ↔ AI 核**经 L2 的流量，不是主机内存↔显存（PCIe 交换不在这些计数器里）
- 同一 OPPROF 内不同 CSV 时长互不相同（OpBasic 58µs / Memory 52.6 / PipeUtil 85.2）——分指标组多 pass 采集，跨 CSV 字段禁止混合计算
- 半行写计费：fp32 整行写 1.02× 零放大 vs bf16 `[32,64]`=64B 半行写精确 2.0×——写侧放大 = 行粒度计费；"写缺失先读整行(RMW) 落主存"已被 read_main 账本排除（无伴生读余量）
- 缓存行大小 128B 仍为假设（chip_spec 无此字段），行计费结论随行大小证伪需重算
- L2Cache.csv 的 hit/miss/allocate 列是**事务计数（行数）**，非字节

### 草图 vs 实测（c1 实证，2026-10-26）——草图口径的系统性盲区

| 通路 | 草图推理 | CSV 实测 | 差异 | 根因 |
|---|---|---|---|---|
| GM→L1 读 | 338.5 KB/核 | 580.0 KB/核 | 1.71× | 草图缺编译器读点（v_dec 回读 cube 8KB/迭代，IR L145） |
| GM→UB 读 | 40 KB/核 | 289.8 KB/核 | **7.2×** | 草图只给了 u；实测 83% 是中间结果读回 |
| L0C→GM 直写 | **0**（假设片上交接） | 480.0 KB/核 | **从 0 到 7.68MB** | 草图最大盲区：mix 拆核后 dot 结果必须经 GM |
| UB→GM 写 | 128 KB/核（逻辑） | 249.4 KB/核 | 2.0× | 半行写行计费 |
| L1→GM 写 | 0 | 52.1 KB/核（estimate） | 0→52KB | 未归因 |

总流量：草图 11.05 MB vs 实测 33.36 MB = **3.02×**。
结论模板：**草图 tile 层取数逻辑量正确（IR 证实无隐藏 tile），草图错在通信模型——假设单核执行、无子核间交换；mix 拆核架构下，凡"跨计算步的中间量"都要过 GM，草图逐项漏计。评估 mix 类 kernel 时，草图流量下限须加"中间结果往返"项：Σ(各 dot/中间张量 f32 字节 × 迭代数 × 2 方向)。**

### 使用纪律（本轮教训）

1. 流量判定唯一权威 = read/write_main_memory_datas 求和（单次发射确认后）
2. 禁止用"指令数相同"推断"重复执行"——判别器是数据地址（instr detail 的 XN 寄存器值）
3. 未知数显式列出，禁止调容差区间去盖实测值（2026-10-26 RMW 拟合事故）
4. IR 读点清单（nd2nz/load_gm_to_ubuf/set_l1_2d 按 mix_aic/mix_aiv 分节）是通道期望值的正确来源；源码层会漏编译器插入的协作读写
