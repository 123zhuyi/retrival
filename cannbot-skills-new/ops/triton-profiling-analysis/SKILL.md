---
name: triton-profiling-analysis
description: >
  Triton-Ascend 算子性能分析技能（分析层，不采集、不改代码）。性能数据一律调用
  triton-simulator 获取：上板 msprof CSV（秒级，判 bound）、仿真 trace.json（指令级，定位
  pipe 气泡成因）与 IR ttir/ttadapter/last_pass.mlir（性能采集后强制采集）。分析方法论：
  上板数据定位当前 bound（优化性价比最高的 pipe），按 bound 类型给优化建议；
  SCALAR bound 必须先做 IR 标量降级分析（指针张量 gather / 逐元素算术 / cumsum /
  常量未标记）后才能给建议；所有 pipe 占比均 <50%（无 bound）时，
  用仿真数据定位 pipe 间气泡成因（掩盖失效 / WAIT_FLAG 等待），消除气泡使某 pipe
  占比超过 50% 达成 bound；气泡消除尝试 10 次仍未达成则取占比靠前的 pipe 优化或
  重排流水。理想终止条件：某 pipe 达到 80% 理论性能。优化落地交主 agent /
  triton-latency-optimizer。
  入口（skills 链传导，编排层文件零改动）：Phase 4 归因唯一来源——
  ① latency-optimizer 主流程步骤 0 的归因前置（每轮优化前调用，无指纹匹配报告时）；
  ② simulator-optimizer 被调用时整体转交（仿真深钻分支）。
  latency-optimizer 只消费本 skill 的归因报告做优化，不自行归因。
  触发：Phase 4 每轮归因、拿到 kernel 需要定位性能瓶颈、给出优化方向时；
  每轮优化后需要重新采集分析、判断 bound 是否转移时。
---

# Triton 算子性能分析（上板判 bound + 仿真消气泡）

## 方法论总览（Profiling 数据使用逻辑）

两路数据各回答一个问题，循环使用：

| 数据路 | 回答的问题 | 用途 | 理想终止条件 |
|---|---|---|---|
| **上板数据**（msprof CSV） | 当前 bound 在哪个 pipe？ | 定位优化性价比最高的 pipe，按 bound 类型给优化建议 | 某 pipe 达到 **80% 理论性能**（算子用时 ≈ 80% × 数据量/理论带宽，或 数据量/理论算力） |
| **仿真数据**（trace.json） | pipe 里的气泡从哪来？ | 定位气泡成因，消除气泡让某 pipe 占比超 **50%** 达成 bound | 所有管线实现掩盖，某 pipe 形成绝对 bound（占比超 **80%**） |

优化循环：**基于仿真结果**开启双流水、降低空泡率 → **基于上板结果**优化计算
bound → 优化后 bound 发生变化 → 重新上板判定 → 往复，直至终止条件。

## 角色分工

| 角色 | 职责 |
|---|---|
| `triton-simulator`（上游） | **唯一数据源**：上板采集（board 模式 `msprof op`，产出 msprof CSV）与仿真采集（标准流程，产出 instr_exe/code_exe CSV + trace.json + visualize_data.bin）。典型 case 选择、采集策略、产物完整性判据一律由它负责，本 skill 不复制其命令细节 |
| **本 skill** | 消费 triton-simulator 产物：bound 判定 + 气泡分析 + 优化建议 + 循环控制 |
| 主 agent / `triton-latency-optimizer` | 按本 skill 建议落地代码优化（本 skill 对被测代码只读） |

## 工作流程（强制顺序）

```
1. 输入判定与数据盘点
2. 上板采集（委托 triton-simulator）→ bound 判定          ← 每轮循环的入口，必须先上板
3. IR 采集（性能采集后强制）                                ← 委托 triton-latency-optimizer 的
   run_and_extract.sh 提取 ttir/ttadapter/last_pass.mlir；与性能数据同一代码状态
4. 分支：
   A. 达成 bound
      - SCALAR bound → 强制 IR 标量降级分析（见 3A.1），未完成禁止给建议
      - 其他 pipe    → 输出 bound pipe 清单 + 各自问题 + 优化建议
   B. 无 bound     → 仿真采集（委托 triton-simulator）→ 气泡分析 → 消气泡建议
5. 循环控制（10 次规则 / bound 转移重测 / 流水重排；每轮代码改动后 IR 必须重采）
6. 输出报告
```

### 1. 输入判定与数据盘点

明确分析对象（kernel / 实现文件 / 算子工作目录）、分析目标（bound 定位还是气泡
下钻）、当前循环轮次；盘点是否已有可复用的上板 CSV / 仿真数据包——**有则直接
复用进入分析，禁止重复采集**。注意跨轮数据可比性：代码已改动（impl 指纹变化）则
旧数据不可用于判定新 bound，必须重采。

### 1.6 计算草图推导（agent 判断，非模板——强制替代内置草图）

- 管线 §2.5（pipe_time_model.py）**不含任何算子模板**：内置的 r39 五段草图与
  GDN 融合草图已删除（2026-10-26 事故：r39 草图对 201_GDN 静默产错值——
  草图绑定实现而非算子，对任何新实现必然张冠李戴）。
- **推导主体是 agent（本 skill 的执行者），不是代码；推导必须三方锚定，
  禁止单源臆断**（2026-10-26 教训：未查设计草图导致 k/w 搬运少算 V/BV 倍）：
  1. **设计意图** `{workdir}/sketch.txt`（Phase 2 产物）——grid 分解、tile 常量、
     架构决策（如 v-dim-program-split 决定 k/w 搬运含 V/BV 复制因子）
  2. **实现现实** `ir_dump/*_ttir.mlir`——tt.load/tt.store 的真实形状与循环内外
     位置；管线自动做 IR 对账（programs×NT×逐迭代字节 vs 草图 gm_in，偏差
     >25% 标 warning 人工复核）
  3. **任务下限** 输入字节和（机械）
  完成后写 `{workdir}/kernel_sketch.json`：
  - `code_md5_prefix`：最终代码 md5 前 10 位（管线做指纹闸，不匹配即拒）
  - `constants`：kernel 常量（BT/BV 等，标注来源行）
  - `aliases`：语义符号 → 白名单表达式（如 `NT = nseg*cdiv(in0_d2,BT)`），
    机械符号表只有 `in{i}_d{k}` / `in{i}_bytes` / `nseg` / `total_len`
  - `kernels.<名>.exprs`：`c_flop / v_flop / gm_in / gm_out / fixp_bytes`
    五条表达式（对任意 shape 泛化）
  - `design_refs` / `ir_refs`：每条判断对应的设计草图行号与 ttir 行号引用
    （**必须真实行号，禁止"待补"占位——占位草图按闸门失败拒收**）
  - `redundancy_notes`：**无效搬运/无效计算判断**（对照任务下限逐 pipe 说明）
- **三道机械闸（管线自动执行，agent 不可绕过）**：表达式白名单
  （数字/符号/四则/cdiv，其余语法拒绝）；`gm_in ≥ 任务下限`（实现不可能
  读得比输入还少——草图低估即拒收并回落下限口径）；代码指纹 md5 匹配。
- 三级口径语义（禁止混淆）：任务下限 = "任何正确实现最少读多少"（真下限）；
  草图量 = "该实现声称搬多少"（冗余度 = 草图/下限）；实测 = "实际执行多少"。
  草图缺失时报告只出下限 + 实测，并标注「实现量与冗余度未知」，禁止臆造。

### 2. 上板采集与 bound 判定

- **采集**：委托 `triton-simulator` 的 board 模式（真实 NPU，秒级）。要求
  `--aic-metrics` 覆盖 `PipeUtilization,ResourceConflictRatio,L2Cache`（bound
  判定三张表的字段来源）。命令与环境准备查 triton-simulator 文档，本 skill 不复制。
- **判定**：按 `references/board-bound-analysis.md` 的判定表逐条核对，产出：
  - 各 pipe 占比表（Task Duration / vec / cube / mte1 / mte2 / mte3 / scalar）
  - 命中的附带问题（核间负载不均衡 / Bank Conflict / L2 命中率低）
  - **bound 结论**：占比最高且 >50% 的 pipe（SCALAR 特例 >30%）= 达成 bound；
    全部 <50% = 无 bound
- 上板数据按 kernel 逐行区分：框架辅助 kernel（`.contiguous()` 等触发的拷贝）是
  端到端真实开销，单独归因到实现层，不混入算子 kernel 指标，也不当噪声过滤。

### 3A. 达成 bound：输出优化建议

对每个达成 bound 的 pipe，按 `references/board-bound-analysis.md` 判定表的解决
方案列给出优化建议（VEC→UB 融合、CUBE→L0C 累加、MTE→访存连续化/增大搬运量、
SCALAR→防标量降级/常量化等）。输出「达成 bound 的 pipe 列表 + 该 pipe 的问题
（利用率低等）+ 建议」，交主 agent 落地。

### 3A.1 SCALAR bound 强制下钻：IR 标量降级分析（未完成禁止给建议）

上板判定为 SCALAR bound 时，必须在给建议前完成以下分析（IR 来自步骤 3 的采集；
若未采集，先补采且与性能数据同一代码状态，禁止复用历史 IR 快照）：

1. **热行清单**：仿真 code_exe 周期降序 top 行；每行与 kernel 源码区间核对——
   **越界行（跨 kernel 区间 / host 侧行）一律隔离，不作优化靶点**
   （行标签错位或编译器共享代码合并会产生假热点；实测案例：prep 采集中出现
   scan 区间行标签且周期 7000× 于该 kernel 自身执行值）。
2. **逐行 IR 形态核对**（ttir 层），归类标量降级形态——**脚本化**：
   `scripts/ir_loc_resolver.py <kernel_ttir.mlir>` 自动把每条 MLIR 操作的
   `#loc` 锚点解析回**源文件:行:列 + 源码语句**，并按下列形态归类汇总
   （含各自命中的源码行清单，可直接贴入 §4.5 报告）：
   - 指针张量 gather：`arith.muli(stride)` → `tt.addptr` 链 →
     `tt.load tensor<Nx!tt.ptr>`——1D strided/逐元素寻址，机器级落为标量 LD 序列
     （每条访存在途可达 ~30 cyc）；
   - 逐元素 i32 比较/算术（minimum/where 掩码、clamp）；
   - 一维 `tl.cumsum` / 标量归约；
   - 输入常量未标记 constexpr（运行期标量）。
3. **占用成因**以仿真 trace 交叉定位（热行有无标量访存指令在飞 / 等待边指向谁），
   修复方向按成因定：向量化（算术）、block_ptr/连续化（访存）、消同步（等待）。
   trace 侧定位用 `scripts/blocking_point_analysis.py <trace.json>`（v3）：
   阻塞点输出**源文件:行 + 源码语句**，含「按等待行（哪一行在等）」与
   「执行→等待 联合定位」两张表（口径 6b/6c 的脚本化）。
4. **产出**：热行清单 + IR 操作引用（`#loc` 行号 → mlir 指令 → **源码行**，
   由 `ir_loc_resolver.py` 生成）+ 降级形态归类，交 3A 的建议输出与主 agent 落地。

### 3B. 无 bound：仿真采集与气泡分析

- **采集**：委托 `triton-simulator` 标准流程（dispatch 探测 → 典型 case 中位
  选择 → 逐 kernel 采集 → sim_index.json）。只消费 verdict=complete 的数据包
  （四类产物齐全），部分采集的偏置占比不得流入诊断。
- **分析**：按 `references/simulator-bubble-analysis.md` 解析 trace.json——
  筛选 `ph="X"` 事件得到各 pipe 活跃/空泡区间，计算 pipe 对重叠度（<10% 命中
  掩盖失效），筛选 WAIT_FLAG 事件定位空泡在等谁。一个完整计算核心（1C+2V）共
  18 个 pipe。
- **分析工具**：`scripts/pipe_overlap_analysis.py <trace.json>`（单子核）或
  `--dir <OPPROF_*/simulator>`（全子核汇总）——产出 pipe 活跃表 / 掩盖矩阵
  （重叠度=交集/并集，<10% 命中失效）/ WAIT_FLAG 归因三张表，口径见
  `references/simulator-bubble-analysis.md`。
- **输出**（三个粒度，缺一不可）：
  - 每 pipe：空泡率 / 空泡原因 / 是否可优化
  - 每核心：idle 空泡占比 / 空泡原因 / 是否可优化
  - **阻塞点 → 代码位置**：WAIT_FLAG/BAR 按 `args.code` 聚合（哪一行在等谁、
    等了多久；DEVI 与内部标签不参与，越界行隔离）——见 references 口径 6b
  - **全部根因 → 代码位置（强制）**：等待边 / 阻塞点（独占执行期，归因执行方行）/
    掩盖失效对（双方主导行）/ 空泡与 Top 气泡，一律落到 源文件:行；DEVI 与
    编译器插入项显式标注「无用户行」并给 pc_addr/循环体定位——references 口径 6c
- 消气泡建议优先级：使能双流水（掩盖搬运与计算）→ 增加并行度 → 消除等待
  （数据阻塞 / 标量阻塞 / 跨核阻塞，按 WAIT_FLAG 归因）。

### 4. 循环控制

- **每轮优化后必须重新上板**：优化会改变各 pipe 占比（bound 转移），旧 bound
  结论即失效。
- **10 次规则**：无 bound 路径（3B）尝试 10 次后仍未达成阈值（某 pipe 占比超
  50%），不再继续消气泡，转入：取当前**占比靠前**的 pipe 按 bound 路径优化；
  或尝试**重排流水**（按 `references/pipeline-rearrangement.md`，从计算草图
  依赖关系重排）。
- 达成 bound（≥50%）→ 走 3A；某 pipe 接近 80% 理论性能 → 达理想终止条件，
  建议主 agent 收敛。

### 5. 输出报告（固定结构）

```markdown
# Triton 算子性能分析报告：<op_name / kernel_name>

## 1. 分析对象与环境
kernel / 实现文件 / 典型 case / SOC_VERSION / CANN 版本 / 分析目标 / 当前循环轮次

## 2. 数据采集清单
| 数据路 | 来源（triton-simulator 模式） | 产物路径 | 完整性 |

## 3. 上板 bound 判定
各 pipe 占比表 + 命中的附带问题（核间不均衡 / Bank Conflict / L2）+ bound 结论
（达成：pipe 名 + 占比；未达成：最高占比 pipe）

## 4. 气泡分析（仅无 bound 时）
18 pipe 空泡率表（pipe / 空泡率 / 原因 / 可优化）+ 每核心 idle 统计 +
Top3 气泡（类型 / 时长 / 根因 / WAIT_FLAG 证据）

## 4.5 IR 分析（SCALAR bound 时强制，见 3A.1）
IR 产物路径（ttir/ttadapter/last_pass.mlir）+ 热行清单 + 标量降级形态
（IR 操作引用）+ 越界行隔离记录

## 5. 优化建议（按优先级）
| # | 建议 | 判定依据（字段=值 @ 产物路径） | 预期方向 |

## 6. 循环状态
当前轮次 / 是否触发 10 次规则 / 下一轮建议路径（消气泡 or 优化 bound pipe or 重排流水）

## 7. 证据附录
引用的 CSV 字段原始值 / trace 事件片段 / 数据缺口说明
```

### 5.1 产物归档（随迭代，强制，skill 自包含——不依赖编排层文件）

归档由**本 skill 在流程末尾自行完成**（采集→分析→报告→归档→自检一体），无需编排层
agent 文件做任何配合；开关 = 是否调用本 skill。性能结果与报告归入当前迭代目录：

```
<workdir>/output/iter_N/            （或 opt_iter_N）
├── code.py / optimized_code.py     # 本轮代码（含指纹）
├── verify/verify_result.json       # 精度结果（passed/total）
├── verify/perf_result.json         # benchmark 结果（geomean/逐 case）
├── log.md                          # 改动/判定/归因/指纹（AGENTS.md 约定）
└── profiling/                      # ★ 本轮 profiling 归档（自包含）
    ├── *_report.md                 #   skill §5 结构分析报告；与采集时代码指纹匹配
    └── collections/<采集名>/        #   原始采集数据（board/sim OPPROF），按采集归属的
                                     #   代码状态定位迭代；只增不删，重采写新目录名
```

规则：
1. 每轮迭代的 verify/benchmark JSON 落在 `iter_N/verify/`（benchmark 前置闸门同源）；
2. 本轮有新采集的分析报告**复制**（非链接）进 `iter_N/profiling/`，报告 §1 须标注代码状态；
3. 报告的代码状态与迭代代码指纹不匹配时禁止归档（历史数据引用在报告内注明路径即可）；
4. 跨迭代总控（总表/总报告）放工作目录顶层，不替代逐迭代归档；原始采集数据随
   `profiling/collections/` 归入采集所属迭代（禁止顶层散放采集目录）；采集驱动等工作
   目录级脚本放 `tools/`；
6. **归档自检（skill 内置闸门）**：流程末尾运行
   `python3 <skill_dir>/scripts/check_archive.py <算子工作目录>`，退出码与错误清单
   写入报告 §6 循环状态——FAIL 时本 skill 的返回明确标注「归档未通过」，调用方
   （主 agent）据此决定是否阻断下一轮；verify/perf 基础文件沿用工作目录既有约定，
   无本轮 profiling 时 manifest 记 `"profiling_flow": false`。
5. 每轮迭代写 `iter_N/manifest.json`（机读指纹链，本 skill 校验器的对象）：

```json
{
  "iter": "opt_iter_N",
  "code": {"file": "optimized_code.py", "md5": "<10位>"},
  "verify": "verify/verify_result.json",
  "benchmark": "verify/perf_result.json",
  "geomean": 1.2801,
  "verdict": "adopted|rejected",
  "profiling_reports": [
    {"file": "profiling/final_board_case6_report.md", "code_state_md5": "<采集时代码指纹>"}
  ],
  "collections": [
    {"kind": "board|simulator", "case": 6, "path": "../../prof/final_board_case6/"}
  ]
}
```

## 关键行为认知

1. **先上板后仿真**：上板秒级、直接回答 bound 在哪；仿真是指令级、慢而贵，
   只回答"为什么有空泡"。跳过上板直接仿真是流程错误。
1b. **性能采集后必须采 IR，且同代码状态**：IR 是 SCALAR bound 下钻与热行归因的
   唯一指令级依据；代码改动后 IR 必须重采（禁止复用历史快照），否则归因错位
   （实测案例：code_exe 行标签跨 kernel 错位，需 IR loc 交叉验证才能识破）。
1c. **code_exe 热行必须与源码区间核对**：越界行（跨 kernel / host 侧）一律隔离，
   行标签错位与共享代码合并会产生假热点，直接采信会导致错误优化立项。
2. **无 bound ≠ 无问题**：所有 pipe 占比均 <50% 说明空泡吞掉了执行时间，
   出路只有消气泡（双流水 / 并行度），此时优化单个 pipe 收益有限。
3. **bound 会转移**：每轮优化后各 pipe 占比重新洗牌，必须重新上板判定，
   禁止沿用旧结论指导新优化。
4. **方向比幅度重要**：优化收益是 shape 的函数；方向错了说明诊断错了，
   方向对但幅度小可能只是测试 shape 不代表真实负载。
5. **数据可比性**：跨轮对比前确认 impl/task 指纹一致（sim_index.json 的
   comparability）；代码改过，旧仿真数据不能用于新一轮气泡归因。

## 边界与禁止

- 禁止无采集证据下结论（含"硬件极限 / 不可优化"类判断）；数据缺口在报告
  「数据采集清单」与「证据附录」显式标注。
- 禁止绕过 triton-simulator 自行采集或在本 skill 维护采集脚本（知识真源在上游）；
  分析脚本（pipe_overlap_analysis.py）只消费已采集产物，不属采集脚本。
- 禁止消费不完整仿真产物（verdict=fail / 部分 pipe 缺数据）做气泡归因。
- 本 skill 对被测代码**只读**；优化建议交主 agent / triton-latency-optimizer 落地。
- 禁止静默丢弃与预期不符的判定结果；判定表未命中的异常模式如实上报。

## 脚本工具

| 脚本 | 用途 |
|---|---|
| `scripts/detect_soc.py`
| `scripts/chip_spec.py [--soc X] [--json]` | **按 SOC 查各 pipe 带宽/算力**：缺省自动探测当前卡，--soc 可查任意变体；（$ASCEND_HOME_PATH/{arm64,aarch64}-linux/data/platform_config/<soc>.ini，SOC 由 detect_soc 自动探测）→ 核数/频率/UB/L1/L0/L2/HBM 容量 + 推导 cube 稠密峰值（2×m×n×k×核数×频率，如 9372=294.9 TFLOPS bf16）；给 roofline/MFU 提供本机真实规格分母；pipe 视角表（MTE1/MTE2/MTE3/FIXP 带宽 + CUBE/VECTOR 算力），跨 GM 边界 pipe 取瓶颈端（HBM 锚定实测） | | **探测当前机器 SOC 型号**（torch_npu get_device_name → 驱动权威值，如 Ascend910_9372）；采集/分析脚本缺省自动调用，失败宁可报错不猜测 |
| `scripts/run_full_analysis.py --impl <generated.py> --task <task.py> --workdir <算子工作目录> [--shrink 256]` | **端到端流水线（全流程脚本化复现）**：board 逐 case 上板采集+解析 → bound 自动判定（判定表代码化）→ kernel 枚举（硬门禁，未触发入状态表）→ 策略B 收缩复验闸门（发射路径不一致即中止）→ 逐 kernel 仿真采集（collect.sh，并行受控）→ pipe 四口径 + 阻塞点 + **bound pipe→Top3 代码行 证据链** → `roundsN/report.md` 自动生成；产物自包含于 `simulator/roundsN/`（N 自动递增不覆盖），同参数重跑即复现 |
| `scripts/dual_buffer_vars.py <dump1.full> [dump2 ...] [--src <impl.py>] [--json out]` | **双流水源码变量统计**：按源码变量（loc 行号→左值名）聚合 pointer_cast 地址，硬判据 **≥2 个互不相同地址**——双参数 pointer_cast(offA≠offB)=编译器乒乓对（最强），跨 op 多地址次之，双参数同址/单地址明确不成立；输入为 --mlir-print-debuginfo 全量 dump（run_full_analysis 的 <kernel>_last_pass_mb.mlir.full） |
| `scripts/diff_pass.sh <kernel.ttadapter> <pass名> [额外flags]` | **pass 级变换验证**（合并自 triton-ir-analysis）：--mlir-print-ir-before/after=<pass> 重放 bishengir-compile，前后 IR 字节级 diff + 结构计数（scf.for/mmad/set_flag/wait_flag）→ "转/拒转"——判定某 pass（如 cv-pipelining / hivm-enable-multi-buffer）是否真正变换；BISHENGIR_COMPILE 环境可覆盖编译器路径；pass 名以 `--mlir-print-ir-after-all` 全量 dump 为准；比较前滤除编译器警告行（曾致假阳性） |
| `scripts/find_variants.py <kernel名> [cache根]` | **cache 编译变体辨认**（合并自 triton-ir-analysis）：constexpr 常量 + tensor 形状 + json options + mtime 列出某 kernel 全部缓存变体，辅助确认"哪个缓存目录对应哪个配置"（严禁凭目录名猜） |
| `scripts/dual_pipeline_check.py --dir <OPPROF_*/simulator> [--ir-dir <.triton_cache>] [--kernel <name>] [--json out.json]` | **双流水生效性判定**：IR 意图（ttadapter/ttir 的 num_stages>1 循环）× trace 实效（装载×计算掩盖对重叠度 C_MTE2_MTE1/C_MTE1_MAC/C_MTE2_MAC/V_MTE2_VEC/V_VEC_MTE3 + pipe 在途并发 raw/merged>1=多缓冲在飞）交叉 → 生效/部分/未生效 + 综合判定（如"流水化已编译但运行时未重叠"）；已接入 run_full_analysis 报告章节 |
| `scripts/pipe_overlap_analysis.py <trace.json>` | 单子核 pipe 掩盖度分析：pipe 活跃三口径（队列占用/空等/有效计算）+ 掩盖矩阵（重叠度=交集/并集，<10% 命中失效）+ WAIT_FLAG 归因 |
| `scripts/pipe_overlap_analysis.py --dir <OPPROF_*/simulator> [--json out.json]` | 自适应聚焦第一个核心的 1C+2V（子核/pipe 构成随算子类型变化，按 core0.* 实际存在者选取）+ 跨子核汇总（中位数 [min~max]），机器可读结果落盘 |
| `scripts/blocking_point_analysis.py <trace.json> [--pairs] [--no-src]` | 阻塞点定位 v4（**严格口径**：1 pipe 独占执行 + 本子核其余主要工作 pipe（cube:CUBE/MTE1/MTE2/FIXP，vec:VECTOR/MTE2/MTE3）全等待 + 相邻切片合并为阻塞事件后**周期重复≥3次且间隔均匀**；一次性/启动期阻塞不计）——输出**源文件:行 + 源码语句**：按执行行 / 按等待行（哪一行在等）/ 执行→等待 联合定位三表；`--pairs` 掩盖失效对 + 双方主导代码行归因；internal 帧与 ⚠外部行按口径 6c 隔离标记 |
| `scripts/ir_loc_resolver.py <kernel.mlir> [--op tt.load] [--gather-only]` | **IR 源码行定位（3A.1 步骤 2 的脚本化）**：#loc 锚点（含 callsite/fused 链）→ 源文件:行:列 + 源码语句；标量降级形态归类汇总（指针张量 gather / splat-addptr 构造 / 逐元素算术 / cumsum，各附源码行清单） |
| `scripts/check_archive.py <工作目录>` | 归档规范校验（§5.1 自检闸门：指纹链/报告结构/collections 存在性） |
| `triton-latency-optimizer/scripts/run_and_extract.sh <driver.py>` | **IR 采集（步骤 3，委托调用）**：提取 ttir / ttadapter / last_pass.mlir；环境与用法查该 skill 的 `references/IR_triton.md`，本 skill 不复制其命令细节 |

（采集类脚本一律通过 `triton-simulator` 调用，见其 SKILL.md「脚本工具」；IR 采集通过
`triton-latency-optimizer` 调用。）

## 参考资料

| 文件 | 内容 | 对应方法论 | 使用时机 |
|---|---|---|---|
| `references/board-bound-analysis.md` | 上板 bound 判定表（问题/判定条件/子问题/解决方案）+ 80% 终止条件 + 字段速查 | 上板数据分析（每轮入口） | 步骤 2 / 3A |
| `references/dual-pipeline-judgment.md` | **双流水判定口径固化**（判定时无需再查文档/源码）：三维度定义（c内/v内/cv，≥30%线）+ mark 决策树五条件（bishengir 源码级）+ 双地址硬判据 + 枝5空间组关闭识别 + 未双缓冲原因分类 + pass diff 滤警告坑 + pc 级乒乓严格口径 + 证据优先级与局限 | 双流水分析（§4） | 判定双流水/解释未双缓冲变量 |
| `references/ir_dump.md` | IR 提取三途径（cache/TRITON_DEBUG dump/手动重放）+ 变体辨认 + 重放参数与全部实测坑 | IR 层取证 | 双流水/pass 判定、IR 提取 |
| `references/ir_reading.md` | 逐层 IR 读法（ttir→ttadapter→HIVM/npuir）、关键 op 清单、CV 流水生效判定签名 | IR 层取证 | IR 结构解读 |
| `references/profiling.md` | profiling 结果解读（simulator/board）+ 与 IR 交叉验证方法 | IR×profiling 交叉 | 证据互洽核对 |
| `references/simulator-bubble-analysis.md` | trace.json 解析方法（ph=X 区间、重叠度、WAIT_FLAG）+ 气泡定位表 + 18 pipe 口径 | 仿真数据分析 | 步骤 3B |
| `references/pipeline-rearrangement.md` | 流水重排方法（计算块分类 → 依赖图 → 排布）+ 触发时机 | 10 次规则的兜底路径 | 步骤 4 |
