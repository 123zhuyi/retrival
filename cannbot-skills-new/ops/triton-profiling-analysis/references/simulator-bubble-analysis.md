# 仿真数据分析：pipe 气泡定位

> 来源：profiling 方案（用户提供）。仿真数据用于定位 pipe 中气泡出现的原因，
> 尽可能消除气泡，让某个 pipe 达成 bound 条件（占比超过 50%）；理想终止是所有
> 管线实现掩盖、某 pipe 形成绝对 bound（占比超过 80%）。

数据输入：`triton-simulator` 仿真产物 `trace.json`（Chrome Trace Format，位于
`sim__<kernel>/` OPPROF 目录）。只消费 verdict=complete 的数据包（两张 CSV 非空 +
trace.json + visualize_data.bin 齐全）——部分仿真的占比代表已仿真前段而非稳态，
偏置数据不得流入诊断。

## trace.json 结构与分析口径

```json
{
  "traceEvents": [{
    "ph": "X", "ts": 3.568, "dur": 0.001,
    "pid": "core2.veccore1",     // 计算核.子核（coreN.cubecoreN / coreN.veccoreN）
    "tid": "SCALAR",             // 指令队列（PIPE）：SCALAR / MTE2 / MTE3 / VECTOR / MAC / FIXPIPE / ...
    "name": "MOV_XD_IMM",        // 指令名
    "args": { "code": "/path/kernel.cpp:22", "pc_addr": "0x10d0d000" }
  }]
}
```

- **pipe = tid**：一个完整计算核心（**1C+2V**：1 个 Cube 子核 + 2 个 Vector 子核）
  共计 **18 个 pipe**，每个有唯一 ID 区分。pipe 清单从 trace 的 metadata 事件
  （`ph="M"`, `name="process_name"` / `thread_name"`）**动态发现**，禁止硬编码。
- **活跃区间**：筛 `ph="X"` 的事件，每个事件即一段指令执行区间 `[ts, ts+dur]`。
- **空泡**：同一 pipe 相邻事件之间的 idle 间隙；空泡率 = 1 − Σ活跃时长 / 核运行时长。
- **WAIT_FLAG 事件**：trace 中的等待标志事件，指明空泡期间在等哪个 pipe/资源——
  这是空泡**成因**归因的直接证据。

## 解析步骤（用 `scripts/pipe_overlap_analysis.py` 执行）

脚本一键完成下述第 1-5 步并输出三张表（pipe 活跃 / 掩盖矩阵 / WAIT_FLAG 归因）：

```bash
# 单个子核 trace（快速定位）
python3 {skill_path}/scripts/pipe_overlap_analysis.py <OPPROF_*/simulator/coreN.veccoreM/trace.json>

# 整目录自适应分析：只聚焦第一个核心的 1C+2V（vector 型无 cubecore、cube 型无
# veccore、mix 型全有——按 core0.* 实际存在者自适应选取，pipe 个数动态发现）
python3 {skill_path}/scripts/pipe_overlap_analysis.py --dir <OPPROF_*/simulator> \
    --json overlap_result.json
# 可选：--threshold 10（掩盖失效阈值%）、--min-activity 1（噪声 pipe 过滤）、--top 8
# 人工核对：打印 pipe 的原始/归并活跃区间与源事件原文（对照 trace.json 验证用）
python3 {skill_path}/scripts/pipe_overlap_analysis.py <trace.json> \
    --show-intervals MTE2 VECTOR --max-show 10
# 完整区间数据导出（raw_intervals 与 ph=X 事件一一对应，可逐条对照 trace.json）
python3 {skill_path}/scripts/pipe_overlap_analysis.py <trace.json> \
    --export-intervals intervals.json
# pipe 间依赖图：flow(s/t 按 id 配对) × WAIT_FLAG × SET_FLAG 三源交叉 + 传播时延
python3 {skill_path}/scripts/pipe_overlap_analysis.py --dir <OPPROF_*/simulator> --deps
```

**依赖证据（trace.json 内三份显式编码，语义已实测标定）**：
- flow 事件（`ph="s"/"t"`，按 `id` 配对）：`cat="AToB"` 表示 **A 依赖 B**（A 等 B 置
  flag；s 事件在等待方 A、t 事件在触发方 B），两端 ts 差 = 依赖传播时延；
- WAIT_FLAG 指令（等待方视角）：`args.detail = "PIPE:A,TRIGGERPIPE:B,FLAGID:n"`；
- SET_FLAG 指令（置位方视角）：detail 同编码。
三源对同一条边的计数应一致（实测 recur：VECTOR 依赖 SCALAR 边 flow/WAIT/SET
均 892）——不一致时以 detail 为准并核查解析。

脚本内部口径（修改脚本前必读）：

1. 加载 trace.json，从 metadata（`ph="M"` 的 `process_name`）建 pipe 映射并枚举全部
   pipe——**动态发现，禁止硬编码**；两种结构自动识别（子核文件 pid=数字 pipe ID /
   合并文件 pid="coreN.子核N"+tid=pipe 名；掩盖度是子核内 pipe 对的属性，优先用子核文件）。
2. 逐 pipe 收集 `ph="X"` 事件 → 活跃区间序列；**pipe 内并发区间先归并为互不相交集合**
   （否则交集按在途份数重复计数）；原始忙碌和 >100% span = pipe 内在途并发，
   即双缓冲已生效的证据（与 pipe 间掩盖度是两个独立指标）。
3. **pipe 活跃四口径**（`ph="X"` 是事件类型过滤，剔不掉空等——WAIT_FLAG/BAR 本身
   就是占队列的 X 事件，须按 `name` 剔）：`merged%` 物理队列占用（含等待/汇合）→
   空泡率 = 100 − merged%；`wait%` 其中 WAIT_FLAG 空等占比（**逻辑空泡**）；`bar%`
   其中 BAR barrier 汇合占比（同步，非计算）；`eff%` 剔除 WAIT_FLAG 与 BAR 的有效
   计算。屏蔽集 `SYNC_BLOCK_INSTR = {WAIT_FLAG, BAR}`（SET_FLAG 为置位方、1 cycle，
   不屏蔽）。实测示例（recur core0.veccore0）：VECTOR 队列占用 96.2% 但 wait 49.9% →
   有效仅 46.3%——「队列假忙」会把空等误判成计算，诊断递推类 kernel 必看 eff%。
4. **活跃区间与掩盖矩阵（v3 口径，强制）**：**活跃区间 = pipe 全部 `ph="X"` 事件
   剔除等待类指令（`SYNC_BLOCK_INSTR = {WAIT_FLAG, BAR}`）后的归并区间**；
   掩盖矩阵的重叠度与空泡归因的覆盖率**一律基于活跃区间**——等待指令占队列但
   非工作，用队列占用口径会把「双方同时在等」误计为掩盖（实测 recur：
   MTE2×VECTOR 占用口径 53.93% vs 活跃口径 6.75%——搬运躲在了 VECTOR 的等待
   时间里，与有效计算实际串行）。重叠度 = **max(交集/活跃A, 交集/活跃B)**（=
   交集/min(活跃A,活跃B)，即「用时较少一方的活跃被对方覆盖的比例」）——分母
   不用并集：并集会惩罚用时少的 pipe（短 pipe 即便被完全覆盖 Jaccard 也低）。
   **<10% 命中掩盖失效**；活跃 <1% span 的噪声队列不进矩阵。注意：PPT 六对以外
   的 pipe 对（如 SCALAR×SCALARLDST，同一标量单元的两队列）低重叠属结构必然，
   不构成掩盖失效结论。
5. **每 pipe 空泡率与成因（默认输出第 3 段，自动合成）**：空泡(P) = 活跃区间(P)
   的补集 = 物理空闲(100−队列占用) + 等待驻留(WAIT_FLAG 记录在 P 队列的部分)；
   覆盖率 (Q,空泡) = |活跃(Q) ∩ 空泡| / |空泡|（只算他人真干活）。成因合成规则
   （按优先级）：「P 等谁」（detail 的 PIPE:=P，等待方视角，>15% span 判依赖链
   主导）→「谁等 P」（TRIGGERPIPE:=P，**被依赖方=瓶颈供给**）→ 物理空闲为主
   （附空泡期间活跃 top，判节拍/碎片）→ 近饱和(<5%)。⚠ detail 缩写须归一
   （VEC→VECTOR，PIPE_ALIAS），否则等待边接不到 pipe 上。
6. **WAIT_FLAG 归因**以事件 `args.detail`（`PIPE:x,TRIGGERPIPE:y,FLAGID:z`）为准——
   事件自身 pid 与等待方/触发方不保证一致，**禁止按 pid 归因**。
6b. **阻塞点 → 代码位置归因（重点输出，缺一不可）**：WAIT_FLAG / BAR 事件的
   `args.code` 携带源码位置（`/path/kernel.py:LINE`）——按（code 行, 边）聚合
   Σdur 与次数，产出「哪一行代码在等谁、等了多久」的阻塞点清单，与空泡率表
   互相定位。**脚本化**：`scripts/blocking_point_analysis.py <trace>` v3 直接产出
   三张带 **源文件:行 + 源码语句** 的表——按执行行（独占执行期归因到执行方指令行）、
   按等待行（哪一行在等）、执行→等待 联合定位（依赖链可视化）。注意：
   - `args.code` 可能多行（`\n` 分隔），取用户源文件行、弃 `internal:0`；
     脚本取首个非 internal 帧为主帧，其余计为内联帧数展示；
   - **WAIT_FLAG_DEVI 的 code 是编译器内部标签**（常指向 host 侧函数行），
     不参与代码归因（DEVI 只有边级统计）；脚本统一标记为 `internal`；
   - 行标签存在错位噪声（跨 kernel 区间 / host 行），与源码区间核对，
     越界行一律隔离（同 SKILL.md 认知 1c）；脚本以事件占比最高文件为
     主源文件，非主源文件的行自动标 `⚠外部`。
6c. **根因 → 代码位置强制规则（所有 trace.json 根因结论缺一不可）**：从 trace.json
   得出的每一类根因都必须落到代码位置（`args.code` 的 源文件:行），否则结论不得进入报告：
   - 等待边（WAIT_FLAG/BAR）→ 等待指令自身的 code 行（口径 6b）；
   - 阻塞点（独占执行期：1 pipe 执行 + 其余等待）→ **执行方**当前指令的 code 行
     （`scripts/blocking_point_analysis.py <trace>`）；
   - 掩盖失效对（重叠度 <10%）→ **双方各自的主导代码行**（活跃区间按 code 聚合
     Σdur top，`blocking_point_analysis.py <trace> --pairs`；噪声管道活跃 <1% span 不进矩阵）；
   - 空泡 / Top 气泡 → 归因到该 pipe 等待边的 code 行（等待类）或物理空闲期间的
     他方执行行（节拍类）；
   - DEVI / `internal:0` / host 行标签 → **显式标注「编译器插入，无用户行」**，
     以 `pc_addr` 或所在循环体代定位，禁止硬配一个用户行充数。
7. 输出两个粒度的结论（见下表 + 报告格式）。

> V_C 掩盖（整体 C 流水 × V 流水）需跨子核聚合 C/V 侧 pipe，脚本不自动判定，
> 由分析者基于各子核结果聚合。实测样例见 `profiling` 报告 §4（ChunkGatedDeltaRule
> opt_iter_4 活跃口径：MTE2×VECTOR 6.75%★、MTE3×VECTOR 46.57%、VEC←SCALAR 等待 57% span）。

## 气泡问题定位表

| 气泡问题定位 | 判定条件 |
|---|---|
| C_MTE2_MTE1 掩盖失效 | 计算两个 pipe 的重叠度，重叠度 <10% 命中 |
| C_MTE1_MAC 掩盖失效 | 同上，重叠度 <10% 命中 |
| C_MAC_FIXPIPE 掩盖失效 | 同上，重叠度 <10% 命中 |
| V_MTE2_VEC 掩盖失效 | 同上，重叠度 <10% 命中 |
| V_VEC_MTE3 掩盖失效 | 同上，重叠度 <10% 命中 |
| V_C 掩盖失效 | 整体 C 流水和 V 流水的重叠度，重叠度 <10% 命中 |
| PIPE 数据阻塞 | 等待标志为 `MTE*_WAIT`（等待数据搬运） |
| PIPE 标量计算阻塞 | 等待标志为 `SCALAR_WAIT` |
| PIPE 资源竞争 | 需要结合 ResourceConflictRatio.csv 分析 |
| PIPE CV 阻塞 | 等待标志指向其他计算核心 |
| PIPE 多核合作阻塞 | 代码中涉及多核规约，且多核出现 idle 事件 |

判定优先级建议：先查掩盖类（结构性问题，决定整体排布）→ 再查阻塞类（WAIT_FLAG
直接归因）→ 资源竞争与多核合作需结合 CSV 与代码，证据不足如实标注。

## 成因 → 消除方向

| 成因 | 消除方向 | 可优化性 |
|---|---|---|
| 掩盖失效（C_MTE2_MTE1 / C_MTE1_MAC / C_MAC_FIXPIPE / V_MTE2_VEC / V_VEC_MTE3 / V_C） | **使能双流水（双缓冲）**，重叠搬运与计算；增加并行度 | ★ 可优化（本表主要目标） |
| 数据阻塞（MTE*_WAIT） | 双缓冲；增大单次搬运量、访存连续化减少搬运次数 | ★ 可优化 |
| 标量计算阻塞（SCALAR_WAIT） | 简化标量逻辑、参数预取、输入标量常量化 | ★ 可优化 |
| 资源竞争 | 按上板 ResourceConflictRatio.csv 子表处理（改 stride / 调调度） | △ 结合上板 |
| CV 阻塞 | 调整 Cube/Vector 任务配比（AIC:AIV），或流水重排 | △ 视依赖结构 |
| 多核合作阻塞 | 合理化分核、减少/异步化跨核规约 | △ 视代码 |
| 正常发射间距 / 首尾 tile 填充排空 | 结构性开销 | ✗ 基本不可优化 |

> 「是否可优化」必须逐 pipe 给出：可优化空泡占比高 → 消气泡收益大；以结构性
> 开销为主 → 消气泡无益，回到占比靠前的 pipe 做计算优化或重排流水。

## 输出格式（并入报告 §4）

```markdown
### 18 pipe 空泡情况（1C+2V）
| 子核 | pipe | 活跃占比 | 空泡率 | 空泡原因（定位表条目） | 代码位置（源文件:行） | 是否可优化 |

### 每核心空泡统计
| core | idle 空泡占比 | 主导原因 | 是否可优化 |

### 掩盖矩阵（pipe 对重叠度，仅列 <10% 命中项）
| pipe 对 | 重叠度 | 判定 | 双方主导代码行（各自 Σdur top） |

### 阻塞点 → 代码位置（WAIT_FLAG/BAR 按 args.code 聚合，Σdur 降序）
| 源码行 | 等待边（PIPE←TRIGGERPIPE,FLAGID） | 次数 | Σdur | 所属 pipe |

### Top3 空泡（按时长降序；根因必须带代码行）
#1 [类型] gap=XXus @ coreX.pipeY | 根因：WAIT_FLAG=... @ kernel.py:LINE | 优化：...

### 阻塞点（独占执行期：1 pipe 执行 + 其余等待）
| 执行 pipe | 代码行 | Σdur | 期间等待的 pipe |
（DEVI 等编译器插入的同步在此表标注「无用户行」并给 pc_addr/循环体定位）
```

## 注意事项

- **单位校验（实测标定，CANN 9.0.0 msprof op simulator）**：trace 文件头声明
  `displayTimeUnit: "ns"` 但 **ts/dur 实际单位是 µs**。标定证据：dur 100% 量化在
  **1.85GHz 仿真时钟周期**（1 cycle = 1/1850 µs ≈ 0.5405ns）上——`dur × 1850`
  恒为整数，最小非零 dur 恰为 1 cycle；span 按 µs 解释与板级时长覆盖率自洽
  （实测 143.9µs ≈ 板级 668.7µs 的 21.5%，部分采集），按 ns 则物理不可能。
  查看器若忠实按声明单位显示会差 1000 倍；结论一律按相对比例采信。
  （对比：cannsim/npusim 家族 ts 单位是 ns——两族工具声明与实际相反，勿混用。）
- **ts 不从 0 开始**：trace 时间轴含 app 启动偏移（实测起点 ≈124.3µs）；绝对 ts
  无意义，区间长度与相对位置才有意义。timeline 查看器常把 track 起点归零显示。
- **覆盖校验**：trace 覆盖时长应与 kernel 量级相符；碎片 trace（只覆盖启动段）
  不采信。
- **口径一致**：跨轮对比气泡率变化前，确认 sim_index.json 的 comparability
  （impl/task 指纹一致），否则无对比意义。
