# CANN 提取决策导则（任何算子通用，op_8 SelfAttention 回溯蒸馏）

> 配套 `ascendc_to_okf_workflow/README.md` 的「CANN 提取价值方法论」章节（§A~§F）。
> 本文档是**操作流程**（怎么抽 / 信几分 / 何时回头），供提取会话加载，直接照做。
> 环境标注：910b / triton-ascend 3.2.2 / aclnn 参考容差；骨架变更后强制性重估（见 §E 复活规则）。

## 0. 一句话总纲

**CANN 给「目标结构」和「根因事实」，本地 profiling 给「数值杠杆」。先抽 host 结构把骨架档位解锁到 CANN 齐平，再用本地实测在解锁后的档位上精确收割。** 两大死局：① 没抽结构直接本地磨（0.55 卡住的地毯式空挖）；② 抽了却照搬全部 kernel 手工（不可移植机制白耗）。

## 1. 提取顺序（按性价比从高到低，每步产出决策）

```
STEP-0  校验 arch + 参考模板多样性      → 目标：别拿错 arch 的模板对；模板价值看差异、不看数量
    ├─ 先确认目标芯片实际跑哪个 arch（op_kernel/arch<NN>/ 目录、__DAV_*_CUBE__ 宏；910b=arch35，
    │    勿用旧 arch32 骨架对照——会解释出「triton 写错」的假象，实为拿错 arch）
    ├─ 参考多个模板时：价值 = 跨变异差异（跨 arch / 跨骨架），≠ 数量。同源/同 arch 的几十个模板
    │    批量对照 = 在同一先验里打转，不产生新信息
    └─ 决策：优先对照跨 arch 差异（arch32 vs arch35 的 sync/L1/流水差异是信息密度最高的单点），
          同源模板取一个代表即可，别用「参考了多少个模板」当提取质量的指标

STEP-1  读 host 侧 tiling 档位阶梯      → 目标：把骨架的核必档位(BSQ/BSK/BD/动态D)解锁到 CANN 最高档
    ├─ op_host/*_tiling_basic.cpp（attention 族：CalcDBasicBlock/CalcS1S2BasicBlock）
    ├─ op_host/*_tiling.cpp / tilingdata 字段
    └─ 决策：骨架当前档位 vs CANN 最高档。落后 → 改骨架档位 + 过编译门看真机可达性（含 UB 守卫放行，见 §C）

STEP-2  读 arch 事实 + 不可移植机制清单  → 目标：知道哪些 CANN 机制 triton 永远拿不到、别浪费时间
    ├─ op_kernel/arch<NN>/*kernel_train.h（流水/同步/缓存复用）
    ├─ config/*.json（芯片/dtype 支持矩阵）
    └─ 决策：列出「triton 无对应物/不可移植」清单（如 L1 3-buffer、taskId deep 流水、跨核 flag ring buffer）→ 显式排除，不抄

STEP-3  本地实测（P2.5 之后进入）        → 目标：在解锁后的档位上精确定位真瓶颈
    ├─ 先过 simulator 4 表 pipe 门（Cube+Vector 各 instr_exe+code_exe）
    ├─ 专门扫「精度补偿/中间重算」overhead（hi/lo 双 dot、双精度拆解、冗余 cast）——load 和 tile 轴扫不到
    ├─ **先建「case→档位」决策表再实测**：不同 case 的最优模板不同，无全局最优档。
    │    把 CalcS1S2BasicBlock/CalcDBasicBlock 的 if/else（自变量→档位映射）原样平移到 triton
    │    host 门控，Triton 侧每个 case 落进自己的桶（详见 README P2.5 ③.SRC/③.5）
    └─ 决策：pipe 占比 → 是访存/同步/计算/精度补偿哪类 → 对应修（解耦 / 调整 tile / 去补偿 / 不动）
    │    实测只裁决「每桶 1 个代表 shape 的执行确定」：桶内相对（CANN 档位权威）× 可达性上界
    │    （真机裁决，如 HD144 桶 400KB 收 / HD224 桶 462KB 挡）——两级都过才写 verified，
    │    禁按单点最优把所有桶压到一个档（防 HD≥192 被 {128,128} 误导）

STEP-4  复活规则检查                    → 目标：不继承过期禁令
    └─ 读过的每张证伪/精度卡，核对作用域 {arch, 骨架, 参考容差, case集}。与本次上下文任一不同 → 重测，禁止直接继承
```

## 2. 信几分：CANN 四层归责速查表

| CANN 给的东西 | 可信度 | 用法 |
|---|---|---|
| host 档位阶梯数字（{128,128}/动态 D/AlignUp64）| ✅ 高 | **直接喂 triton tile 阶梯权重**，但必须过编译门实测真机可达（守卫可能过度保守）|
| UB 预算守卫依据（CANN 靠 L1 3-buffer 放 {128,128}）| ⚠️ 参考层 | 反推「triton host 守卫过保守」→ 按 HD 字节两态放行，勿删全局守卫 |
| kernel 流水机制（taskId/3-buffer/flag）| ❌ 不移植 | 只入「排除清单」，不抄进 triton |
| 数值最优解（GEMM tile / PV 单 dot）| ⚠️ 佐证层 | **本地 profiling 触发**，CANN 只事后背书；别等项目先读 CANN 再动手 |

**关键**：把「CANN 给结构的贡献」算进功劳，但别把「本地精修的数字」全记成 CANN——归因两层都算（方法论 §A）。

## 3. 三大死局与反制（直接查）

| 死局 | 症状 | 反制 |
|---|---|---|
| D1 结构锁死 | 卡在低档位空挖，profiling 只显示「当前配置慢」| 先回 host tiling 阶梯看是否有更高档（§B/C/STEP-1）|
| D2 排除法自封 | 「num_stages 证伪」→「整个方向榨干」| 下硬件极限结论前必须过 pipe 门，且专项扫精度补偿 overhead（§D）|
| D3 继承过期禁令 | 旧卡「PV 必须 hi/lo 双 dot」挡掉单 dot 提速 | 核对证伪作用域，变则重测（§E/STEP-4）|

## 4. simulator 实操红线（少了会白跑）

1. **多主循环块稳态**：S=64（1 KV 块）sync% 虚高，定型瓶颈失真 → 用 S=512/tile 128×128（4 块）稳态。
2. **绕开框架算子**：`torch.randn(device='npu')`=Normal+Cast 在 simulator 下炸 → 用 `torch.empty` + 直接调 kernel 启动器。
3. **用 `msprof op simulator` 而非 board 模式 `msprof op`**：board 只采框架 Cast，采不到 triton kernel 会误判「无数据」；CSV 落盘才算成功。
4. **2 个 shape 点以上**（短/长 T 或 小/大 BK），防单点定型以偏概全。