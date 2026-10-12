# Ascend C 源码 → Triton OKF 知识卡片转化工作流

> 版本：v1.5（2026-10-09，op_8 SelfAttention 实测回溯 → 新增「CANN 提取价值方法论」章节：四层归责 / 结构性解锁优先 / host tiling 阶梯>kernel 流水 / 排除法≠穷举 / 复活规则作用域声明 / simulator 实操补遗）
> 上一版：v1.4（2026-09-18，P2.5 实测新发现回流 backend_facts 立法——证据纪律第 5 条；首批实例 cumsum ①编译门 3 条 → C-020~C-022）
> 目标：把 Ascend C 算子源码（`src/mhc/<op_name>/` 结构）**按算子详设模板的章节体系**语义抽取、转化为可指导 Triton 算子生成的 OKF 知识卡片，并通过「回放验证」确认卡片对生成效果的实际指导力。
>
> 依据的模板与详设（`model/design-model/`）：
> - `算子详设文档模板/算子详设文档模板.md`（主模板，章节 1~5）
> - `1.1.1 算子详设模版/1.1.1 算子详设模版.md`（tiling/内存/流水三节细化）
> - `XX算子功能设计说明书/XX算子功能设计说明书.md`（原型约束/规格约束/DFX）
> - `【设计】A5平台grouped_matmul前向triton算子详设文档.md`、`【设计】Lighting Indexer的triton算子设计实现.md`、`【设计文档】FlashAttentionV2 前向算子/`（**Triton 详设实例**，作为卡片正文的体裁基准）
>
> 输出卡片格式对齐既有知识库 `zmm_knowledge/triton-ascend-kb` 的 OKF v1 schema（`kind: operator` / `type: operator_spec`）。

---

## 0. 总体设计思路

详设模板的章节体系本质是算子知识的**正交分解**：需求/公式（算什么）→ 原型/约束（接口长什么样）→ tiling（怎么切）→ 片上内存（放哪里）→ 流水编排（怎么排）→ 测试标杆（怎么算对）。这正是生成 Triton 算子时 Agent 所需的全部决策点，因此**卡片章节与详设模板章节一一对应**，保证无信息遗漏；转化层只做一件事：把 Ascend C 的实现语言（TQue/TBuf/DataCopy/Reg API/tilingkey）翻译成 Triton 的决策语言（grid/BLOCK/mask/tl.dot/multibuffer/autotune），并显式标注 **Triton 可达性边界**（卡片不承诺 Ascend C 的每一处优化都能在 Triton 复现）。

### 流水线总览

```
src/mhc/<op>/                          model/design-model/（模板，只读参考）
      │                                        │
      ▼                                        ▼
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
│ P0 预检    │→ │ P1 语义抽取│→ │ P2 语义转化│→ │ P2.5      │→ │ P3 卡片撰写│→ │ P4 质量门  │→ │ P5 入库闭环│
│ 材料归集   │   │ 按模板章节 │   │ AscendC→  │   │ 骨架实测门│   │ OKF v1    │   │ 完备性+   │   │ lint+反链 │
│ manifest  │   │ extraction│   │ Triton    │   │ 编译/正确/│   │ card.md   │   │ 回放验证  │   │           │
│           │   │           │   │           │   │ simulator │   │           │   │           │   │           │
└───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘   └───────────┘
   产出物统一放在 ascendc_to_okf_workflow/output/<op_name>/ 下
```

| 阶段 | 目标 | 输入 | 产出 | 质量门 |
|---|---|---|---|---|
| P0 预检归集 | 找齐源码材料，建立行号锚点 | `src/mhc/<op>/` 全目录 | `manifest.md` 材料清单 | 五类材料齐全或有缺省说明 |
| P1 语义抽取 | 按详设模板章节填事实（不含转化） | manifest + 源码 | `extraction.md` 结构化事实表 | 每节均有 `file:line` 锚点 |
| P2 语义转化 | Ascend C 概念 → Triton 决策 | extraction.md | `mapping.md` + Triton 骨架伪码 | 映射表每条可对照；不可达点全部显式列出 |
| **P2.5 骨架实测** | 用真机/simulator 把骨架的编译边界、正确性、瓶颈类型实测固化 | triton_skeleton.py + NPU 环境 | `p25_report.md`（编译配置表 + smoke 结果 + pipe 诊断） | 四门全过或失败项写入卡片 ⚠️ 两态 |
| P3 卡片撰写 | 生成 OKF v1 卡 | mapping.md + p25_report + 卡片模板 | `cards/<op>.md` | frontmatter 合规；正文八节齐全 |
| P4 质量门 | 验证卡片能指导生成 | 卡片 + 标杆 | `validation_report.md` | 完备性 checklist 全过 + 回放验证达标 |
| P5 入库闭环 | 合入知识库并挂反链 | 过检卡片 | KB ops/ 卡 + lint 全绿 | `kb_finalize.py` 0 blocker |

---

## CANN 提取价值方法论（op_8 SelfAttention 实测回溯, 2026-10-09，v1.5 新增）

> 来源 = op_8 SelfAttention 全链路回溯（speedup 0.55 卡死 → CANN 提取 → 1.0+，含 PV 单 dot
> +13.1% 单点优化）。它回答 P0~P5 没覆盖的三个问题：**CANN 依然贡献多少、先抽哪一层、什么时候该回头重抽**。
> 本条是**跨算子横切反思**，不替代 P0~P5；各算子按本条指定优先级再走 P0~P5。

### A. CANN 贡献四层归责（别把功劳算错，也别低估）

| 层 | CANN 贡献形态 | 本地能否替代 | 案例 |
|---|---|---|---|
| **L0 架构事实/根因** | 芯片实际跑的 arch、不可移植机制清单 | ❌ 本地永远探测不到 | arch35 regbase（910b 跑 arch35 `KERNEL_MIX_AIC_1_2`，仓库旧模板是 arch32 BN2GS1）→ 解释 triton 的 sync 为什么对不上 CANN |
| **L1 结构性解锁（最高价值）** | 激活当前架子里「想不到/被守卫判死」的正确目标配置 | ⚠️ 单靠 profiling 走不到 | S1/S2 主档 {128,128} 被 host 210KB UB 守卫判死刑；只有 CANN tiling 源码告诉它才是默认档，才去绕过守卫放行 + 动态 D |
| **L2 负面边界** | 哪些 CANN 机制不可移植、实测劣化 | ✅ 可本地试（但贵） | num_stages（taskId 深流水对应物）、大 BSK（DN-mode s2=256）——两笔都被证伪 |
| **L3 事后佐证** | 给本地已发现的优化提供「CANN 也这么干」背书 | ✅ 本地优先 | PV 单 dot（CANN bmm2 就是原生单 dot）——本地 simulator 触发，CANN 仅背书 |

**关键结论**：CANN 的价值集中在 **L1 结构解锁**，不在数值杠杆。数值最大的收益（GEMM tile、PV 单 dot）是本地 profiling 挖的，但它们的「分母」（128×128 大档位）是 CANN 铺的——**没有 CANN 先把大档位解出来，本地优化既无着手点、收益也小一个量级**。归因时两层都要算，缺一不可。

### B.「优化挖不动」要先怀疑目标结构，再怀疑方法（知识缺口 ≠ 努力缺口）

op_8 卡在 0.55 时不是没认真挖，是**当前骨架的核心 tile 档位本身就错了**（host 守卫把 128×128 判死，只在 64×128 打转）。**P2.5 的 simulator 门有一个致命盲区：它测的是「你已经派生出来的骨架」，测不出「存在一个更好档位」**。profiler 能告诉你「现在的 64×128 慢在哪」，不能告诉你「有个 128×128 真机可达且更快」。

**反制规则**：当 P2.5 ③ 显示当前配置已近饱和（MMAD 占比降不下去 / 访存同步占比高但不可再压）时，**先回头读 CANN host 侧 `*_tiling_basic.cpp` 的档位阶梯（`CalcDBasicBlock` / `CalcS1S2BasicBlock`），别深入 kernel 流水**——host 档位是「候选配置」本身，kernel 流水大半不可移植。骨架档位落后于 CANN 最高档，是「挖不动」的第一顺位原因。

### C. host 侧 tiling 阶梯 > kernel 侧流水：attention 族提取优先级

| 提取目标 | 可移植性 | 性价比 | 落点 |
|---|---|---|---|
| **host `*_tiling_basic.cpp` 的 D/S1/S2 档位阶梯** | ✅ 高（纯配置数字）| ★★★ 最高 | `CalcDBasicBlock`（`AlignUp(HD,64)`→{64,128,192,256}→768）、`CalcS1S2BasicBlock`（HD≤256 非 fp32→{128,128}；fp32/大 HD→{64,128}）——直接喂 triton tile 阶梯权重 |
| **UB 预算放行依据** | ✅ 高 | ★★★ | CANN 能放 {128,128} 靠 L1 3-buffer；triton 无此机制但 host 守卫判死反而是过度保守 → 按 HD 字节两态放行 |
| **kernel `kernel_train.h` 流水**（taskId ring buffer / L1 3-buffer / 跨核 flag）| ❌ 大多不可移植 | ★ 低 | 只作排除清单（CANN 有、triton 无），poin不抄 |

**符号化结语**：attention 族提取**先 host tiling 阶梯，后 kernel 流水**；kernel 流水多数时候只输「接不动」清单，别在其上耗时间。

### D. 排除法 ≠ 穷举：「无改进」不能当成「无空间」

op_8 曾以「num_stages 证伪 + 大 BSK 证伪」推出「attention 已榨干」——错在两个否定只覆盖**两根我试过的轴**（流水排程、tile 形状），没碰「周期到底烧在哪条指令」。实际 simulator 4 表一发：Cube 47% 烧在 **PV 精度补偿 2-dot**（一类既跟 load 流水无关、也跟 tile 切分无关的浪费），单 dot 化 +13.1%。

**反制规则（强化 P2.5 门）**：下「无优化空间 / 硬件极限」结论前，必须过 pipe 门（Cube+Vector 各 instr_exe+code_exe 4 表），且采集中要**专门扫「精度补偿/中间重算」这一类 overhead**（hi/lo 双 dot、双精度拆解、冗余 cast）——它不会在 load 或 tile 轴上暴露。**「尝试过的轴无效」只能否定那个轴，不能否定未探测区域。**

### E. 复活规则：旧证伪的作用域必须声明，作用域变必重测

模板曾记「half PV 单 dot → 精度证伪，必须 hi/lo 双 dot」。op_8 在（910b + skeleton + **aclnn 宽松容差** atol 9e-2/1e-1）下重测**通过** 50/50，+13.1%。旧证伪在旧（芯片 + 更严容差 + 小 D 桶）下成立，作用域变了。

**卡片新增要素**：每条证伪/精度策略须写**作用域声明** = `{arch, 代码骨架, 参考容差, case 集}`。任一作用域改变时，后续提取会话**必须重测**，禁止继承禁令。这直接复用 CLAUDE.md 复活规则，但要求显式落在卡片上。

### F. simulator 实操补遗（写入 P2.5 ③ 门，op_8 踩坑）

1. **小 shape 必须用多主循环块稳态**：S=64（tile 内仅 1 个 KV 块）时 sync/bar 占比被人为抬高，看不到稳态流水成本。用 S=512/tile 128×128（4 块）才有代表性。**别用单块小 shape 定型瓶颈。**
2. **simulator 下框架算子会炸**：`torch.randn(device='npu')` = Normal+Cast，动态算子 JSON 缺失导致 `AclOpKernelInit failed`，跑不到 triton kernel。**用 `torch.empty` 分配 + 直接调 kernel 启动器**（纯 triton，绕开框架算子）。
3. **board 模式 `msprof op` 采不到 triton kernel ≠ 跳过**：board 模式只采框架 Cast，会误判「采不到」。必须用 `msprof op simulator`（`--kernel-name` + `--launch-count=1`），CSV 落盘才算成功。

---

## 参考源与检索指引（两个外部库）

本工作流的萃取对象是 CANN 算子源码（`src/`），但**卡片中凡涉及"后端现实"的论断**
（编译选项、tl op 支持、dtype、硬件常数、编译器自动行为、降级路径），一律不允许
凭模型先验撰写，必须从以下两个权威参考源检索取证。证据仲裁顺序：
**IR 源码 > zh 文档 > P4 实测 > 模型先验（先验仅允许标「未经后端确认」）**。

| 参考源 | 路径 | 角色 |
|---|---|---|
| **backend_facts（本地蒸馏层，首查）** | `ascendc_to_okf_workflow/backend_facts/` | 已从两个外部库提炼的六卡一表，P2/P4 的默认入口 |
| **zh：Triton-Ascend 官方文档** | `zmm_source_retrieval/model/zh/` | 用户视角的后端说明书：API 支持、编译选项、性能指南、算子开发范式、调试手段 |
| **IR：AscendNPU-IR 编译器源码** | `zmm_source_retrieval/model/AscendNPU-IR/`（`bishengir/`，本地副本，快照 @`9c51a351` 2026-09-04；刷新副本时更新此记录并复核锚点行号） | 后端实现的地面实况：pass 行为、op 约束、target spec；与 zh 冲突时以它为准 |

### 检索顺序

1. **先查 `backend_facts/`**：命中即引用（卡片写引用，不复写内容）；
2. 未命中 → **查 zh**（用户可读的官方表述，适合卡片直接引用）；
3. zh 未覆盖或与实测冲突 → **查 IR 源码**（最终裁判）；
4. 三处都无 → 标「未经后端确认」，禁止编造；
5. P2.5 ①②门 / P4 新发现的后端事实 → **当次回流 backend_facts**（流程见 P2.5 证据纪律第 5 条）。

### 按阶段的检索任务

| 阶段 | 检索任务 | 检索位置 |
|---|---|---|
| P0/P1 | 不查外部库。P1 只做源码事实搬运，后端论断不得提前出现 | — |
| **P2**（检索最重的阶段） | ① 骨架每个 tl op 的 lowering 代价分级；② 每个 ⚠️/❌ 可达性判定的依据；③ 编译选项选择；④ UB/对齐预算校核常数；⑤ dtype 陷阱；⑥ 契约违禁写法 | ① `backend_facts/tl_op_lowering.md` ← zh `architecture_design_and_core_features.md` §3.2.2 ← IR `bishengir/triton/lib/Conversion/`；② `backend_facts/auto_passes.md` ← IR `bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp`；③ `backend_facts/compiler_options.yaml` ← zh 环境变量文档 ← IR `lib/Tools/bishengir-compile/PassPipeline.cpp`；④ `backend_facts/hardware_platforms.md` ← IR `include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td`；⑤ `backend_facts/dtype_support.md` ← zh `migration_guide/performance_guidelines.md` ← IR `include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td`；⑥ `backend_facts/backend_contracts.md` |
| P2（降级路径） | 「❌不可达」的替代写法优先找官方细粒度 API | zh `triton_api_extention/al/`、`bl/`（sync_block 族、bind_buffer、fixpipe 等） |
| P2（范式参考） | 同类算子的官方开发范式与完整实现 | zh `programming_guide/{vector,cube,cv_fusion}_operator.md`、`examples/` |
| P3 | 卡片引用格式：外部论断挂 backend_facts 卡名或 zh 章节；源码锚点挂 IR `file:line` | — |
| P4 | ① 生成前 preflight（grep 清单）；② 性能测量与 profiling 手段；③ 新陷阱取证 | ① `backend_facts/backend_contracts.md` 文末清单；② zh `debug_guide/`；③ 按上面检索顺序取证后回流 |
| P5 | KB 入库卡片的外部引用链接规范化 | zh 在线文档 URL / IR GitCode @sha 永久链接 |

### backend_facts 维护纪律

- 六卡一表是**蒸馏层，不是终点**：发现 zh/IR 有新事实（尤其 950 平台相关内容），
  先回流对应卡再使用；
- **回流触发点**（2026-09-18 扩展）：① P2.5 各门实测新发现（强制，流程见 P2.5
  证据纪律第 5 条）；② P4 回放新发现；③ zh/IR 刷新对照——三类同源同流程；
- 每卡「维护记录」节留有已知待办（如 950 dtype 矩阵、regbase 差异），认领后更新；
- 禁止在 ops 卡片中复写 backend_facts 已有的通用事实（稀释检索相关性），只挂引用。

---

## P0 源码预检与材料归集

**目标**：把分散在算子目录各处的材料定位、归类，为 P1 建立可引用的行号锚点。

**标准目录结构**（以 `src/mhc/mhc_sinkhorn/` 为基准，缺失项需在 manifest 中注明）：

| 类别 | 路径模式 | 承载的详设章节 |
|---|---|---|
| 原型定义 | `op_host/<op>_def.cpp` | 原型设计、原型约束（dtype/format/shape） |
| aclnn 接口 | `op_host/op_api/aclnn_<op>.h/.cpp`、`<op>.cpp` | 对外接口、属性表 |
| shape 推导 | `op_host/<op>_infershape.cpp` | Infershape 设计、输出 shape 推导 |
| tiling host 侧 | `op_host/<op>_tiling.cpp/.h`（或 `op_tiling/`） | 分核策略、tilingdata 字段语义、tilingkey 选择逻辑 |
| 平台配置 | `op_host/config/<platform>/*.json`、`*_simplified_key.ini` | 支持芯片、tilingkey 枚举 |
| kernel 入口 | `op_kernel/<op>_apt.cpp` | 模板分支（TEMPLATE_MODE）、AIC/AIV、tilingkey dispatch |
| kernel 实现 | `op_kernel/arch<NN>/*.h`（class 定义、Init/Process/CopyIn/Compute/CopyOut） | 片上内存分配、流水编排、计算流程 |
| tiling 数据结构 | `op_kernel/arch<NN>/<op>_struct.h` 或 `*_tiling_data.h`、`*_tiling_key.h` | tilingdata 字段、tilingkey 位定义 |
| 文档 | `README.md`、`docs/aclnn<X>.md` | 功能描述、规格约束、值域约束 |
| 测试 | `tests/ut/**`、`tests/st/**`、`examples/test_aclnn_<op>.cpp` | 测试标杆、典型用例 shape、**官方验收容差与填充分布**（ST json 的 `error_threshold`/`range_values` 字段）、**tiling UT 的期望 tilingkey**（分派链正确性的官方测试证据）、CPU-sim UT 是否活跃 |
| 图级语义 | `op_graph/fusion_pass/`、`op_graph/` 其他 pass | 图级融合/消除 pass 对算子语义的利用（如 in-place 语义被 tensormove pass 消除上游拷贝）——**算子语义的最强旁证**，P0 漏扫会导致卡片语义节缺图级证据 |

**动作**：
1. 按上表逐项扫描，生成 `manifest.md`：每项记「路径 + 是否存在 + 关键符号/行号 + 对应详设章节」。
2. 标记架构覆盖：`arch22`/`arch35` 目录并存时，卡片以 **arch35（950）为主线**，arch22 差异写入「架构差异」小节。
3. 标记 kernel 类型：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` → AIV-only（纯向量）；含 `Mmad`/`Cube` → CV 融合。该判定决定 P2 的映射路径，必须在 manifest 显著位置标注。
4. **判定壳算子**（add 经验）：若 kernel 入口仅是 dtype/format 分派壳（如 `add.cpp` 4 路 constexpr → `BroadcastSch<schMode, OpDag>`），真正的调度器/DAG 在 base 库或头文件——此时：① 计算 DAG（`*_dag.h`）是**在仓内的核心语义资产**，P1 逐链抽取精度策略（cast 提升点/回写舍入模式）；② 调度器本体不可得时先标 inferred，再查 `model/opbase-master-src-op_common/src/op_common/atvoss/<家族>/`（broadcast/elewise/reduce 家族 tiling 实现已在库，可补 §4/§5 源码锚点）；③ 不可达部分显式列入 manifest 缺省项，不许静默留空。
5. **比对 benchmark 语义**（add 经验）：卡片服务 benchmark 生成，P0 必须 diff 两边公式——benchmark 常为源码语义**超集**（如 `3_Add.py` 的 `torch.add(x,y,alpha)` 含 alpha，而 GEIR kernel 本体 `y=x1+x2` 无 alpha，aclnn 层才消化 alpha）。缺口写入 manifest 缺省项第 1 条，P1 优先溯源（aclnn 层的参数消化策略是答案高发区）。
6. **算子复杂度预判**（fused_bias_leaky_relu P4 经验）：满足以下全部条件的算子标「**低价值**」——❌ 不可达项为 0、无广播/规约/gather、公式与精度策略在基座模型先验内（纯 elementwise 标准链）、无 benchmark 语义超集。低价值算子的卡片正确性增益≈0（对照组首轮即满分），P4 可跳过或降为单组冒烟；卡片价值密度在「有硬件陷阱的算子」（Reg API、split-K、布局差异、精度坑）。预判结论写入 manifest 显著位置。

**质量门**：五类材料（def / tiling / kernel / 文档 / 测试）齐全；缺项写明「源码未提供，需需求方补充」——不允许留空进入 P1；若属壳算子，须额外确认「DAG + aclnn 策略 + base 库 tiling」三类替代资产中至少一类在仓内可读。

---

## P1 语义抽取（按详设模板章节）

**目标**：只做「事实搬运」，不做转化。按模板的章节顺序输出 `extraction.md`，每个论断带 `file:line` 锚点（可追溯性是 P4 回放验证的前置条件）。

| 详设章节 | 抽取来源 | 抽取要点 |
|---|---|---|
| 1.1 功能分析（算什么） | README、def.cpp 注释、kernel Compute 函数 | 算子语义一句话 + 数学公式（从 Compute 反推，写出每个变量含义）；反向算子列梯度公式 |
| 2 原型设计（接口） | def.cpp `REGISTER_OP` / aclnn 头文件 | 输入/输出/属性表：名称、必选性、dtype、format、shape 语义（对齐模板「原型设计」表格式） |
| 2.1 原型约束 | docs/aclnn*.md、def.cpp 校验逻辑 | shape 边界、值域边界、输入间依赖关系、拦截校验 |
| 3 规格约束 | config/*.json、README、def.cpp | 芯片型号、dtype 全集、format、确定性计算、交付形式（对齐模板规格约束表） |
| 4.1.1 分核策略 | tiling.cpp 中 `usedCoreNum`/blockDim 计算 | 切分轴、核数决策公式、尾核处理（normal/tail core 划分） |
| 4.1.2 单核内切分 | tilingdata 字段（如 `tUbFactor`/`tUbFactorTail`/`tTailCoreLoop`） | 轴切分优先级（能全载全载、否则对齐切分）、UB 因子语义 |
| 4.1.3 tilingdata/tilingkey | `*_struct.h` 字段 + `*_tiling_key.h` + apt.cpp 模板分支 | 每个字段：语义、设计边界、host 侧计算公式；tilingkey 各位含义与分支条件 |
| 5 片上内存 | kernel 类中 `TQue`/`TBuf` 声明 | 每个 buffer：位置（VECIN/VECOUT/VECCALC）、容量表达式、buffer_num（是否 double buffer） |
| 6 流水编排 | Process/CopyIn/Compute/CopyOut 函数体 | CopyIn→Compute→CopyOut 的循环结构、循环内搬运粒度、是否双缓冲掩盖、特殊指令（Reg gather/scatter、DataCopyPad） |
| 6.x 计算流程 | Compute 函数逐步注释 | 步骤序列（对齐模板「算子整体计算流程图」），每步标注 Vector 指令级操作 |
| 精度策略 | Compute 中的 cast/提升点 | 中间累加精度（bf16→fp32 等）、与竞品对齐点 |
| 测试设计 | tests/、examples/ | 标杆定义（单标杆/双标杆）、典型 shape、边界用例 |

**产出**：`output/<op>/extraction.md`，章节号与详设模板一致（1.1 / 2 / 2.1 / 3 / 4.1.x / 5 / 6 / 测试）。

**质量门**：任一章节无锚点 = 不合格；公式与 Compute 实现不一致时必须以代码为准并标注「文档与实现偏差」。

**语义分层归集**（add 经验，壳算子/多入口算子必做）：同一算子常有**三层语义**——GEIR kernel 本体（`op_kernel/`，无属性）、aclnn 接口层（`op_api/aclnn_*.cpp`，消化 Python 侧参数如 alpha）、benchmark 层（torch golden）。P1 须分层各抽一节并显式标注差异（如「aclnn `aclnnAddGetWorkspaceSize(self, other, alpha, out, ...)` 含 alpha，经 4 策略消化：直调 Add / Axpy 融合 / AxpyV2 / Mul+Add 两 kernel」）；**严禁沿用 P0 假设**——add 曾先断言「aclnn 层也不含 alpha」，P1 逐行核 aclnn_add.cpp 后推翻，偏差记录必须允许自我纠正。base 库 tilingkey 若语义不明，破译路径为「innerKey = computeKey×OFFSET + scheduleKey」（broadcast_tiling.cpp:604-605），scheduleKey 常只是 NDDMA 维数分档——破译后若与 Triton 无关，P2 直接重定义新 constexpr。

---

## P2 语义转化（Ascend C → Triton）

**目标**：把 P1 的事实翻译为 Triton 生成决策。映射规则集中在 [`ascendc_triton_semantic_mapping.md`](./ascendc_triton_semantic_mapping.md)，本节规定转化顺序与判定逻辑。**凡涉及后端现实（tl op 支持、编译选项、dtype、硬件常数、编译器自动行为）的论断，按『参考源与检索指引』检索取证：先 backend_facts → 再 zh → 最后 IR 源码。**

### 2.1 转化顺序（与 grouped_matmul / Lighting Indexer 详设的决策顺序一致）

1. **分核策略 → grid 设计**
   - Ascend C：`usedCoreNum` + blockIdx 核内循环 → Triton：固定范式「总任务块数 ÷ 物理核数，`for block_idx in range(pid, NUM_BLOCKS, NUM_CORE)`」（两实例均用此范式）。
   - 决策点：切分轴选择（输出轴优先做任务轴，规约轴只进循环）；任务块总量公式。
2. **tilingdata → BLOCK 超参**
   - `tUbFactor` 等 UB 因子 → `BLOCK_SIZE_*`；尾块字段 → mask 边界逻辑；autotune 候选值从 tiling 边界推出（如对齐 128/256/512）。
3. **tilingkey 模板分支 → `tl.constexpr` / autotune key**
   - 每个 TEMPLATE_MODE 分支 → 一个 constexpr 开关或独立 kernel；多分支差异大时拆多 kernel，差异小时合并。
4. **片上内存 → 隐式化 + 预算校核**
   - TQue/TBuf 在 Triton 无对应物，但要计算等效 tile 字节数校核 UB 上限（192KB/核级），超预算则缩小 BLOCK 或外提计算。
5. **流水编排 → Triton 可达手段**
   - double buffer → `multibuffer`（编译期开关）或 num_stages；CopyIn/Compute/CopyOut 三段式 → load/compute/store；Reg 级 gather/scatter → **不可达**，标注降级方案（如改 tl.load/tl.store 离散访存并预警性能）。
6. **Compute → Triton 计算骨架**
   - 逐 Vector 指令翻译：`Mmad→tl.dot`、`Add/Mul/Max→对应算子`、`Exp→tl.exp`、`ReduceSum→tl.sum`、`Cast→.to()`；输出 **Triton 骨架伪码**（结构对齐 Lighting Indexer 的 kernel 形式：offsets → tl.load(mask) → 计算 → tl.store(mask)）。
7. **精度策略 → 标杆与验收**
   - 中间精度提升点必须保留（fp32 累加）；验收标准直接写进卡片（如「topk 索引集合一致」这类算子级标准，而非通用 rtol）。

### 2.2 可达性判定（卡片可信度的关键）

每个 Ascend C 优化手段三态标注：
- ✅ **Triton 可达**（给出对应写法）
- ⚠️ **部分可达**（编译器自动完成，如流水掩盖；卡片只写约束不写实现）——须再分「正确性风险 / 性能风险」两级：正确性风险给禁用写法+安全替代，性能风险给劣化量级（2026-09-09 增补）
- ❌ **不可达**（Reg API、bank 冲突规避、NZ 格式、显式 pingpong 等）——必须给出「降级写法 + 预期性能影响」

**产出**：`output/<op>/mapping.md`（决策表：Ascend C 事实 → Triton 决策 → 可达性 → 依据锚点）+ Triton 骨架伪码。

**质量门**：P1 每条 tiling/流水/计算事实都能在 mapping.md 中找到去向（转化/合并/显式丢弃并注明理由），不允许静默丢失。

**转化判定增补**（add 经验）：
- **语义等价优先于形式复刻**：host 侧 `blockFormer = ceil(total/coreNum)` 静态均分 + 尾核 blockTail，与 Triton「grid-stride 循环 + 尾块 mask」是**负载等价**的两种写法——mapping 表须写明等价论证（add mapping.md §1 为范例），卡片 §4 据此标「官方机制已对照」而非「惯例推断」。
- **融合优于分解**：aclnn 层为解决硬件限制做的多 kernel 分解（Mul+Add 两 kernel、Contiguous/Cast 中转），Triton 单 kernel 一律不需要——标 ❌→不需要 并注明「少一次 GM 往返，结构性更优」，生成 Agent 才不会去复刻兜底路径。
- **广播走 expand-stride 折算**：CopyInBrc 的 DMA 展开无 Triton 对应，wrapper 侧 `expand` 取 stride（广播维=0），kernel 内线性 offs 逐维 div/mod 分解坐标折算；等形 case 保留直通路径（SAME_SHAPE constexpr，零整数除法，大 2D 性能关键）。
- **preferMultiCore 类回调**：base 库「核占不满时降 BLOCK 铺核」的策略，Triton 由 autotune 小 BLOCK config 覆盖，mapping 标策略等价，卡片 §10 记一条「BLOCK 初选偏大须优先降 BLOCK」。

**转化判定增补**（2026-09-07 fused_bias_leaky_relu P4 实测经验）：
- **elementwise 分核首选 persistent-core**：「每核 blockFactor 连续段 + 核内 ubFactor 循环」→ `grid=min(cdiv(numel,BLOCK), 物理核数)` + 核内连续段 for 循环（映射表 §1 第 2 行）。**「一块一 program、grid=NUM_BLOCKS」实测为负引导**（超额 block 串行排队，grid=123 vs 40 核仅 0.69x；生成器要花 1~2 轮优化才能爬回），P2 不允许再把源码两级切分抽象成该形态。
- **属性透传给两态，不给禁令**：float 属性默认 runtime 标量（生产语义），benchmark/编译期稳定场景建议 `tl.constexpr`（常量折叠实测 +17.3%）。**给错绝对禁令比沉默更糟**——指令遵循度高的生成路径会被卡片压住不去试（实验 2 优化器据此推理拒绝、未实测丢 17%）；生成器会犹豫的决策点，卡片要么给正确答案，要么给两态权衡。

**转化判定增补**（2026-09-08/09 apply_rotary_pos_emb P4 三组回放经验）：
- **骨架伪码必须过静态自检**：P2 骨架写完后，对照 golden 公式**逐模式逐分支**推演一遍（half/quarter/interleave 各走一遍符号推导）——本次 quarter 分支 `o4 = x4*c4 + x3*s4` 笔误（变量错位）在 P4 回放中直接造成 quarter case 全挂，靠回放才暴露。静态自检是免费的，回放是昂贵的。
- **⚠️ 项按严重性分级**：部分可达项须标「正确性风险」或「性能风险」两级——正确性风险（如 strided-2 访存的三类后端 bug）必须给禁用写法+安全替代，性能风险（如 shifted-load 劣化）给劣化量级即可。不分级会导致生成 Agent 把致命项当可选项踩。
- **勿复刻清单禁止无实测等价断言**：卡片曾写「cos/sin 外提可由 L1/L2 驻留等价」——P4 实测「cache 驻留 ≠ 复用」，逐行重载慢一倍（2.0956→4.2060）。凡「不必复刻 X，因为 Y 机制等价」的断言，必须有实测或源码级论证，否则改写为「⚠️ 未验证，建议实测两态」。
- **P2 映射强制核对后端文档**：`num_stages`/`num_warps` 在 Triton-Ascend 运行时不生效（静默 WARNING）——映射表凡引用 Triton 上游特性（autotune、num_stages、tl.trans 等），须核对 `model/zh/` 后端支持矩阵或在库内找既有 runbook 佐证；未核对的一律标「未经后端确认」。本次直给组因骨架不含 autotune 而避开此坑，KB 组保留 autotune 被 µs-kernel do_bench 噪声系统性误选（geomean 低 15%）。
- **kernel 内禁用 `%` 运算符**（写 `a-(a//b)*b`）、fp32 导出 mask 触发非确定 store 污染、stride-0 memref 常量折叠 MLIR 崩溃——此类后端契约已从本卡沉淀为 runbooks/{portability,memory-sram} 独立卡，并汇总索引于 `backend_facts/backend_contracts.md`（含生成前 preflight 清单），P2 转化时先查 backend_facts 与 runbooks 目录，命中即引用而非复写。

---

## P2.5 骨架实测门（Skeleton Grounding，2026-09-17 新增）

**目标**：在写卡之前，用真机编译/运行与 msprof simulator 把骨架的**编译边界、
正确性、瓶颈类型**实测固化——把「第一轮回放的发现职能」前置到提取期。
立项依据：`output/gdn2_fwd/p25_pilot/pilot_report.md`（回溯试点三项全中：
BV=64、串行同步地板 51.1%、预取方向均可在提取期产出；94 v1 卡的三个根因
全部可被本阶段拦截）。

**前置条件**：P2 骨架已产出且自带 `__main__` smoke（对 ref golden 的全开关
组合小 shape 自测）；本机有 NPU + CANN（`msprof` 可用）。环境红线：
`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、避开坏卡（`ASCEND_RT_VISIBLE_DEVICES`）、
脚本内 `import torch_npu` 注册设备（无此 import 会静默回退 cpu 报指针错误）。

**四道门（顺序执行，全部数据标 [EXP-skeleton] + shape 上下文）**：

| 门 | 内容 | 拦截的缺陷类型 |
|---|---|---|
| ① 编译门 | 对 kernel 的 BLOCK 配置网格（如 BK×BV 多组）真编译+最小运行，产出「可编译配置表」 | UB 边界误判（94 的 BV=16 钉死）、C-016/C-017 类编译陷阱、配置型 UB overflow |
| ② 正确性门 | NPU 上跑骨架 `__main__` smoke（varlen/init/flag 全组合对 ref golden） | 索引错位（hv→hk 类）、dtype 不匹配、环境静默回退 |
| ③ simulator 诊断门 | `msprof op simulator`（小 shape、单次 forward、`--kernel-name` 精确过滤、`--launch-count=1`）→ 解析 4 张表（Cube/Vector 各 instr_exe+code_exe）→ pipe 占比 + 热行 + 瓶颈定型。**至少 2 个 shape 点**（短/长 T 或 小/大 BK），防单点瓶颈定型以偏概全。⚠️ 实操见上方「CANN 提取价值方法论 §F」（多主循环块稳态防 sync 虚高、torch.empty 绕框架算子、board 模式采不到须用 simulator）<br>⚠️ **瓶颈定型前先判档位饱和**：若当前配置近饱和（MMAD 占比低、同步/搬运高且不可再压），先回头读 host `*_tiling_basic.cpp` 档位阶梯，骨架档位落后于 CANN 最高档是「挖不动」第一顺位原因（见方法论 §B/C），别深入 kernel 流水 | 结构天花板不可见导致的错误路线否决（94 的 WY 否决）、杠杆排序无依据 |
| ③.SRC 源码分档决策表（静态，不耗 NPU；③ 判饱和后必做，先于 ③.5） | 不同 case 的最优模板**不同**，不存在一个全局最优档。用 CANN 源码把「case→档位」的**决策表**锁定，避免"锁 1~2 个全局候选"的错误简化、也避免 ③.5 靠网格穷举退回回放。核心：`CalcS1S2BasicBlock` / `CalcDBasicBlock` 的 if/else 本身**就是自变量→档位的映射**，原样平移为决策表：<br>· **档位映射**：`HD×dtype(×S)` → 候选档（HD≤256 非 fp32→`{128,128}`；HD>256 / fp32→`{64,128}`;D 块 AlignUp64→`{64..256}→768`）逐 if 分支带源码锚点<br>· **精度最简方案**：kernel Compute 累加类型 / 是否 hi/lo 拆分 → 锁"该不该去补偿"（PV 原生单 dot，见 §D）。⚠️ **必须读 kernel Compute（arch35 `*_kernel_train.h` 的 acc 声明 + bmm 的 dot 调用），不能只停留 host tiling**——host 档位披露结构，但累加/回写精度藏在 Compute 层（v2 实战漏项：只抓 host、漏了 Compute 导致 single-dot 决策缺失）<br>· **排除清单**：kernel 流水（3-buffer/taskId/flag）标"不可移植不抄"，不列候选<br>产出落卡片 BLOCK 节：「决策表 + 每桶排除 + 锚点」。triton 侧 = host 门控按 HD/dtype 走桶，每个 case 落进自己的桶。<br>**"最优"判定=两级**：桶内相对（CANN 档位权威，同桶 shape 不换档可外推）× 可达性（真机裁决，见③.5）。两级都过才写 verified；任何单档禁写全局"绝对最优"（遵守两态红线）<br>门禁：未做 ③.SRC 禁止直接进 ③.5，否则选型无源码依据、退回回放穷举 | 配置选型无依据退给回放（BV/类 BV 至今靠回放标定）；"锁全局候选"按单点最优误导 HD≥192 桶 |
| ③.5 裁决性实测门（③.SRC 锁定决策表后，每桶必做；成本 ≤ 桶数 × 1 代表 shape） | 对决策表**承载的每个桶取一个代表 shape** 做裁决性确认（非全量基准、非 autotune 阶梯）：<br>· **可达性上界**：HD×dtype×tile 字节 → UB 上界，真机确认「该桶锁定档 编译过 / 爆 GM 慢」（op_8：HD144 桶收 400KB、HD224 桶 462KB 爆 ~90×）<br>· **相对快慢**：该桶锁定档 vs P2 当前档一次对比（≥当前才采纳），因同桶 shape 不换档故一代表可外推<br>· **pipe 复核**：该桶代表再拉一次 ③ 的 4 表，确认瓶颈转移、无隐藏精度补偿<br>⚠️ 不做 50-case geomean / autotune 全扫，留给 P4；本门=把 P4「发现轮」压缩为「确认轮」，只确认决策表每桶边界落地（与证据纪律 2 一致）。已与当前档相同的桶，只做可达性上界 | 配置选型只靠回放标定（慢、且提取期可能无 NPU） |
| ④ IR 取证门（按需） | 对卡片断言的编译器机制做 pass before/after diff（multibuffer/CV 流水/buffer 路径） | 「机制假设，未验证」挂账（C-006/C-011 类） |

**工具与参考（直接复用，不重造）**：
`1-spare/triton-simulator-optimizer/{scripts/enumerate_kernels.py, scripts/parse_prof.py,
references/msprof-simulator.md, references/bottleneck-diagnosis.md}`（采集流程、
环境变量、两表联合解析、pipe→瓶颈→修复方向规则）；
`1-spare/triton-ir-analysis/{scripts/diff_pass.sh, scripts/find_variants.py,
references/ir_dump.md, references/ir_reading.md}`（IR 三途径 dump、pass diff 验证）。

**证据纪律**（与试点报告 §证据纪律 一致，强制）：
1. P2.5 数据一律标 **[EXP-skeleton] + shape 上下文**（芯片/CANN 版本/shape）；
   与 [EXP]（50-case 端到端）区分。外推受 C-012 约束：上下文结论只标两态，
   不升格禁令；
2. simulator 给 pipe 占比不定墙钟——**P2.5 不替代 P4 验收回放**，只把回放从
   「发现轮」压缩为「确认轮」；端到端数字仍以 50-case 真机 benchmark 为准；
3. 瓶颈定型须过「全 kernel 采集覆盖门禁」（多 kernel 算子逐个采，参考
   simulator-optimizer SKILL.md）；热行映射异常时以 instr_exe 为准；
4. 门失败项**不阻塞流程**，但必须写入卡片对应章节的 ⚠️ 两态项（例：某 BV
   配置编译失败 → 卡片 BLOCK 节标注实测边界）；四门全过才可把对应论断的
   confidence 从 inferred 升为 measured(skeleton)。
5. **实测新发现当次回流 backend_facts（强制，2026-09-18 立法）**：①编译门 /
   ②正确性门（及③③.5④门）每抓到一条**静态取证扫不出、只有真机编译/运行
   才暴露**的后端事实（编译陷阱、静默错值、配置包络边界、前端语法约束），
   必须在当次任务内回流共享层，**不允许只记进 mapping.md / validation_report.md
   了事**——单算子产物检索不到，契约只有进共享层才能服务后续生成任务。流程：
   a) **判重**：先 grep `backend_facts/` 同主题条目——命中则修订原条（补证据、
   收窄/扩大适用范围、记日期），不新增编号；未命中才新增契约编号；
   b) **落点**：正确性/性能/用法三级契约 → `backend_contracts.md`（新 C-NNN
   条目，含严重性/现象/禁用写法/安全替代/锚点）；dtype 矩阵 → `dtype_support.md`；
   lowering 代价 → `tl_op_lowering.md`；硬件常数 → `hardware_platforms.md`；
   c) **格式**：证据标 **[EXP-skeleton] + 芯片/CANN 版本/shape 上下文**（单点
   实测值注明「换平台须重探」，方法性结论与数值结论分开写）；锚点挂
   `output/<op>/validation_report.md`；可 grep 化的同步加 preflight 清单行；
   d) **记录**：对应卡的「维护记录」节加一行（日期 + 条目 + 出处算子）；
   e) **一致性**：backend_facts 与卡片论断冲突时，按仲裁顺序修订并双向同步
   （首批实例：cumsum ①编译门 3 条 → C-020/C-021/C-022，2026-09-18）。

**跳过/降级条件**（2026-09-17 增补，与 P0 动作 6「低价值」预判联动）：
- P0 判「**低价值**」的 Vector 简单算子（❌ 项为 0、无广播/规约/gather、公式与
  精度策略在基座先验内、无 benchmark 语义超集）**整体跳过 P2.5**，直接进 P3。
  跳过纪律：① manifest 显著位置记「P2.5 skipped: low-value」；② 卡片 confidence
  保持 `inferred`，不得升 `measured(skeleton)`；③ 卡片后端论断证据等级上限
  [DOC]/[SRC]，无 [EXP-skeleton]；④ 不产出 `p25_report.md`，在 manifest 缺省项
  说明即可。
- 非低价值但想省实测时间的算子，**降级为只过 ①编译门 + ②正确性门**（smoke 随
  骨架自带，成本最低），跳过 ③ simulator / ③.5 board / ④ IR 取证门；BLOCK 选型
  与瓶颈定型论断一律保持两态，不写实测结论。
- 低价值判定与 P4 的「跳过或降为单组冒烟」共用同一份预判结论（manifest 一处记录，
  P2.5/P4 两处生效），不允许 P2.5 跳过而 P4 全量这类自相矛盾的档位组合。

---

## P3 卡片撰写（OKF v1）

**目标**：按 [`okf_operator_card_template.md`](./okf_operator_card_template.md) 生成正式卡片。

关键规则：
1. **frontmatter**：`schema_version: okf.v1`、`kind: operator`、`type: operator_spec`、`confidence: inferred`（静态分析；回放验证通过后升 `verified`）、`tags` 含算子族/计算模式/硬件特性词（禁用 precision_mismatch 等通用症状词，防检索误匹配）。`source_family` 注意：**KB lint 受控词表为 {reference, ops, runbooks, trajectory, curated, community}，无 golden**——工作流产出可写 golden，入库副本（P5）必须改为 `ops`。
2. **正文章节**（与详设模板对应，顺序固定）：功能与公式 → 原型与约束 → 规格约束 → 分核策略（Triton grid）→ BLOCK/Tiling 参数 → 片上内存预算 → 流水编排与可达性 → Triton 骨架 → 精度标杆与验收 → 架构差异 → 复现性声明。壳算子须在总述段显式标注「kernel 本体是 N 路分派壳，调度器 X 本体不可得，§N 机制有 op_common 源码锚点」。
3. **锚点保留**：关键论断保留 `file:line`，使 P4 可回溯。
4. **Agent actions** 小节：面向生成 Agent 的指令式清单（识别算子类别 → 选切分轴 → 取 BLOCK 候选 → 精度提升点 → 验收标准），对齐既有卡的体裁。

**质量门**：frontmatter 字段齐全且枚举值合法；正文八节无空缺（无内容写「不涉及」而非删除章节）。

---

## P4 质量门（验证卡片能真正指导生成）

针对「卡片内容是否真能指导算子生成」的核心疑问，采用三级验证：

### 4.1 完备性 checklist（机械检查）

- [ ] 公式与 Compute 实现一致（抽查 ≥3 个计算步骤）
- [ ] 原型表覆盖 def.cpp 全部输入/输出/属性
- [ ] 分核策略给出**确定的任务块总量公式**（而非定性描述）
- [ ] 每个 BLOCK 参数有候选值或推导式
- [ ] 片上内存预算有具体字节数与 UB 上限校核
- [ ] 流水可达性三态标注全覆盖，❌ 项均有降级方案
- [ ] 验收标准可执行（有标杆函数形态 + 判定准则）
- [ ] 骨架伪码可直接翻译成 @triton.jit（offsets/mask/边界处理齐全）

### 4.2 回放验证（生成效果实测，核心证据）

用**双盲对照**验证卡片指导力：

1. **对照组**：仅给生成 Agent 算子名 + 一句话功能描述，让其生成 Triton 算子。
2. **实验组**：仅给生成 Agent **本卡片**（不给 Ascend C 源码），生成 Triton 算子。
3. 两组在同一环境跑数值正确性（对 torch 标杆）+ 性能。
4. 记录指标写入 `validation_report.md`：

| 指标 | 对照组 | 实验组 | 达标线 |
|---|---|---|---|
| 首次生成可运行率 | | | 实验组 ≥ 对照组 |
| 数值正确迭代轮数 | | | 实验组 ≤ 3 轮 |
| 对标杆精度 | | | 达到卡片所写验收标准 |
| 相对性能（对 aclnn/标杆） | | | 不低于对照组 |

5. **判定**：实验组在迭代轮数或精度上显著优于对照组 → 卡片 `confidence` 升 `verified`；若实验组失败且失败原因可归于卡片缺漏 → 回 P1/P2 补卡（记录缺陷点），重跑。

**测量纪律**（2026-09-07 fused_bias_leaky_relu P4 经验）：
- 三组实验的 perf 参数（warmup/repeats）必须对齐——本次实验 3 repeats=20 vs 其余 repeats=5，直接污染横向对比；NPU 计时空跑抖动 ±3~5%，**性能差 <5% 视为噪音带**，不作结论依据。
- 生成器优化循环是随机的：同一 checklist 同一决策点，两次运行可做出相反决策（本次实验 2 推理拒绝 constexpr、实验 3 实测采纳）。单跑一次 P4 的终值差异不足以为卡片定罪或翻案，须回查优化日志确认差异**归因于卡片内容**（初版骨架/指令遵循）还是**归因于优化循环运气**。
- 简单算子（P0 判低价值）的 P4 预期结论是「卡片增益≈0」而非卡片失效——此时卡片验收标准是「无负引导」（不给次优起点、不写错禁令）。
- **三组对照须控制知识注入纯度**（2026-09-08 apply_rotary_pos_emb P4 经验）：直给组只给主卡+骨架；KB 检索组天然带上下文稀释（本次 134KB：10 张 ops 卡 + 78KB preflight + 无关 runbook），优化器可能沿邻居卡走弯路（本次沿 kernel_splitting.md 白烧 2 轮回滚）。横向对比时须记录各组实际注入材料清单（含 Agent 自主 ls 发现并吸入的文件——直给组 Agent 曾自动发现 triton_skeleton.py 随 inspirations 传入，导致「直给卡组」实为「卡+骨架组」，归因时按实际注入计）。
- **回放产物必须清算 autotune 残留**：生成器在多 shape 扫掠下保留 `@triton.autotune` 时，µs 级 kernel 的 do_bench 噪声会系统性误选 per-shape 配置（本次 KB 组最终代码带 autotune，34/45 case 显著劣化）。验收前先静态 grep 终版代码的 autotune/num_stages/num_warps，命中即按 runbooks/autotuning 卡整改后重测，否则 geomean 不可信。
- **大 case 地板比 geomean 更能区分知识价值**：三组 geomean 差距 0.6~1.0x 时，大 shape case（本次 case 32/33/34）的地板差距（卡组 1.5~2.5x vs 无知识 0.58~0.78x）才是知识注入的决定性证据，validation_report 必须单列。

### 4.3 可追溯性抽查

随机抽 5 条卡片论断回查源码锚点，错锚/无锚 >1 条则整卡退回 P1。

---

## P5 入库与闭环

1. **双副本同步**（add 经验）：`output/<op>/<op>.md` 是源；KB 副本（`zmm_knowledge/triton-ascend-kb/ops/<op-name>.md`）是 lint 适配版——frontmatter 改 `source_family: ops`、sources[].url 换 GitCode @sha 永久链接（本地快照挂上游仓，如 `cann/ops-math@a6dca6d1.../math/add`；benchmark 挂 `sijiaali/ascendc-kernelgen-data@4cafd80...`）、`role: example`（**词表无 benchmark**），并保留 kb_finalize.py 自动注入的尾部 `okf:related` 块。**修订须同步两处**，diff 复核仅剩上述刻意适配差异。
   - **骨架是 bundle 资产**（2026-09-09 经验）：KB 卡 §8 骨架伪码必须**自包含**（完整分支、可直接翻译），**禁止本地路径死引用**（如 `见 output/<op>/triton_skeleton.py`——KB 检索组 Agent 拿到该引用后无法解析，等于 §8 缺失）；检索场景下 Agent 只能拿到卡文本，卡内一切资产须在卡内。工作流目录的 `triton_skeleton.py` 保留为 P4 回放用的完整可运行版，卡内注明其存在与一致性关系即可。
   - **后端契约沉淀为独立 runbook 卡**：P4 实测出的后端行为（访存契约/标量化陷阱/autotune 噪声等）不入 ops 卡正文，按维度入 `runbooks/{memory-sram,compute-pipeline,portability,autotuning}/` 并登记对应 index.md，ops 卡只挂链接——避免 ops 卡膨胀稀释检索相关性。
2. 卡片登记进 `ops/index.md`（条目必须带描述），再跑 `kb_finalize.py`（build/candidates/judge/aggregate/inject/verify/lint）至 0 blocker；自动注入「相关」反链（挂到同算子族 runbook 卡，如 vector 融合/CV 融合类）。
3. `validation_report.md` 摘要回写卡片「复现性声明」节，形成「来源 → 卡片 → 验证证据」闭环。

---

## 目录约定

```
ascendc_to_okf_workflow/
├── README.md                              # 本文档（工作流主文档）
├── okf_operator_card_template.md          # P3 卡片模板
├── ascendc_triton_semantic_mapping.md     # P2 映射规则速查表
├── backend_facts/                         # 后端事实库（P2 必查、P4 preflight 依据）
│   ├── backend_contracts.md               # 后端契约卡：正确性/性能/用法三级契约 + preflight 清单
│   ├── compiler_options.yaml              # 17 个 NPU 编译选项事实表（按算子类别速查）
│   ├── dtype_support.md                   # dtype 支持与标量化陷阱（含 preflight 增补项）
│   ├── hardware_platforms.md              # 平台硬件规格（UB/L0/L1/核数/对齐，target spec 源码抽取）
│   ├── tl_op_lowering.md                  # tl op → lowering 路径与代价分级（T1~T4）
│   ├── auto_passes.md                     # 编译器自动行为清单（"不要手写什么"）
│   ├── perf_patterns.md                   # 官方性能模式（care_padding/multibuffer 条件等）
│   ├── extension_apis.md                  # al/bl 扩展 API 速查（❌不可达的官方降级路径）
│   ├── operator_class_patterns.md         # 三类算子官方范式（Vector/Cube/CV 融合）
│   └── cann_extraction_decision_guide.md  # 【提取会话首查】CANN 提取决策导则：抽取顺序(host档位>kernel流水)/四层归责/三大死局反制/simulator红线
└── output/<op_name>/                      # 每个算子的过程产物
    ├── manifest.md                        # P0
    ├── extraction.md                      # P1
    ├── mapping.md                         # P2
    ├── triton_skeleton.py                 # P2 骨架伪码
    ├── <op_name>.md                       # P3 正式卡片
    └── validation_report.md               # P4
```

## 执行建议

- **单算子全流程**可走一个 Agent 串行执行（P0→P5），但 **P2 转化判定** 与 **P4 回放验证** 建议独立会话执行，避免抽取偏差自我强化。
- 批量处理 `src/mhc/` 下 9 个算子时，建议每算子一个并行实例跑 P0~P3，P4 统一串行（共享生成环境，防资源争抢）。
- mhc 这批算子含大量 Reg 级 gather/scatter（如 `mhc_sinkhorn_arch35.h`），预期 ❌ 不可达项较多——卡片价值恰在于把这些点显式写出，避免生成 Agent 在 Triton 中空转尝试。
- **backlog 复用**（add 经验）：elementwise/reduce 家族的 §4/§5 事实可直接查 `model/opbase-master-src-op_common/src/op_common/atvoss/{elewise,reduce}/` tiling 实现，不必再标「惯例推断」；批量跑 level1 算子前建议先建 benchmark↔KernelWiki artifact 覆盖对照表。
