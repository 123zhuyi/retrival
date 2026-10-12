---
schema_version: okf.v1
kind: runbook
type: operator_class_patterns
source_family: curated
title: 三类算子官方开发范式（Vector / Cube / CV 融合）
description: zh programming_guide 三篇的蒸馏：按硬件执行单元划分的算子开发范式、调参检查单与参照实现索引。P2 骨架选型按本卡定范式，P4 性能排查按本卡检查单执行。
tags: [vector, cube, cv_fusion, programming_pattern, tiling, grid, attention, matmul]
confidence: inferred
status: draft
created_at: "2026-09-11T00:00:00Z"
updated_at: "2026-09-15T00:00:00Z"
---

# 三类算子官方开发范式

> [DOC] 数据源：zh `programming_guide/{vector,cube,cv_fusion}_operator.md`。
> 配套：分核/UB/对齐通用规则见 `backend_contracts.md` C-005/C-010/C-011；扩展 API 见 `extension_apis.md`。

## 0. 算子归类判据（P0/P2 第一步）

| 类别 | 判据 | grid 基准 |
|---|---|---|
| Vector | 无 `tl.dot`；逐元素/归约/cast/gather-scatter/小融合 | `num_vectorcore` |
| Cube | 以 `tl.dot`/matmul/batch-mm 为主负载，后处理很轻 | `num_aicore`（Cube 数） |
| CV 融合 | `tl.dot` + 明显 Vector 后处理（softmax/归约/跨核同步） | `num_aicore`，Vector 按 1:2 协同 |

后处理轻重是 Cube 与 CV 融合的分界：轻（bias/scale/activation）归 Cube，重归 CV。

## 1. Vector 算子范式

**简单算子骨架**（persistent-core，固定范式）：

```python
@triton.jit
def kernel(x_ptr, ..., n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); num_core = tl.num_programs(0)
    for blk in range(pid, tl.cdiv(n, BLOCK), num_core):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, f(x), mask=mask)
```

**复杂算子（离散访存/多输出/长 hidden）组织法**：
1. 外层任务按物理 Vector 核切分，每 program 负责一段 indices/token；
2. hidden 维按 UB 容量切 `BLOCK_X`（预留 double buffer + 索引 + 临时张量空间）；
3. `SUB_BLOCK_SIZE` 合并小粒度离散任务：批量搬入 UB 组织成连续块，减少 GM 标量访存；
4. UB 内局部数据用 `tl.insert_slice` 合并、`tl.extract_slice` 取子块；
5. 多重 mask（index/column/边界）分别命名，只在 load/store 处组合。

**UB 预算经验式**（[DOC] 官方给出）：
```python
block_x = round_up(min(num_columns, max_block_x), 16)
sub_block = max((ub_budget - block_x*elem_bytes) // (block_x*elem_bytes + index_bytes), 1)
```

**调参检查单**：① 索引/长度/偏移用 int32（dtype_support §2）；② BLOCK 在 UB 内尽量大；③ grid 远大于物理核数 → 固定核数；④ 离散访存能否"批量搬 UB 后 UB 内选择"；⑤ 尾轴 32B 对齐。

## 2. Cube 算子范式

**核心**：M/N/K 三维 tile 设计；K 维循环 load A/B 子块，`acc = tl.dot(a, b, acc)`，
fp32 accumulator，出循环后 cast + mask 写回（标准 matmul 骨架见
zh examples/05，与 GPU 版同构，差异在 grid 与 tile 约束）。

**调参检查单**：① `BLOCK_M/N/K` 满足 UB/L1 容量（hardware_platforms.md：L1 512KB，L0 系列）；
② K 循环开 `multibuffer` 形成搬运/计算流水；③ 后处理轻 → 留在 Cube 类，重 → 转 CV 融合组织；
④ 输出 tile 数 ≫ Cube 核数 → 核内循环处理多 tile，或 `TRITON_ALL_BLOCKS_PARALLEL=1`（C-004）。

**复杂 Cube（attention/batched/grouped/不规则）拆解顺序**：
1. 先抽纯矩阵乘核心（每次 dot 的 tile shape/dtype/累加 dtype）；
2. 不规则访存（KV cache 低维离散）→ 先按连续维搬 UB，再转置/`insert_slice` 重组布局——
   直接二维 load 可能退化标量访存；
3. 归约/归一化（max/sum/exp）属于 Vector 逻辑 → 转入 CV 融合思路；
4. 长 K/长序列设内层循环，控制单次片上占用；
5. autotune 管理多组 `BLOCK_M/N/K` + `multibuffer` 组合。

## 3. CV 融合算子范式

**目标**：消除 kernel 边界和 GM 往返。原则：Cube 产出二维 accumulator，Vector 在**同一 tile 内**
做逐元素/小规约后处理；Vector 需跨多个 Cube tile 共享状态时才引入同步/workspace/拆 kernel。

**子块切分范式**（Cube 输出拆给 2 个 Vector 子核）：

```python
acc = tl.dot(a, b, acc)
SUB_M: tl.constexpr = BLOCK_M // 2
for s in extension.parallel(0, 2, bind_sub_block=True):
    acc_sub = extension.extract_slice(acc, (s*SUB_M, 0), (SUB_M, BLOCK_N), (1, 1))
    acc_sub = tl.where(acc_sub >= 0, acc_sub, 0.01*acc_sub)   # Vector 后处理
    tl.store(c_sub_ptrs, acc_sub.to(tl.float16), mask=c_mask)
```

**复杂 CV（attention 类）五层组织法**：① 主计算层（哪些必须走 Cube）→ ② Vector 后处理层
（能否同 tile 内完成）→ ③ 访存重排层（离散 KV/MoE/短尾轴：UB 内 insert_slice/转置/借轴转置）
→ ④ 流水同步层（`multibuffer`、`set_workspace_multibuffer`、`tile_mix_vector_loop`、
`tile_mix_cube_loop` 选项探索 C/V 重叠）→ ⑤ 分核层（grid=Cube 核数，勿沿用 GPU 大 grid）。

**attention 类递进开发顺序**：非 causal + 短序列 + 小 head_dim 跑通 → causal mask 分阶段 →
长序列 KV block 循环 → `m_i`/`l_i` 数值稳定 softmax → 大 HEAD_DIM 的 workspace/子块切分 →
KV cache 离散索引 load 重排。

**调优检查单**（看 profiling 三方时间占比）：
- Cube 等 Vector → 减 Vector 后处理粒度，或开 `enable_hivm_auto_cv_balance`；
- Vector 等搬运 → 查离散访存、尾轴 padding、multibuffer 配置。

## 4. 第四类：状态常驻递推核（chunk recurrent / linear-attention state 族）[EXP]

> 判据：每 task 有大块**常驻** tile（状态矩阵 h [K,V]），task 内为**长串行循环**
> （chunk 间递推），通常含 `tl.dot`（形态像 CV 融合，行为不是）。官方三类范式不覆盖
> 本类；以下全部为该族实测沉淀（chunk_gated_delta_rule_fwd_h 三组生成实验 + MCTS 0901，
> 详见 `output/chunk_gated_delta_rule_fwd_h/generation_retrospective_20260915.md`）。

### 4.1 与官方范式的关键差异

| 维度 | 官方 CV 融合范式 | 状态常驻递推核 [EXP] |
|---|---|---|
| grid | persistent-core = num_aicore | **一块一 program**（C-011 适用范围修正：persistent 外层循环撑爆 UB，三独立实测均一块一 program 胜出） |
| UB 预算 | 解析估算 | 解析不可靠（hardware_platforms §3.2 校准警告）；**实测标定规则：BLOCK_K·BLOCK_V ≤ 2048 元素**（A1：cap 4096/8192 反亏——latency-bound 下 program 数减少更慢） |
| 流水 | multibuffer 默认获益 | 仅作用于内层 chunk 循环；外层领任务循环被流水会复制大块 tile（机制假设，未验证） |

### 4.2 已证杠杆清单（Phase 4 应作为待试假设注入，[EXP] A1 2026-09-09/10；B4 2026-09-15 二次独立复现）

| 杠杆 | 收益（A1 → B4 复现） | 要点 |
|---|---|---|
| svf host permute-view | **+177% / +173%**（A1/B3） | states/final_state 内部恒 [K,V] 连续写，state_v_first 由返回 permute 视图消化；转置 store 标量化慢 30~400 倍。两法并列：kernel 内小 tile `tl.trans` 连续 store（A2 实测小 tile trans 比千级标量跨步 load 快 2.4-2.8x，bf16-svf1 组 3.84 vs 2.83 占优；B4 全局对照被拒 2.3365<2.3456——分组上下文优势，无全局收益，两态闭环） |
| k 条件预转置 | **+21.1% → B4 +32.7%** | NT≥3 或 (NT==2 且 CS≥32) 时 host `k.permute(0,2,3,1).contiguous()`，tt 维 stride=1；NT=1/CS=16 小 case 反亏（~10μs 固定拷贝）。50-case 标定阈值，勿恒开 |
| g 整块 load + 片上 select | +16.9% → B4 +13.8%（gm=1 case） | [BT,HVP2] 连续块 load（hv 维连续）+ `tl.sum(x*(hh==hv),1)` 选择，替代跨步 load + 标量 g_last load，值逐位等价 |
| h-无关 load 前置 | +4.5% → B4 +5.4% | 循环体内与 loop-carried h 无依赖的 load 全部先行发射，与上轮末尾 h 依赖链流水 |
| fp32 原生 dot | fp32 组 1.4-2.0 vs 仿真 0.8；**B4 复现：0.80→1.75** | `tl.dot(..., input_precision="ieee")`（tf32 达不到容差）；**bf16x3 3-dot 仿真非必要**——B3 反例：上下文失败后误写「原生 fp32 dot 不可用」，17 个 fp32 case 全亏 |
| exp2 直译 | 等价 | `tl.math.exp2 ≡ ×ln2+exp`（实现侧差 ≈0，两态已闭环）——按标杆 golden 直译即可 |

> B4（v2 卡+KB，2026-09-15）按 v2 卡 §10 清单逐一「假设→实测→采纳」：②③④ 全部
> 采纳且收益与 A1 一致（1.4674→2.3456），⑤ 实测拒绝——**杠杆清单的可移植性首次
> 跨实验验证**（A1 的优化资产经卡片传递后在独立生成会话中复现）。

### 4.3 勿重试清单（[EXP]，含双独立命中）

1. **loop-carried tile 软件流水**：scf.for iter_arg memref 类型不匹配必崩（A1、B3 双命中）
2. k 侧门控移入 k_t 缩短串行链：双舍入复利累积，max_rel 54 精度失败（A1）
3. 恒 k 预转置（无 NT/CS 门控）：小 case 固定拷贝反亏（A1：1.9432→1.7654）
4. mask 全除：越界读崩溃 + 墙钟噪声下收益不可证（A1）
5. 全局 multibuffer=False：净损失（B3；与 compiler_options 默认 true [SRC] 一致）
6. UB cap 4096/8192（放大 BV 减切分）：latency-bound 下 program 数减少反而更慢（A1：case50 200→229→312μs）
7. **「失败上下文结论」写成禁令**（C-012 变体）：B3 把「本上下文 fp32 dot UB 爆」泛化为
   「原生 fp32 dot 不可用」——A1/A2 双反证。上下文失败只能标 ⚠️ 两态，不得升格禁令
8. **stride 参数全量 constexpr 化**：非确定性 NaN 误编译 + UB 需求爆炸（B4 opt_iter_0
   三变体 bisect 实证；标量 constexpr 安全）——C-017

### 4.4 语义不变量（benchmark golden 为准，源码语义不可直接照搬）

- g/gk **同为 exp2/log2 域**（golden `torch.exp2`）——AscendC 源码 g 为自然对数域，
  以 golden 为准（B3 report 决策#1）；`g_last/gk_last` 必须取 chunk 内 **actual−1** 行。
  **⚠️ 族内不通用**：Gdn2Fwd（94）golden 恰为自然对数域 `exp`（94_Gdn2Fwd.py:62；
  CANN doc/kernel Exp 指令双佐证）——**每个算子必须独立仲裁 g 域，勿从族卡片照搬**。
- 大 shape 安全：int64 仅限标量块基址（dtype_support 4e）；50-case 小 shape 无感，
  MCTS 大 shape（T≤32768）必需

### 4.5 跨算子适用范围校准（2026-09-15，Gdn2Fwd 双臂对照 [EXP]）

> 证据：`op_94_Gdn2Fwd_20260915_1713_7285`（卡片臂：gdn2_fwd.md 唯一知识源，
> template 全禁用，15.095x，gen2/opt3）vs `op_94_94_Gdn2Fwd_20260914_1219_8124`
> （template 臂：linear-recurrent 模板 + L1.1-1.9 约束，38.2648x，gen1/opt11）——
> 同 benchmark 同 50 case，framework 基线两边一致（7.84 vs 7.73ms），impl 逐 case
> 可比：**template 臂 50/50 全胜，geomean 快 2.54x**。完整复盘见
> `output/gdn2_fwd/generation_retrospective_20260915.md`。

1. **2048 元素 UB 标定规则是 chunk 结构算子的标定，不可外推到逐 token 递推核**。
   规则来自 op-85（UB 内有 [BT,·] chunk 块 + 状态）；逐 token 路线 B 瞬态足迹小
   （无 chunk 块），BV 可大幅放宽：template 臂实测 [128,64] fp32 状态（8192 元素，
   32KB）编译通过且为最优点，cap=32 劣化 24%（L1.8）。卡片臂把 BV 压到 16
   （K=128 时）→ NVT 多达 8、K 侧向量 8 倍重复 load，是 2.54x 差距的主因之一。
   **规则：BV 上限按 kernel 结构两态——chunk 结构 2048 起点；逐 token 结构直接
   两态 [BK,64]/[BK,BV_full]，勿默认继承 2048。**
2. **「整块 load + 片上 select」杠杆在逐 token 递推核上被证伪**（卡片臂 opt_iter_0
   ×0.4707：onehot-select 开销 + 短 T 按 chunk 上取整空转；GM 小向量访存不是该
   结构的瓶颈）。该杠杆保留给 chunk 结构算子（op-85 双证仍有效）。
3. **「h-无关 load 前置」可迁移但收益缩水**：+2.2%（94，depth-2 逐 token 预取）
   vs +4.5~5.4%（op-85）。template 臂的强化版「六张量 cur/nxt 寄存器轮换
   double-buffer + 原始 dtype load」实测 ×1.363（opt5）——**逐 token 递推核的
   预取应以寄存器轮换 double-buffer 为首选形态**（新增族杠杆，[EXP] 94 opt5）。
4. **逐 token 递推核的性能天花板是 per-token pipe 同步**（template 臂 L1.1：
   BAR+SET/WAIT ≈56% 占比），长 T case 须并列评估 chunked/WY 骨架 + 成本模型
   dispatch（template 臂：dispatch 组 ×1.88、K1 直读+cumsum ×1.45）。卡片臂在
   假设阶段否决 WY（op 量对消分析）被 template 臂实测推翻——**[DERIVED] 定量
   否决的权重 < 同结构实测；WY 路线应作为骨架层双路线之一而非 Phase 4 假设**。
   **实现范式已单独立卡：`backend_facts/chunk_wy_paradigm.md`（2026-09-17）**——
   代数结构、子块参考点防 exp 上溢、BT=32/K1 直读+cumsum/dispatch 守卫实测
   参数、五条勿重试（含「op 量对消否决 WY」）。
5. **masked 2D 状态 store 静默丢单元格**（template 臂 L1.3 [EXP]：被 mask 单元格
   不写且不报错、读回脏值）——但卡片臂同款 masked 2D store 50/50 通过（反例）。
   判**上下文相关 ⚠️**：写 [BK,BV] 状态优先 1D 连续 flat 写回或 padded 全量
   store+host slice；用 masked 2D 时须 verify 全量 case（勿升格禁令，C-012 规则）。
6. exp2 换底 + 标量 constexpr 在 94 上下文一致轻负（×0.9793）——两态闭环，
   不作为默认建议。

### 4.6 骨架复用纪律（三次同类回归的教训）

骨架的任何重写（哪怕一行）必须 diff 已验证前作：hv→hk（v2 初版）、BV 守卫
max/min/while（v2 初版）、零初始化丢 `.to(tl.float32)`（Gdn2Fwd v1）——三次
回归全是「重写已验证代码时丢细节」，各耗 1 轮 gen 迭代。**规则：骨架演进用
增量 patch，不用整段重写；每处与前作的差异须能在 changelog 里说出理由。**

## 5. 参照实现索引（P2 骨架的直接模板）

| 场景 | 位置 |
|---|---|
| vector add / 基础范式 | zh `examples/01_vector_add_example.md`；`third_party/ascend/tutorials/01-vector-add.py` |
| fused softmax / layernorm | zh `examples/02`、`03` |
| fused attention（FA v2） | zh `examples/04_fused_attention_example.md` |
| matmul | zh `examples/05`；`third_party/ascend/tutorials/03-matrix-multiplication.py` |
| gather/scatter（Megablocks/binned/padded MoE） | triton-ascend-ops `tutorial/best_practice/004~006` |
| decode grouped attention（CV 复杂） | triton-ascend-ops `tutorial/best_practice/002-decode_grouped_attention.py` |
| fused cat+slice+conv1d | triton-ascend-ops `tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md` |

## 维护记录

- 2026-09-11 初版：三篇 programming_guide 全量蒸馏；参照实现索引含仓外链接
  （github.com/Ascend/triton-ascend-ops），使用时需确认可访问性或找本地镜像。
- 2026-09-15 新增 §4「状态常驻递推核」第四类范式（原 §4 参照索引顺延为 §5）：
  chunk_gated_delta_rule_fwd_h 三组生成实验（A1 KB 2.3535x / A2 仅 A 卡 1.6873x /
  B3 B 卡+KB 1.3251x，同机同套件）+ MCTS 0901 的 [EXP] 全量沉淀——一块一 program
  grid、BLOCK_K·BLOCK_V≤2048 UB 标定、六个已证杠杆、七条勿重试、golden 语义优先原则。
- 2026-09-15 **B4 复跑回流**（v2 卡+KB，2.3456x ≈ A1）：§4.2 杠杆②③④ 二次独立
  复现（+32.7/+13.8/+5.4%），⑤ svf trans-store 全局拒绝（两态闭环）；fp32 ieee dot
  复现（0.80→1.75）；§4.3 增第 8 条（stride constexpr 化，C-017）。
- 2026-09-15 **Gdn2Fwd 双臂对照回流**：新增 §4.5（跨算子适用范围校准——2048 规则
  限 chunk 结构、整块 load+select 杠杆逐 token 证伪、寄存器轮换 double-buffer 新杠杆、
  WY 应作骨架层双路线、masked 2D store 上下文相关）与 §4.6（骨架复用纪律：三次同类
  回归）；§4.4 增「g 域族内不通用」警告（94 golden 为自然对数域 exp，与 85 相反）。
- 待办：`examples/` 7 篇的完整实现未逐篇蒸馏（作为骨架模板直接引用即可，暂不需要再抽）。
