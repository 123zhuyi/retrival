---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Cumsum（沿轴前缀和，Sklansky 扫描家族）Ascend C → Triton 语义转化卡
description: 沿 dim 轴前缀和 y_i=Σ_{j≤i}x_j，M×R×N 折叠。浮点家族 8 tilingkey（oneway/twoway × R全载/UB_SS/CORE_SS/复合）+ 整型家族 3 key，全部 Reg 级 Sklansky 扇出（log2(R) 趟）；CORE_SS 档跨核 SyncAll 全局屏障 + GM(y) 读改写合并——Triton 结构性 ❌，降级为扫描线级并行（每 program 负责整条 R），M×cdiv(N,BLOCK_N) 不足时性能两态。exclusive=inclusive−x 代数恒等；reverse=后缀和两遍法 total−prefix（降序行 tile 后端存取错误——实测）。f16/bf16 升 fp32 中间域。P2.5 ①编译门实测三条硬结论：tl.cumsum→func::CallOp UB 放大 ~48×（tile 元素数硬上限 1024）；行序循环「store(carry) 后原地更新 carry」WAR 模式后端流水偶发失同步（输出前移一行、逐 run 非确定）；降序行地址 tile 存取错误。
tags: [cumsum, scan, prefix_sum, sklansky, exclusive, reverse, core_ss, syncall_cross_core, tl_cumsum_callop, ub_overflow_1024, war_hazard, descending_tile_trap, aiv_only, ascend950, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-math/math/cumsum/（CANN ops-math 本地快照，仅 ascend950 AICore；AICPU 兜底另册）
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

sources:
  - url: src/ops-math/math/cumsum/
    role: primary

# Cumsum（y_i = Σ_{j≤i} x_j 沿 dim 轴）

沿 dim 前缀和，输出 shape/dtype 与 x 相同。属性 **exclusive**（排除自身，首元 0）、**reverse**（后缀和），默认均 false（def.cpp:67-68）。**axis 是 tensor 输入**（i32/i64，ValueDepend，def.cpp:38-52）——wrapper 按标量 int 接（aclnn V1 = torch.cumsum 语义；V2 = tf.cumsum 语义才暴露 exclusive/reverse 属性，但 kernel 层两属性全支持）。仅 ascend950（def.cpp:77）；AICPU 兜底不参与本路径。

## 1. 功能与公式（详设 §1）

- 语义：`y_i = Σ_{j≤i} x_j`（inclusive）；exclusive：`y_i = Σ_{j<i} x_j`；reverse：从尾部累加。
- golden（golden.py）：np.cumsum 主线；**exclusive = roll(1)+首位置 0**；**reverse = flip→cumsum→flip**；f16/bf16 升 fp32 计算后回原型。

## 2. 原型与约束（详设 §2/§3）

| 项 | 值 |
|---|---|
| dtype | i8/i32/i64/u8/u64/f16/f32/bf16，in/out 同型（def.cpp:26-35,55-64） |
| axis | tensor 标量（i32/i64），host tiling 值依赖 |
| 属性 | exclusive/reverse OPTIONAL bool，默认 false |
| 平台 | ascend950 AICore（本卡范围） |
| ST | V1 202 case（torch.cumsum）+ V2 200 case（tf.cumsum 含属性组合）；acc=default |

## 3. 分核策略（详设 §4 → Triton grid）

Ascend C 事实：折叠 **M×R×N**（M=dim 外维、R=扫描轴、N=内维）；决策树：M 够切核→M 分核（不够→借轴 N）；再不够→**借轴 R 跨核（CORE_SS，SyncAll 协议）**（tiling:449-560）。
Triton 映射：**扫描线级 persistent-core**——每 program 负责 (m, n 条带) 的**整条 R**：
```python
NUM_TASKS = M * cdiv(N, BLOCK_N)
grid = min(NUM_TASKS, num_vectorcore)   # 连续段循环；NUM_TASKS 传上界、kernel 内按真实 nb_n 钳制
```
等价论证：官方主路径（1001/1011，M 分核 + 核内 R 全载或 UB_SS 分块进位）= 本设计逐字同构（R_CHUNK 块 + carry 循环 = UB_SS 语义）。

## 4. BLOCK / tile（①编译门实测硬约束）

- **R_CHUNK×BLOCK_N ≤ 1024 元素**——`tl.cumsum` lowering = `func::CallOp` 库调用（tl_op_lowering.md:70 [SRC]），UB 放大 ~48×，超限即 bishengir "ub overflow, requires 6324224 bits"（恒定需求，与 tile 尺寸解耦）。普通 elementwise 的静态 UB 预算法对含 scan CallOp 算子**失效**。
- 候选 (R_CHUNK, BLOCK_N) ∈ {(16,64), (32,32), (8,128), (4,256), (2,512)}；尾轴 32B 对齐（C-010）。
- 官方 twoway 折叠（N≤VL/4 时多行折进向量组）= 纯布局工程，不翻译；其要解决的「窄 N 向量利用率」问题 Triton 侧由 BLOCK_N 候选承接（两态）。

## 5. 计算语义 → Triton（核心映射）

```python
# 前向：R 分块，块内 tl.cumsum + 块间 carry（UB_SS 同构）
s = tl.cumsum(tile, axis=0) + carry[None, :]
if EXCLUSIVE: s = s - tile          # 代数恒等：exclusive = inclusive − x
tl.store(...); carry += tl.sum(tile, axis=0)

# reverse：后缀和两遍法（纯升序寻址）
# pass1: total = Σ tile；pass2: out = total − prefix (+tile 若 inclusive)
```

| 属性 | 映射 | 注记 |
|---|---|---|
| exclusive | `inclusive − x` | 整型**精确**；fp ≤1ulp（golden 是精确移位） |
| reverse | `total − prefix (+x)` | 两遍法；fp 下 total 与 prefix 抵消误差，长 R 大动态范围两态 |

## 6. 片上内存 / 流水（详设 §5/§6）

- 官方：xBuf + fp32 cast buffer（非 fp32 输入同尺寸，oneway_sklansky.h:88-93）；CORE_SS 另 5 个 TBuf + **GM(y) 自身读改写**。Triton 侧编译器自管；tile ≤1024 元素硬约束见 §4。
- multibuffer 默认 true [SRC Options.td]；不显式编码。

## 7. Triton 可达性总表

| Ascend C | Triton | 可达性 |
|---|---|---|
| M 分核（借 N） | 扫描线级 persistent-core | ✅ 同构 |
| UB_SS 核内分块进位 | R_CHUNK 块 + carry 循环 | ✅ 同构 |
| Reg 级 Sklansky（log2 R 趟，LoadAlign/StoreAlign） | `tl.cumsum`（func::CallOp） | ⚠️ 语义等价；tile ≤1024 硬约束【实测】；性能两态 |
| twoway 折叠（MAX_N=32/MAX_FOLD=8） | 无对应 | ❌ 布局工程，不翻译（无语义影响） |
| **CORE_SS 跨核 SyncAll + GM 读改写** | 无对应 | ❌ **降级：放弃跨核并行**；M×cdiv(N,BLOCK_N)<核数时核占不满（两 kernel 分段前缀和方案登记不实现） |
| exclusive（CopyInExclusive 错位+Duplicate(0)） | `inclusive − x` | ✅（fp ≤1ulp） |
| reverse（flag=−1 镜像寻址） | 两遍法 total−prefix | ✅（降序 tile 后端错误【实测】，勿用镜像寻址） |
| Cast NONE 进 / RINT 出（fp32 中间域） | `.to(tl.float32)` / `.to(out)` | ✅ 必须保留（f16/bf16） |
| 整型三 key（NO_SPLIT/AR_SPLIT/WITH_GROUP） | 单 kernel + IS_INT constexpr | ✅（i64/u64：vcum* 标量化 [SRC]，性能两态） |

## 8. Triton 骨架（生成用伪码）

完整骨架 `output/cumsum/triton_skeleton.py`（P2.5 ①+② 通过）。结构要点：persistent-core task 循环 → (m, nb) 标量解码 → carry 零初始化（整型原生/fp 用 fp32，IS_INT constexpr 分支）→ R_CHUNK 块循环（REVERSE 时两遍）。

## 9. 精度标杆与验收

- golden：§1；容差 ST acc=default（fp32 2e-4 / fp16 2e-3 / bf16 2e-2；int 精确）。
- **必须保留**：① f16/bf16 升 fp32 中间域 + RTNE 回写；② exclusive/reverse 的代数映射（勿用显式 roll/flip——Triton 无块内行移位原语）；③ tile ≤1024；④ 全部升序寻址。
- 树形求和（tl.cumsum 内部）vs golden 顺序求和的舍入路径差：通常优于顺序，容差内，登记。
- 已知限制：numel ≥ 2³¹ 超 int32 安全域（4e，wrapper 断言拦截）；M×cdiv(N,BLOCK_N) 并行度不足时性能低于官方 CORE_SS（❌ 降级，两态）。

## 10. Agent actions（生成指引）

1. 识别类别：**scan（前缀和）**——非 elementwise；顺序依赖沿扫描轴，并行度在 M×N。
2. **先查 tile 上限再写块大小**：含 `tl.cumsum`/`tl.scan` 的 kernel，tile 元素数 **≤1024**（CallOp UB 放大 ~48×，超限编译报 ub overflow 6324224 bits）——这是本卡第一工程点。
3. **不要写降序行地址 tile**（`rows = R−1−…` 作 load/store 第一维）：后端存取错误（只写首行）。reverse 一律两遍法 `total − prefix`。
4. **不要写「store(x) 后同迭代原地更新 x」的 loop-carried 模式**（逐行 `carry+=row; store(carry)`）：WAR 失同步、逐 run 非确定。用「每迭代生成新 SSA 值再 store」的结构。
5. **必须保留**：① fp32 中间域（f16/bf16）+ RTNE；② exclusive=`s−tile`、reverse=两遍法；③ dtype 分支走 wrapper 传 `tl.constexpr`（前端不折叠 `.is_int()` 本地 if，or 链也不支持）；④ 任务解码用标量 div（勿进 mask 表达式）；int64 仅标量基址。
6. **不要复刻**：Reg 级 Sklansky 扇出、twoway 折叠、CORE_SS 跨核协议、UB_SS 的 TBuf 预算常数、AICPU 兜底。
7. 分核：`grid=min(M*cdiv(N,32), num_vectorcore)` 上界 + kernel 内 `end=min(end, M*nb_n)` 钳制（grid 先于 autotune 定，NUM_TASKS 只能传上界）；核数运行时查询。
8. i64/u64 输入：vcum* 标量化 [SRC] + i64 向量算术标量化预警——能跑但性能两态，勿写性能结论。
9. numel ≥ 2³¹ 超域：wrapper 断言 + 按 M 维切块循环调用，勿在 kernel 引 int64 向量 decode。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化，**confidence: inferred**（P2.5 降级①编译门+②正确性门已通过——64 组合全 PASS，报告见 `output/cumsum/validation_report.md`；③ simulator/③.5 board/④ IR 未执行，BLOCK 选型与瓶颈定型保持两态）。锚点指向 `src/ops-math/math/cumsum/`；后端论断经 backend_facts 检索 + 真机编译实测（tl.cumsum tile ≤1024、降序 tile 错误、WAR 失同步三条为 2026-09-17 实测新取证）。未经实测的点：BLOCK_N 优选、窄 N（twoway 场景）性能、i64 标量化实际开销、M 并行度不足场景与官方 CORE_SS 的差距。残留不确定项：AICPU 兜底触发面未定位；axis tensor 动态值场景 host 行为未确认。
