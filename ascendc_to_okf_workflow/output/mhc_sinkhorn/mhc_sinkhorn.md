---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: MhcSinkhorn（mHC 超连接 Sinkhorn 双随机化）Ascend C → Triton 语义转化卡
description: mHC 架构核心算子，Sinkhorn-Knopp 迭代把 n×n 初始混合矩阵（n∈{4,6,8}，批量 T 个）投影为双随机矩阵；AIV-only 纯向量算子，第 0 轮为逐行 softmax 初始化后交替行/列归一（分母加 eps）。Triton 转化的关键决策：T 轴分核 + [BLOCK_T,8,8] padding tile + axis 规约，Ascend C 中的 Reg gather/scatter 布局转换与 UB 规约中转全部不需要。
tags: [mhc, sinkhorn, doubly_stochastic, row_col_normalize, softmax, elementwise, reduce, aiv_only, fp32, small_tile, ascend950, mhc_sinkhorn]
confidence: inferred
status: draft
resource: src/mhc/mhc_sinkhorn/（zmm_source_retrieval 本地源码，ascend950 only）
created_at: "2026-09-03T00:00:00Z"
updated_at: "2026-09-03T00:00:00Z"
---

sources:
  - url: src/mhc/mhc_sinkhorn/
    role: primary

# MhcSinkhorn（mHC Sinkhorn 双随机矩阵投影）

mHC（超连接）架构核心算子：对批量 T 个 n×n 初始混合矩阵（n∈{4,6,8}）做 Sinkhorn-Knopp 迭代，投影到双随机矩阵流形（行和=列和=1），输出供 MhcPost 使用（README L16）。**AIV-only**（apt.cpp:26 `KERNEL_TYPE_AIV_ONLY`），无 Cube，全程 fp32。实现特征：n 极小（≤8），单矩阵寄存器可容；Ascend C 用 Reg 级 gather/scatter 做行列布局转换，属向量效率手段而非语义——Triton 中以 padding tile + axis 规约整体替代（见 §7）。

## 1. 功能与公式（详设 §1.1）

- 功能：Sinkhorn-Knopp 迭代双随机化（README L14-28）
- **实现公式（以代码为准）**——⚠️ 与 README 公式有偏差：第 0 轮不是直接归一，而是**逐行 softmax 初始化**（arch35.h:213-235）：

$$
M^0 = \mathrm{softmax}_{row}(X),\quad
M^{(k)} = M / (\mathrm{rowsum}(M)+\epsilon)\ \ (k\ge1\text{ 先行归一}),\quad
M = M / (\mathrm{colsum}(M)+\epsilon)\ \ (\text{每轮后接列归一})
$$

- 执行序（Process，arch35.h:472-551）：`softmax → 列归一 → [行归一 → 列归一]×(num_iters-1) → 写回`。共 num_iters 次列归一、num_iters-1 次行归一；**每处分母都加 eps**。
- 变量：X 输入 [T,n,n]（或 [B,S,n,n]，T=B·S）；eps 默认 1e-6；num_iters 默认 20。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`op_host/mhc_sinkhorn_def.cpp:23-49`；⚠️ README 参数名 init_matrix/h_res 与 IR 不一致，以 def.cpp 为准）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| h_res | 输入 | 必选 | FLOAT32 | ND | (T,n,n)/(B,S,n,n) | 初始混合矩阵 |
| y | 输出 | 必选 | FLOAT32 | ND | 同输入 | 双随机化结果 |
| norm_out | 输出 | IR 必选 / aclnn 可空 | FLOAT32 | ND | out_flag=1: [2·num_iters·T·n·8]；=0: [1] | 每轮归一后中间矩阵（调试） |
| sum_out | 输出 | IR 必选 / aclnn 可空 | FLOAT32 | ND | out_flag=1: [2·num_iters·T·8]；=0: [1] | 每轮行/列和（调试） |

### 属性

| 属性 | 类型 | 默认 | 约束 | 锚点 |
|---|---|---|---|---|
| eps | float | 1e-6 | 建议 1e-6~1e-4 | def.cpp:47 |
| num_iters | int | 20 | **[1,100]**（tiling 强校验） | def.cpp:48; tiling.cpp:85-89 |
| out_flag | int | 0 | {0,1} | def.cpp:49; tiling.cpp:93-97 |

### 原型约束

- 维度数仅 3/4；末两维相等（方阵）；**n ∈ {4,6,8}**（tiling.cpp:162-185）
- 全 dtype 仅 fp32，输出 dtype 必须等于输入（tiling.cpp:107-147）
- aclnn 层 normOut/sumOut 可传空指针（docs/aclnnMhcSinkhorn.md L151）
- infershape 支持 unknownRank/unknownDim（infershape.cpp:66-113）

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | 仅 Ascend 950PR/950DT | README L7；def.cpp:59 仅注册 ascend950 |
| dtype | FLOAT32 | def.cpp:25 |
| format | ND | def.cpp:26 |
| 确定性计算 | 天然确定（核间无共享、无 atomic） | Process 结构 arch35.h:465-552 |
| 交付形式 | aclnn + GEIR 图模式 | examples/; op_graph/ |
| tilingkey | = out_flag（0/1） | tiling.cpp:308-315 |

### 架构差异

不涉及（仅 arch35 一套实现）。

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C 事实：**仅 T 轴切分**（T 个矩阵互不依赖）；`tNormCore=CeilDiv(T, AIV核数)`，`usedCoreNum=CeilDiv(T, tNormCore)`，尾核单独参数（base_tiling.cpp:125-127）；host `SetBlockDim(usedCoreNum)`（tiling.cpp:327）
- **Triton 决策**：
  - 任务轴：T 轴按 BLOCK_T 分块；n 轴不切（padding 成 8 进 tile）
  - 任务块总量：`NUM_BLOCKS = ceil(T / BLOCK_T)`
  - 核分配范式：
    ```python
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)   # grid = AIV 物理核数
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        ...
    ```
  - 尾块不做正常核/尾核二分，统一 `offs_t < T` mask（简化自 tUbFactorTail/tUbTailTail 机制）

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### tilingdata → Triton 超参映射

| tilingdata 字段 | 语义（struct.h:18-31） | host 计算式（base_tiling.cpp） | Triton 对应 | 候选值/推导 |
|---|---|---|---|---|
| tUbFactor | 单循环矩阵数 | UB 预算反推，FloorAlign 8 | `BLOCK_T` | {32, 64, 128}，autotune key=["T","N"] |
| tUbFactorTail / tUbTailTail | 尾循环矩阵数 | 差值 | 尾块 mask | `offs_t < T` |
| tNormCore / tNormCoreLoop / tTailCoreLoop | 核内总量/循环数 | CeilDiv 链 | `NUM_BLOCKS` 循环 | — |
| n / nAlign=8 | 矩阵维/对齐维 | shape | `N`（运行时）+ `BLOCK_N=8` constexpr | mask `offs < N` |
| num_iters / eps / out_flag | 迭代/保护/调试开关 | 属性透传 | kernel 标量参 + `OUT_FLAG: tl.constexpr` | — |

### tilingkey → constexpr

tilingkey=out_flag → 单 kernel + `OUT_FLAG: tl.constexpr`，合并 TEMPLATE_MODE 0/1 两实例（差异仅为调试 store，编译期裁剪）。

## 6. 片上内存预算（详设 §5）

| Ascend C buffer（arch35.h:102-115） | 容量表达式（fp32） | Triton 等效 | 字节数（BLOCK_T=64） |
|---|---|---|---|
| inputQue_ ×2 | align(t·n·n,8)·4B | load tile [BLOCK_T,8,8] | 16KB |
| normOutQue_ ×2 | t·n·8·4B | 寄存器 tile m（迭代零搬运） | 16KB |
| sumColQue_ / sumRowQue_ ×2 | t·8·4B | 规约中间 [BLOCK_T,8] | 4KB |
| outputQue_ ×2 | align(t·n·n,8)·4B | 与 m 复用 | ~0 |
| mask/mid/sumBuffer | 64+256+256B | 无对应 | 0 |

- **预算合计 ~36KB ≪ UB 192KB** ✅ 大幅富余（n=8 时单片仅 ~1.2KB，Ascend C 侧 tUbFactor 也几乎总能全载，base_tiling.cpp:131-138）
- 结构性优势：Ascend C 把中间矩阵放 UB 队列迭代 2·num_iters 轮；Triton 迭代全程在寄存器 tile 上，**批内零搬运**。

## 7. 流水编排与 Triton 可达性（详设 §6）

### 流水结构

批循环 `CopyIn → softmax → 列归一 → [行归一→列归一]×(k) → Scatter → CopyOut`；全队列 double buffer 掩盖批间搬运；循环内 4 处 `SetFlag/WaitFlag(MTE3_V)` 仅在 out_flag=1 搬出中间结果时构成实质依赖（arch35.h:482-543）。

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| DataCopyPad 连续搬入/搬出 | arch35.h:125-147 | ✅ | `tl.load/tl.store` + mask + `other=0.0` | 等价 |
| Reg gather/scatter 行列布局转换（rowIndexReg 合成） | arch35.h:195-215, 426-435 | ❌ | **不需要**：直接 [BLOCK_T,8,8] tile，`tl.sum(axis=2)` 行规约 / `tl.sum(axis=1)` 列规约 | 不低于原实现（n≤8，gather 仅为寄存器排布服务） |
| 规约值经 UB 中转广播（LocalMemBar + DIST_E2B_B32 回读） | arch35.h:216-229, 358-372 | ❌ | `tl.sum/tl.max` 结果直接 broadcast | 减少 UB 往返，更优 |
| 手工 MTE3_V 事件链 | arch35.h:482-484 等 | ❌ | 丢弃，编译器自动同步 | 无 |
| 队列双缓冲 | arch35.h:107-115 | ⚠️ | 批循环 + `multibuffer=True`（编译开关） | 编译器决定掩盖质量 |
| StoreUnAlign 非对齐调试输出 | arch35.h:362-367 | ✅ | 普通 `tl.store` + mask | 等价 |

## 8. Triton 骨架（生成用伪码）

完整骨架见 `output/mhc_sinkhorn/triton_skeleton.py`（可直接翻译为 @triton.jit）。核心结构：

```python
@triton.autotune(configs=[BLOCK_T ∈ {32,64,128}], key=["T","N"])
@triton.jit
def mhc_sinkhorn_kernel(x_ptr, y_ptr, norm_ptr, sum_ptr, T, N: tl.constexpr,
                        num_iters, eps, OUT_FLAG: tl.constexpr,
                        BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr):  # BLOCK_N=8
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs_t / offs_r / offs_c;  mask = (offs_t<T) & (offs_r<N) & (offs_c<N)
        m = tl.load(x_ptr + t·N·N + r·N + c, mask=mask, other=0.0)   # [BLOCK_T,8,8]
        mx = tl.max(m, axis=2); e = tl.exp(m - mx[:,:,None])
        m = e / (tl.sum(e, axis=2) + eps)[:,:,None]                  # 行 softmax 初始化
        m = m / (tl.sum(m + eps, axis=1) + eps)[:,None,:]            # 列归一(第0轮)
        for it in range(1, num_iters):
            m = m / (tl.sum(m, axis=2) + eps)[:,:,None]              # 行归一
            m = m / (tl.sum(m, axis=1) + eps)[:,None,:]              # 列归一
            if OUT_FLAG: tl.store(norm_ptr/sum_ptr + it·stride, ...)  # 调试输出
        tl.store(y_ptr + t·N·N + r·N + c, m, mask=mask)
```

wrapper 要点：grid=AIV 物理核数；T = shape[0]（3 维）或 shape[0]*shape[1]（4 维）；out_flag=1 时 norm_out/sum_out 按 `2·num_iters·T·n·8` / `2·num_iters·T·8` 分配；断言 contiguous + fp32 + n∈{4,6,8}。

## 9. 精度标杆与验收（详设 §测试设计）

- **⚠️ 定义域超集**：benchmark（`ascendc-kernelgen-data/npu_benchmark/mhc/3_Sinkhorn.{py,json}`，NPUKernelBench kernel_generator 场景）case 的 **n 到 16**（如 [1,16384,16,16]），超出 Ascend C 950 约束 {4,6,8}——Triton 实现**不要硬编码 n∈{4,6,8}**，`BLOCK_N` 取 `next_pow2(n)`（≤16）+ mask `offs<n` 即可泛化
- **输出布局以 benchmark golden 为准**（golden `3_Sinkhorn.py` L64-88）：`norm_out=[2·num_iters, n, n, T]`（T 维在最后，**无 8 对齐 padding**）、`sum_out=[2·num_iters, n, T]`、iter0（softmax 轮）的 sum 槽位填零——与 Ascend C workspace 布局 `[2i, T, n, 8]` 不同，跑 benchmark 时按 golden 布局输出（T 维 strided store），§8 骨架中的 8 对齐布局仅对应 aclnn 形态
- **判定阈值**（npu-kernelbench evaluator 默认，correctness.py:34-39）：fp32 输出 rtol=2⁻¹⁰、atol=2⁻¹⁶、max_abs≤1e-2，另有 required_matched_ratio；workload 可覆盖
- 标杆实现：benchmark 的 torch golden（精度）+ evaluator 实测参考耗时（性能，perf_reference_source 默认 "measured"）
- 公式与 golden 已核对一致：第 0 轮行 softmax 初始化（L39）、col 和为 `Σ(prob+eps)+eps`（L44-45）、后续行/列归一分母 +eps（L50-57）
- 典型用例 shape（benchmark json）：(1,16,4,4)、(1,8161,12,12)、(1,16384,16,16)、(4,1024,4,4)、(8,2048,8,8)、(16,16384,16,16)，num_iters=20，全部 out_flag=1；另保留 Ascend C UT 形态（T 非 BLOCK_T 整数倍、out_flag=0、异常 rank/dtype）
- **源码未提供标杆脚本**（torch 参考实现需在 P4 自建：rowsoftmax 初始化 + 交替行列归一，fp64 真值）

## 10. Agent actions（生成指引）

1. 识别类别：elementwise+reduce 复合、**AIV-only 纯向量**，无 tl.dot；n∈{4,6,8} 极小矩阵批量处理。
2. 分核：T 轴 BLOCK_T 分块，grid=物理核数，§4 范式；n 不切，BLOCK_N=8 padding。
3. BLOCK 候选 {32,64,128}（UB 极宽裕，§6）；autotune key=["T","N"]。
4. **必须保留的语义点**：第 0 轮是 rowsoftmax（不是普通归一）；每处分母 +eps（4 处）；执行序 softmax→列→[行→列]×(k-1)，不可调换行列顺序。
5. **不要尝试复现** Ascend C 的 Reg gather/scatter 布局转换与 UB 规约中转——用 axis 规约直写（§7 表）。
6. out_flag=1 调试输出用 `OUT_FLAG: tl.constexpr` 分支，布局按 [2·num_iters, T, n, 8] / [2·num_iters, T, 8]。
7. 验收按 §9 双标准执行（标杆比对 + 双随机性直检）。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred），锚点指向 `src/mhc/mhc_sinkhorn/`。未经硬件实测的点：BLOCK_T 最优值、相对 aclnn 性能、fp32 迭代误差标定值。README 与 def.cpp 的接口命名偏差已按代码修正。验收标杆已对齐 NPUKernelBench（npu_benchmark/mhc/3_Sinkhorn），其 n 范围（≤16）为 Ascend C 约束超集、调试输出布局以 golden 为准（§9）。回放验证：未做（P4 待执行，报告将补至 `output/mhc_sinkhorn/validation_report.md`）。
