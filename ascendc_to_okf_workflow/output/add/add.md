---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: Add（broadcast 逐元素加 + alpha）Ascend C → Triton 语义转化卡
description: 基础广播 elementwise 算子，aclnn 语义 out = self + alpha × other（GEIR kernel 本体 y = x1 + x2）。AIV-only，kernel 侧调度委托 CANN base 库（BroadcastSch 本体不可得），但分核/BLOCK 机制有 op_common 源码锚点（blockFormer 均分、preferMultiCore 回调、256B 对齐的 maxElemNum 推导、DimensionCollapse 合轴）。in-repo 可抽取资产 = 四类计算 DAG 的精度策略（fp16/bf16 升 fp32 相加、RTNE 回写）与 aclnn 层 alpha 四策略分解。Triton 转化天然单 kernel：alpha 融为 runtime 标量、广播用 expand-stride 地址折算，Ascend C 侧调度工程（schMode/Axpy 分解/Contiguous 中转）全部不需要。
tags: [add, broadcast, elementwise, alpha, axpy, aiv_only, fp32, fp16, bf16, ascend950, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: Triton-Ascend-KernelWiki/kernelwiki/artifacts/ops-math/math/add/（KernelWiki 本地 artifact，ascend950 only）
created_at: "2026-09-03T00:00:00Z"
updated_at: "2026-09-04T00:00:00Z"
---

sources:
  - url: Triton-Ascend-KernelWiki/kernelwiki/artifacts/ops-math/math/add/
    role: primary
  - url: ascendc-kernelgen-data/npu_benchmark/level1/3_Add.py
    role: benchmark

# Add（broadcast 逐元素加 + alpha）

GEIR Add：逐元素二目加 + numpy 广播（README L20-28）。**AIV-only**（TilingPrepare 取 `GetCoreNumAiv()`，add_tiling_arch35.cpp:182），kernel 侧调度器 BroadcastSch 本体不可得，但分核/BLOCK 机制已在 op_common base 库源码（`atvoss/broadcast/broadcast_tiling*.cpp`）确认（§4/§5）——**半壳算子**：kernel 本体是 4 路 dtype 分派壳（add.cpp:24-51），计算 DAG（add_dag.h）、aclnn 层 alpha 分解策略（aclnn_add.cpp:371-466）与 base 库 tiling 实现在仓内，是主要语义资产。benchmark 标杆为 `3_Add.py`（torch.add(x, y, alpha=alpha)）。

## 1. 功能与公式（详设 §1.1）

- aclnn 语义（benchmark 采用）：**out = self + alpha × other**（aclnn_add.cpp:318 起；alpha 必选标参）
- GEIR kernel 语义：y = x1 + x2（README L24-28；无 alpha）
- 广播：标准 numpy 规则，输出形状 = broadcast_shapes(x1.shape, x2.shape)（infershape 委托 base 库 InferShape4Broadcast，add_infershape.cpp:17-23）
- **计算顺序：alpha 先乘 other 再加 self**（torch.add 契约），不是 (x+y)·alpha

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（GEIR，add_def.cpp；⚠️ README 参数表含 INT16/FLOAT64/COMPLEX128/STRING 为文档过期，以 def.cpp + add_binary.json 为准）

| 参数名 | 方向 | dtype | 约束 |
|---|---|---|---|
| x1 | 输入 | BF16/FP16/FP32/INT32/UINT8/INT8/INT64/BOOL/COMPLEX32/COMPLEX64 | ND，动态 rank/shape（def.cpp:24-35, 64-70） |
| x2 | 输入 | 同 x1；跨 dtype 仅允许 FP16×FP32 / BF16×FP32 四种组合（add_tiling_arch35.cpp:37-43） | ND |
| y | 输出 | 同 x1；mix-dtype 组合时必须是 FP32（add_tiling_arch35.cpp:60-67） | ND |

属性：**GEIR 无属性**；alpha 仅存在于 aclnn 层（CheckNotNull 校验，aclnn_add.cpp:77-81；cast 合法性校验 L201-228）

### aclnn 层 alpha 消化策略（alpha ≠ 1 时四路径，aclnn_add.cpp:371-466）

1. alpha==1 → 直调 `l0op::Add`（mix-dtype 且 alpha==1 走 L0 混合 kernel，L352-365）
2. `IsSupportAxpy(promoteType)` → **`l0op::Axpy(self, other, alpha)` 单 kernel 融合**（L404-421）
3. RegBase 平台 `IsSupportAxpyV2` → alpha 张量化（L422-442）
4. 兜底 → `Mul(other, alpha)` + `Add` **两 kernel**（L443-466）
5. 输出统一 Cast + ViewCopy（支持非连续 out，L470-476）

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 锚点 |
|---|---|---|
| 支持 NPU | GEIR 仅注册 ascend950（def.cpp:72）；README 产品表另列 A3/A2（走其他 kernel 仓） | README L6-15 |
| dtype | 见 §2；binary 实例 13 个（add_binary.json），shape=[-2] rank 无关 | §3 |
| format | ND，FormatAgnostic | add_binary.json |
| 确定性 | 天然确定（无归约/无 atomic/无跨核共享） | — |
| 交付 | aclnn（Add/Adds/AddV3/Inplace*）+ GEIR 图模式 + AICPU 兜底 | op_api/、op_graph/、op_kernel_aicpu/ |
| workspace | GEIR kernel 无（add_tiling_arch35.cpp:143-146）；aclnn 层按需（Contiguous/Cast 中转） | §5 |
| tilingkey | 仅 schMode 一维（add_struct.h:21-23），枚举在 base 库 | §4.1.3 |

## 4. 分核策略（详设 §4.1.1 → Triton grid）

- Ascend C 事实（源码锚点，op_common base 库）：coreNum = AIV 核数（add_tiling_arch35.cpp:182）；官方两级切分——UB 内切分（ubFormer/ubOuter/ubTail，broadcast_tiling.cpp:494-536）+ 核间均分 `blockFormer = ceil(fusedProduct/coreNum)`、尾核 blockTail（L565-588）；**preferMultiCore 回调**：核数占不满时按 CACHE_LINE 步长缩 BLOCK 直到铺满（L569-586）；连续输入先 DimensionCollapse 合轴、合轴后 ≤8 维（L151-269, 632-634）
- **Triton 决策**：
  - 任务轴：输出展平 1-D，`NUM_BLOCKS = cdiv(NUMEL, BLOCK)`
  - 核分配范式：
    ```python
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)   # grid = AIV 物理核数
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        ...
    ```
  - 语义等价说明：Ascend C「每核 blockFormer 块 + 尾核 blockTail」静态均分 ≈ Triton「grid-stride 循环 + 尾块 mask」动态轮询；preferMultiCore 策略由 autotune 小 BLOCK config 覆盖（§10-6）

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### tilingkey → constexpr

schMode（语义已破译）：`innerKey = computeKey(恒1)×OFFSET + scheduleKey`（broadcast_tiling.cpp:604-605），scheduleKey 仅区分 UB 内轴数 ≤5（NDDMA 搬运上限）与否（100/1100，broadcast_tiling_base.cpp:28-34）——与 Triton 无关。重定义 `SAME_SHAPE: tl.constexpr`：1 = 等形直通，0 = 通用广播（stride 折算）。dtype 由指针类型推导天然分派（13 个 binary → 1 个 Triton kernel）。

### BLOCK

Ascend C 官方推导（broadcast_tiling_base.cpp:36-45）：`maxElemNum = (ubSize - extraSize) × BITS / bufferDivisor`，按 256B 重复粒度对齐。Triton 候选 {1024, 2048, 4096, 8192}，autotune key=["NUMEL", "SAME_SHAPE"]；UB 预算：BLOCK=8192 fp32 时 x/y/out 三 tile 96KB ≤ 192KB ✅。

## 6. 片上内存预算（详设 §5）

- Ascend C：全 UB 队列进出（MemCfg=LEVEL_2，add_dag.h:39/56/72/87），无 GM workspace
- Triton：BLOCK×3 tile，中间零驻留；workspace = 0（优于 aclnn 层 Mul+Add 兜底的 GM 往返）

## 7. 流水编排与 Triton 可达性（详设 §6）

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| BroadcastSch/DAGSch 自动排布 | base 库（调度器本体仍不可得，⚠️ inferred） | ❌ | 不需要：load→计算→store 编译器自动流水 | 无 |
| 连续输入合轴 DimensionCollapse / 非连续 KEEP-MERGE-ABSORB 计划（broadcast_tiling.cpp:151-333, 344-492） | op_common 源码 | ✅ 语义替代 | wrapper 侧 `expand` + stride 折算天然覆盖（等形 case 合轴后 = SAME_SHAPE 直通）；Triton 无需显式合轴 | 无 |
| CopyInBrc 广播 DMA 展开 | base 库 | ❌→替代 | expand-stride 地址折算（§8 骨架） | 等价；最内维 stride=0 的重复 load 有带宽浪费风险，见 §10 优化项 |
| Contiguous 中转 | aclnn_add.cpp:358-364 | ❌→不需要 | wrapper 断言连续（benchmark 输入天然连续） | 少一次 GM 拷贝 |
| Cast/ViewCopy 输出中转 | aclnn_add.cpp:470-476 | ❌→不需要 | 直写 out | 少一次 GM 拷贝 |
| aclnn 四策略分解（Mul+Add 两 kernel） | aclnn_add.cpp:443-466 | ❌→不需要 | alpha 融为 runtime 标量，单 kernel | 少一次 GM 往返，优于兜底路径 |
| AddWithCastCompute 的 fp32 提升 + RTNE | add_dag.h:48-51 | ✅ | `.to(tl.float32)` → 计算 → `.to(out_dtype)`（RTNE 为默认舍入） | 等价 |

### 广播坐标折算（通用路径核心）

输出连续 ⇒ 线性 offs 逐维 div/mod 分解坐标，按各输入 expand 后 stride 折算：`in_off = Σ_d coord_d · in_stride_d`（广播维 stride=0 自动复用）。最内维 out stride=1。≤4 维（benchmark 域），扩维加 stride 参数对 + 静态分支。

## 8. Triton 骨架（生成用伪码）

完整骨架见 `output/add/triton_skeleton.py`。核心结构：

```python
@triton.autotune(configs=[BLOCK ∈ {1024,2048,4096,8192}], key=["NUMEL", "SAME_SHAPE"])
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, alpha, NUMEL,
               xs0..xs3, ys0..ys3, os0..os2,          # expand 后 stride + out 外维 stride
               NDIM: tl.constexpr, SAME_SHAPE: tl.constexpr, BLOCK: tl.constexpr):
    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs = blk * BLOCK + tl.arange(0, BLOCK);  mask = offs < NUMEL
        if SAME_SHAPE:  x_off = y_off = offs
        else:           # 逐维 div/mod 分解坐标 → stride 折算（broadcast 维 stride=0）
        x = tl.load(x_ptr + x_off, mask=mask);  y = tl.load(y_ptr + y_off, mask=mask)
        res = x.to(tl.float32) + alpha * y.to(tl.float32)   # fp32 中间 + alpha 先乘 y
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)
```

wrapper 要点：grid=AIV 物理核数；`out_shape = torch.broadcast_shapes(x.shape, y.shape)`；`xe = x.expand(out_shape)` 取 stride；等形（含 1D）走 SAME_SHAPE=1 展平直通；断言同 dtype ∈ {fp32, fp16, bf16} + contiguous；alpha 经 `float(alpha)` 传入。

## 9. 精度标杆与验收（详设 §测试设计）

- **标杆**：benchmark `3_Add.py` torch golden（`torch.add(x, y, alpha=alpha)`，L68）+ evaluator 默认阈值（correctness.py:34-39）：fp32 rtol=2⁻¹⁰/atol=2⁻¹⁶/max_abs 1e-2；fp16 2⁻⁹/2⁻⁹/1e-1；bf16 2⁻⁶/2⁻⁶/1.0；性能标杆 = evaluator 实测参考
- **验收域 = 3_Add.json 41 条 case**：
  - 广播形态：1D/2D/3D 等形、[M,1]/[1,N]/[N] 单向广播、4D×2D 尾维广播、[1,C,1,1] 逐通道广播、大 2D [4096,8192]/[8192,4096]（后两者 + 等形是性能关键 case）
  - dtype：fp32/fp16/bf16 同型（Ascend C dtype 域子集，无风险）
  - **alpha 大量 ≠1**（0.5/2/-1/1.5/3/0.25/-0.5）——必须融合进 kernel，不得依赖 aclnn 式两 kernel 分解
- **输入值域（精度风险点）**：偶数 case x~N(μ∈[-100,100], σ∈[1,25])（大数值，fp16/bf16 与 golden 同溢出行为），奇数 case x~U(-5,5)，y~randn；alpha·y 与 x 叠加 → **fp32 中间精度必选**（对齐 add_dag.h AddWithCastCompute）
- 公式与 golden 已核对：alpha 先乘后加顺序、广播规则、RTNE 回写均一致
- 扩展参考（非验收域）：ST csv 的异 rank 高维（6~8 维）、(1,) 标量广播、625 万 1D；int/complex/bool 形态
- 源码未附标杆脚本；benchmark 侧 torch golden 即标杆

## 10. Agent actions（生成指引）

1. 识别类别：纯广播 elementwise，AIV-only，无 tl.dot、无归约；单输出。
2. 分核：输出展平 1-D + BLOCK grid-stride 循环，grid=物理核数（§4 范式）；尾块 mask。
3. **必须保留的语义点**：fp32 中间精度 + RTNE 回写（fp16/bf16 case）；`alpha·y + x` 顺序；广播坐标按 expand-stride 折算。
4. **不要复刻**：aclnn 四策略分解（一律单 kernel 融合 alpha）、Contiguous/Cast/ViewCopy 中转、schMode 分派。
5. SAME_SHAPE=1 直通路径必须保留（大 2D case 性能关键，零整数除法）；通用路径整数 div 每元素每维一次，带宽-bound 下可接受。
6. **preferMultiCore 语义**：BLOCK 初选偏大导致核数占用不足时，优先降 BLOCK（对齐 base 库 broadcast_tiling.cpp:569-586 的官方策略），autotune 会自动完成，但手工定 BLOCK 时须遵守。
7. 优化项（非语义）：最内维 stride=0 的输入（[M,1] 型）按通用路径重复 load 同地址——可改为加载 1 宽最内 tile + tl.broadcast_to 省带宽；benchmark 通过后再做。
8. dtype 不做 constexpr 分支：指针类型推导 + `.to(tl.float32)` 统一路径覆盖 fp32/fp16/bf16。
9. 验收按 §9：41 条 case 全过 + 大数值 case 精度达标；勿引入归约/atomic。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred），锚点指向 `ops-math/math/add/`，分核/BLOCK 事实补充自 op_common base 库源码（`model/opbase-master-src-op_common/src/op_common/atvoss/broadcast/`，2026-09-03）。未经硬件实测的点：BLOCK 最优值、通用广播路径 div/mod 相对直通路径的实际开销、广播重复 load 的带宽影响。残留不确定项：kernel 侧 BroadcastSch 调度器本体（op_common 拷贝未含）与 broadcast_tiling.h 头文件（CACHE_LINE 等常量值从 .cpp 反推）、aclnn 层 Axpy dtype 清单（外部常量）。验收标杆已对齐 NPUKernelBench level1（3_Add.py，41 case，含 alpha≠1 与大数值输入域）。回放验证：未做（P4 待执行，报告将补至 `output/add/validation_report.md`）。

## 维护记录

- 2026-09-18 **⚠️ 骨架毒性警告（回放实证）**：本卡 2026-09-03 的 `triton_skeleton.py`
  广播路径含 **int64 向量 + 逐元素逐维 `//` 除法链**（:73-86，违反 4e + 4a''
  双契约）——卡组 A（new repo 9-17）按骨架直出：广播 case 慢 300-500x
  （geomean 0.0779）；卡组 B（old repo 9-17）经 Phase 4 重写为**双 2D tile
  摊平**结构（无逐元素 decode）→ 0.9246。**本卡 §11 当时已把「div/mod 开销」
  列为未实测点但未标 ⚠️ 严重性、骨架仍带毒直出**——教训：「未实测」的性能
  风险写法若直接进骨架，必须把严重性标到 ⚠️ 并给保守替代。**骨架待按新契约
  翻新（双 2D tile 结构），翻新前勿作生成起点。**
