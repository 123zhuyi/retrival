---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: FusedBiasLeakyRelu（偏置+LeakyReLU+缩放融合）Ascend C → Triton 语义转化卡
description: y = LeakyRelu(x + bias, negative_slope) * scale 的 AIV-only elementwise 融合算子（arch35/ascend950 唯一实现，GEIR-only 无 aclnn 层）。kernel 与 tiling 本体全量在仓内（非壳算子），语义封闭；Triton 侧单 kernel 覆盖双 dtype，全程 fp32 计算域 + 末段单次 RTNE 舍入是唯一精度要点；无任何 ❌ 不可达项。
tags: [activation, elementwise, fused, leaky_relu, bias_add, fp16, fp32, arch35, ascend950, aiv_only, fused_bias_leaky_relu]
confidence: inferred
status: draft
resource: src/ops-nn/activation/fused_bias_leaky_relu/（zmm_source_retrieval 本地快照，CANN ops-nn 仓）
created_at: "2026-09-07T00:00:00+08:00"
updated_at: "2026-09-07T00:00:00+08:00"
---

sources:
  - url: src/ops-nn/activation/fused_bias_leaky_relu/
    role: primary

# FusedBiasLeakyRelu（偏置 + LeakyReLU + 缩放融合）

对输入 x 加**等形**偏置 bias（无广播），施加 LeakyReLU（负半轴斜率 negative_slope），再整体乘缩放因子 scale。纯 elementwise 三输入输出等形（2 入 1 出），**AIV-only**（kernel 仅 Vector 指令 Add/LeakyRelu/Muls + Cast，tiling 取 `GetCoreNumAiv()`，fused_bias_leaky_relu_tiling_arch35.cpp:50）。**非壳算子**：kernel 本体（192 行）与 tiling（218 行）全量在仓内，单模板类 `FusedBiasLeakyRelu<T, BUFFER_MODE>`，无 base 库依赖。**仅 GEIR 图模式入口**（无 op_api/aclnn 层），**仅 arch35/ascend950 一套实现**（def.cpp:53），全套文件为 CANNBot 生成。语义分层：GEIR kernel 本体即全部语义；aclnn 层不存在；benchmark 库未收录（验收标杆须自建，见 §9）。

## 1. 功能与公式（详设 §1.1）

- 功能：`y = LeakyRelu(x + bias, negative_slope) * scale`，兼容 mmcv `FusedBiasLeakyrelu`（proto 注释 fused_bias_leaky_relu_proto.h:39）。
- 计算公式（README.md:19-25）：

$$
y = \begin{cases} (x+bias)\cdot scale, & x+bias \ge 0 \\ (x+bias)\cdot negative\_slope \cdot scale, & x+bias < 0 \end{cases}
$$

  x/bias/y 同 shape（≤8 维 ND）；negative_slope 默认 0.2；scale 默认 1.414213562373（√2）。
- **语义分层**：GEIR kernel 本体 = 上式（属性经 tilingdata 以 runtime 标量透传，tiling_data.h:25-26 → kernel fused_bias_leaky_relu.h:93-94）；aclnn 层**不存在**；benchmark 层**未收录**，无超集偏差。
- 计算流程概述（kernel Compute，fused_bias_leaky_relu.h:146-175）：
  1. fp16 路径：`Cast(x/bias → fp32, CAST_NONE 精确)`；fp32 路径跳过
  2. `t = x + bias`（fp32 Add）
  3. `u = LeakyRelu(t, negative_slope)`（fp32；t=0 时两分支同值 0，文档 `>`/`≥` 表述差异无影响）
  4. `y = u * scale`（fp32 Muls 标量广播）
  5. fp16 路径：`Cast(y → fp16, CAST_ROUND=RTNE)` 单次舍入回写
- 锚点：`op_kernel/arch35/fused_bias_leaky_relu.h:158-169`

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`op_host/fused_bias_leaky_relu_def.cpp:24-41`）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| x | 输入 | 必选 | FLOAT16 / FLOAT | ND | 0-8 维 | 主输入，支持空 tensor |
| bias | 输入 | 必选 | FLOAT16 / FLOAT | ND | 与 x **完全相同**（无广播） | 偏置 |
| y | 输出 | 必选 | FLOAT16 / FLOAT | ND | 与 x 相同 | 输出 |

三者均 `.AutoContiguous()`（def.cpp:29/35/41）——非连续输入由框架转连续，kernel 只处理连续 ND。

### 属性

| 属性 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| negative_slope | float 可选 | 0.2 | LeakyReLU 负半轴斜率（def.cpp:42） |
| scale | float 可选 | 1.414213562373 | 整体缩放因子（def.cpp:43） |

### 原型约束

- shape 边界：dim ≤ 8（tiling_arch35.cpp:60-80）；x/bias/y 三者元素总数相等校验（L86-90）；infershape 直接复制 x 形状（infershape.cpp:32）。
- 值域边界：属性任意 float，无范围校验（L117-120 仅空指针取默认）。
- 输入间依赖：x 与 bias dtype 必须一致（L110-113）。
- 拦截校验：全部在 tiling host 侧 OP_CHECK；空 tensor（totalNum=0）合法走短路（L155-164）。
- **文档偏差**：proto 声明 dtype 含 DT_DOUBLE（fused_bias_leaky_relu_proto.h:42-46），kernel/def 无 double 路径——以 def.cpp 为准。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | 仅 ascend950（arch35） | def.cpp:53（README 产品支持表列 A2/A3 等为文档表述，以 def 为准） |
| dtype 全集 | FLOAT16、FLOAT（三参数同型） | def.cpp:26-38 |
| format | ND | def.cpp:27-40 |
| 确定性计算 | 是（纯 elementwise，无规约/atomic/workspace，WS_SYS_SIZE=0） | tiling_arch35.cpp:34 |
| 交付形式 | GEIR 图模式（REG_OP）；无 aclnn | README.md:116；examples/test_geir_fused_bias_leaky_relu.cpp |

### 架构差异（arch22 vs arch35）

不涉及——仅 arch35 一套 kernel（def.cpp 只注册 ascend950，op_kernel 仅 arch35 目录）。

## 4. 分核策略（详设 §4.1.1 → Triton grid 设计）

- Ascend C 事实：整体打平一维，按元素数静态均分 AIV 核——`blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 32B/typeSize)`、`usedCoreNum = CeilDiv(totalNum, blockFactor)`（tiling_arch35.cpp:166-170, 199）；kernel 侧 `blockOffset = GetBlockIdx()*blockFactor`、末核少载/空载由 `blockLength_ = min(blockFactor, totalNum-blockOffset)` 与 `blockLength_<=0 return` 兜底（fused_bias_leaky_relu.h:79-81, 114-116）。
- **Triton 决策**（2026-09-07 P4 回放修订：初版「一块一 program、grid=NUM_BLOCKS」实测为次优起点，被生成器优化循环修掉；改回与源码结构同构的 persistent-core 范式）：
  - 任务轴选择：打平后的一维元素轴（唯一轴；elementwise 无规约轴）。
  - 任务块总量公式：`NUM_BLOCKS = CeilDiv(numel, BLOCK)`；**grid = min(NUM_BLOCKS, vector_core_num)**——grid 超过物理核数的部分在 NPU 上串行排队，调度开销实测可观（对照实验 case6 grid=123 vs 40 核时仅 0.69x）。
  - 核分配范式（**persistent-core + 核内连续段循环**，即源码 blockFactor/ubFactor 两级切分的忠实映射：每核一段连续区间，核内按 BLOCK 流水迭代）：

    ```python
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)          # 与 host grid 一致
    total_blocks = tl.cdiv(numel, BLOCK)
    per_prog = tl.cdiv(total_blocks, num_progs)
    start_block = pid * per_prog
    end_block = tl.minimum(start_block + per_prog, total_blocks)
    for blk in range(start_block, end_block):   # 连续区间，禁止交织划分
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel                      # 尾块统一 mask，无单独尾分支
        ...
    ```

  - 核数动态读取（`torch_npu.npu.npu_config.get_device_limit`），禁止硬编码。
  - 空 tensor：wrapper `if numel == 0: return out`（对应 host 短路 tiling_arch35.cpp:155-164）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### Ascend C tilingdata → Triton 超参映射

| tilingdata 字段 | 语义 | host 侧计算式 | Triton 对应 | 候选值/推导式 |
|---|---|---|---|---|
| totalNum | 打平总元素数 | x ShapeSize（tiling L92） | `numel` 形参 | wrapper `x.numel()` |
| blockFactor | 每核处理量（32B 对齐） | CeilAlign(CeilDiv(totalNum,coreNum), 32B/typeSize)（L169） | persistent-core 的 per_prog 段长（kernel 内 cdiv 计算） | — |
| ubFactor | 单循环 UB 元素数 | fp32: FloorAlign(ubSize/4/queueBufNum, 8)；fp16: FloorAlign(ubSize/(queueBufNum·2+12), 16)（L184-191） | `BLOCK: tl.constexpr` | {2048, 4096, 8192}，对齐 128；参考量级 ubSize/24B ≈ 8K@192KB |
| negativeSlope | 属性透传 | attrs 缺省 0.2（L117-118） | kernel runtime 标量形参 | wrapper 注入 |
| scale | 属性透传 | attrs 缺省 1.414213562373（L119-120） | kernel runtime 标量形参 | wrapper 注入 |
| 尾块 tailNum_ | blockLength % ubFactor | kernel h:89-90 | `mask = offs < numel` | 主尾统一 |

### tilingkey / 模板分支 → constexpr

| tilingkey 位 / 模板分支 | 分支语义 | Triton 处理 |
|---|---|---|
| BUFFER_MODE ∈ {0,1}（tiling_key.h:26-28；host 判定 loopCountIfDouble≥2，tiling L177-179） | 队列双缓冲开关 | 不需要 constexpr——编译器注流语义，由 autotune `num_stages ∈ {2,3,4}` 覆盖 |
| DTYPE_X 宏（fp16/fp32 两实例） | dtype 模板分派 | **单 kernel**：指针 dtype 自动分派 + `.to(tl.float32)` 统一计算路径（fp32 输入 cast 为恒等，编译器消除） |

## 6. 片上内存预算（详设 §5）

| Ascend C buffer | 位置 | 容量表达式 | Triton 等效 tile | 字节数校核（BLOCK=8192） |
|---|---|---|---|---|
| inputQueueX / inputQueueBias | VECIN | ubFactor×sizeof(T)×BUFFER_NUM（h:56-57, 101-102） | 两次 `tl.load` 的 [BLOCK] tile | fp16: 2×16KB×2(dbuf)=64KB |
| outputQueueY | VECOUT | ubFactor×sizeof(T)×BUFFER_NUM（h:58, 103） | `tl.store` tile | fp16: 16KB×2=32KB |
| xFp32/biasFp32/yFp32 TBuf | VECCALC | ubFactor×4B×1，仅 fp16（h:59-61, 104-108） | 计算寄存器（编译器分配） | 3×32KB=96KB |

- UB 预算合计：host 公式按「打满 ubSize」反解 ubFactor（每元素开销 fp16 双缓冲 24B、fp16 单缓冲 18B、fp32 双缓冲 24B、fp32 单缓冲 12B）；Triton BLOCK=8192 fp16 峰值约 96KB（队列双缓冲）+ 计算 tile ≈ 192KB 上限内但偏紧，**autotune 上限建议 8192**。结论：BLOCK ≤ 8192 有富余，16384 需降 num_stages 权衡。
- double buffer 使用：三条 TQue 队列（BUFFER_NUM=2 当 BUFFER_MODE=1）；cast TBuf 恒单份（tiling L186 注释）。Triton 侧由 num_stages 覆盖，约束：「两次 GM load + 一次 GM store 的往返值得双缓冲掩盖」。

## 7. 流水编排与 Triton 可达性（详设 §6 kernel 方案）

### 流水结构

Ascend C：主循环 `for i in [0, loopCount_): CopyIn(i) → Compute → CopyOut(i)` + 单次尾块（fused_bias_leaky_relu.h:117-126）；CopyIn 双路 DataCopyPad（stride=0、不补零、越界保护，h:134-143），CopyOut 单路 DataCopyPad（h:180-187）；TQue EnQue/DeQue 自动事件同步，无手工 SetFlag/WaitFlag；无 Reg gather/scatter、无 NZ、无 CV 交互——纯向量直通。

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| DataCopyPad（GM↔UB，越界保护） | h:140-141, 186 | ✅ | `tl.load(ptr+offs, mask=offs<numel)` / `tl.store(..., mask=...)` | 无 |
| 三段式 CopyIn/Compute/CopyOut 循环 | h:117-126 | ✅ | load→compute→store 单 BLOCK 内天然三段 | 无 |
| TQue 双缓冲（buffer_num=2） | h:56-58, 39 | ⚠️ 编译器自动 | `num_stages` autotune 候选 {2,3,4}；卡片只约束「GM 往返需掩盖」 | 编译器注流质量决定 |
| TPipe EnQue/DeQue 事件同步 | h:142-143, 172-174 | ⚠️ | 丢弃，编译器自动插同步 | 无 |
| host useDoubleBuffer 判定公式 | tiling L177-179 | 丢弃（语义保留） | num_stages 候选空间覆盖「能循环≥2 次才双缓冲」策略 | 无 |
| blockFactor 32B 对齐（ubBlockSize） | tiling L166-169 | 丢弃 | BLOCK 对齐 128 覆盖 DMA 粒度语义 | 无 |

**本算子 ❌ 不可达项：0 个**——无 Reg API、无手工同步、无 NZ、无 bank 冲突规避手段。

## 8. Triton 骨架（生成用伪码）

完整可运行版见 `output/fused_bias_leaky_relu/triton_skeleton.py`；核心：

```python
@triton.autotune(configs=[triton.Config({"BLOCK": b}, num_stages=s)
                          for b, s in [(1024, 2), (2048, 2), (4096, 3), (8192, 3), (16384, 4)]],
                 key=["numel"])
@triton.jit
def fused_bias_leaky_relu_kernel(x_ptr, bias_ptr, y_ptr, numel,
                                 negative_slope, scale, BLOCK: tl.constexpr):
    # persistent-core：grid = min(cdiv(numel, BLOCK), vector_core_num)，核内连续段循环
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_blocks = tl.cdiv(numel, BLOCK)
    per_prog = tl.cdiv(total_blocks, num_progs)
    start_block = pid * per_prog
    end_block = tl.minimum(start_block + per_prog, total_blocks)
    for blk in range(start_block, end_block):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)     # Cast CAST_NONE 精确提升
        b = tl.load(bias_ptr + offs, mask=mask).to(tl.float32)
        t = x + b                                               # Add
        u = tl.where(t >= 0, t, t * negative_slope)             # LeakyRelu
        y = u * scale                                           # Muls
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)  # RTNE 回写
```

wrapper 要点：`assert x.shape == bias.shape`（无广播）；`x.contiguous()`（对应 AutoContiguous）；`numel == 0` 短路返回；grid = `min(CeilDiv(numel, BLOCK), vector_core_num)`（核数动态读 `torch_npu.npu.npu_config`，禁硬编码）；属性作 runtime 标量传入；输出 `torch.empty_like(x)`。
可选强度折减：`y = t * tl.where(t >= 0, scale, scale * negative_slope)`（2 步替代 3 步）——数学等价但浮点舍入序列不同（`(t*ns)*scale` vs `t*(ns*scale)`），默认不启用以保持与标杆逐点对齐。

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：**自建单标杆**（benchmark 库未收录本算子——ascendc-kernelgen-data `npu_benchmark/level1` 与 `ascendc-migration` 2026-09-07 全库检索无 fused_bias_leaky_relu/leaky 条目；源码侧亦无标杆脚本）。
- 标杆实现要点（torch golden，与 kernel 舍入序列逐点对齐）：

  ```python
  t = x.float() + bias.float()                                   # fp32 计算域
  ref = torch.where(t >= 0, t, t * negative_slope) * scale
  ref = ref.to(x.dtype)                                          # 末段 RTNE
  ```

- **算子级验收标准**：
  - fp32：rtol=1e-6 / atol=1e-6 全通过（kernel 与标杆同序计算，应 bitwise 接近）；
  - fp16：对「fp32 计算 + RTNE 回写」标杆 ≤ 1 ulp（唯一舍入点在末段 cast）；
  - 分段边界专项：构造 x+bias 精确等于 0 的元素，输出必须为 0（两分支同值性质）；
  - 属性变体：ns=0（退化为 ReLU×scale）、scale=1.0、ns=0.5（对齐 kernel UT 用例 L101-126）。
- **benchmark 语义超集检查**：无 benchmark case，无超集；若后续补建 case，须覆盖 {fp16, fp32} × {非默认属性} × {非 32B 对齐 numel}。
- 典型用例 shape：GEIR 示例 (2,4)（examples L151-158）；kernel UT totalNum ∈ {8, 1024, 8192, 65536}（test_fused_bias_leaky_relu.cpp:74-135）。
- 边界/特殊值用例：numel=8（非 32B 对齐、单块尾）、numel=65536（双缓冲域）、空 tensor、x+bias 跨零点混合分布。

## 10. Agent actions（生成指引）

1. 识别算子类别：elementwise 融合（2 入 1 出等形、无广播），kernel 类型 **AIV-only**。
2. 切分轴：打平一维元素轴；**persistent-core**：grid = min(CeilDiv(numel, BLOCK), 物理核数），核内连续段循环 + 尾块 mask（§4；勿用一块一 program 的裸 NUM_BLOCKS grid，NPU 上超额 block 串行排队）。
3. BLOCK 候选 {2048, 4096, 8192} + num_stages {2,3,4}，UB 校核见 §6（峰值 24B/elem）；benchmark 场景可把候选扩到 {1024..16384} 两端（小 shape 回落 1024、大 shape 试 16384）。
4. 精度提升点：**fp16 输入必须 `.to(tl.float32)` 全程计算、末段单次 `.to(fp16)` RTNE 回写**——多一次中间舍入即可能超 1 ulp 验收线。
5. 属性 negative_slope/scale：**默认 runtime 标量形参**（生产语义，对应 tilingdata 透传）；**benchmark/属性编译期稳定的场景可改 `tl.constexpr`** 换常量折叠（scale=1.0 时消去 Muls，P4 实测单项 +17.3%，4.4187→5.1818），注意属性每变一次触发一次重编译——GEIR 语境下属性本来就是编译期常量，constexpr 更贴近源码执行模型。
6. 不可达点：无。勿在 Triton 中复刻 host 双缓冲判定公式与 32B 对齐逻辑（编译器/128 对齐已覆盖）。
7. 「勿复刻」清单：本算子无 aclnn 分解、无多 kernel 策略，清单为空；唯二需主动丢弃的是 useDoubleBuffer 判定与 ubBlockSize 对齐（见 §7 表末两行）。
8. 强度折减 `t * where(t>=0, scale, scale*ns)` 默认不启用；追求指令数时启用并在报告中注明舍入序列变化。
9. 验收按 §9 标准执行（自建标杆 + 边界专项）。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（confidence: inferred），过程产物：`output/fused_bias_leaky_relu/{manifest,extraction,mapping}.md` + `triton_skeleton.py`。未经硬件实测的点：① Triton-Ascend 编译器对本 kernel 的注流质量（num_stages 实际收益）；② BLOCK=8192 fp16 的 UB 实测占用（§6 为静态估算）；③ `LeakyRelu` 向量指令与 `tl.where(t>=0, t, t*ns)` 在特殊值（±0、denormal、inf/nan）上的逐点一致性——tl.where 语义按公式推导，未对硬件指令真值表实测。回放验证：已做（2026-09-07 双盲对照，详见 `output/fused_bias_leaky_relu/validation_report.md`）——对照组（无知识）4.5231x / KB 检索组 4.3725x / 直给卡组 5.2816x，三组首轮 verify 均 10/10。结论：本算子过简，对照组已撞正确性天花板，卡片正确性增益≈0；性能主增量来自生成器优化循环。卡片 §4/§10 据此修订（grid 改 persistent-core、属性补 constexpr 变体），confidence 维持 inferred。
