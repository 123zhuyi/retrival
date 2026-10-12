# P1 语义抽取 — gelu

> 生成时间：2026-09-17｜依据：`output/gelu/manifest.md`（低价值判定成立，P2.5 skipped: low-value）
> 锚点格式：`file:line`。所有 [SRC] 均指向仓内文件；base 库 elewise tiling 在仓（`model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp`，下文简称 et）。

## 1. 功能与公式

高斯误差线性单元（tanh 近似）：
- 文档公式（README.md / docs/aclnnGelu.md）：`out = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`
- **kernel 实际实现形态（sigmoid 形）**：`gelu_dag.h:29-30` 注释自述 `x / (1 + e^(-1.5957691(x + 0.044715x³)))`。
  等价推导：0.5x(1+tanh(z)) = x·sigmoid(2z)，2z = 2·√(2/π)(·) = √(8/π)(·) ≈ 1.5957691(·)。两形数学恒等 [SRC gelu_dag.h:25-30 + 推导]。
- 逐元素、无广播、无规约；输出 shape = 输入 shape（`InferShape4Elewise` 委托，`op_host/gelu_infershape.cpp`）。
- golden 即 `torch.nn.GELU(approximate="tanh")`（`tests/assets/golden.py:30-45`，fp16/bf16 先升 fp32 再回原 dtype）。

## 2. 接口与原型

### 2.1 GEIR op（gelu_def.cpp / gelu_proto.h）
- `Input("x")` REQUIRED，dtype {BF16, F16, F32}，Format ND（`gelu_def.cpp:23-27`）；`Output("y")` 同型同 format（`:28-32`）。
- `DynamicRankSupportFlag(true)`、`DynamicShapeSupportFlag(true)`（`:37-38`）；`PrecisionReduceFlag(true)`（`:40`）；`ExtendCfgInfo("opFile.value", "gelu_apt")`（`:41`）。
- AICore 仅注册 `"ascend950"`（`:42`）。`op_graph/gelu_proto.h`：`REG_OP(Gelu)` 同 dtype 集。
- **偏差**：docs/aclnnGelu.md 产品支持表含 310p/Atlas 训练系列/A3/A2——旧平台走他仓实现，本仓 def 以 ascend950 为准（manifest 注意点 6）。

### 2.2 aclnn 对外接口（op_api/aclnn_gelu.cpp）
- 两段式：`aclnnGeluGetWorkspaceSize(self, out, workspaceSize, executor)`（:89）→ `aclnnGelu(workspace, workspaceSize, executor, stream)`。
- dtype 白名单 `DTYPE_SUPPORT_LIST = {FLOAT, FLOAT16, BF16}`（:23-24）；**bf16 平台门槛**：`NpuArch::DAV_2201 || IsRegbase()`（:28）。
- 空 tensor 早退；非连续输入经 `l0op::Contiguous` 化（:112），调 `l0op::Gelu`（:116），非连续输出 `ViewCopy` 回写（:120）。
- workspace = 0（`ASCEND_WORKSPACE = 0`，`gelu_tiling_arch35.cpp:22,118`）。

### 2.3 原型约束
- 无属性（无 approximate 开关——**本算子只实现 tanh 近似形**，none/erf 形不在本仓）。
- 0~8 维（README），rank 无关动态（DynamicRank）。

## 3. 规格约束

| 项 | 值 | 锚点 |
|---|---|---|
| dtype | fp16 / bf16 / fp32（in/out 同型） | gelu_def.cpp:25,30 |
| format | ND | gelu_def.cpp:26 |
| shape | 0~8 维，全动态 | README；def.cpp:37-38 |
| 平台（本仓） | ascend950 only | def.cpp:42 |
| aclnn bf16 门槛 | DAV_2201 或 Regbase | aclnn_gelu.cpp:28 |
| ST 值域 | [-7,7] 为主；含 inf 边界（fp16） | atk_aclnnGelu.json |

## 4. 分核策略与 tiling（✅ base 库源码锚点）

### 4.1 分核（源码事实，et = elewise_tiling.cpp）
- 纯 AIV：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`（`gelu_apt.cpp:31`）；TilingPrepare `coreNum = GetCoreNumAiv()`、`ubSize` 取 UB（`gelu_tiling_arch35.cpp:144-145`）。
- 展平一维均分：实际核数 = `ceil(dim0·minDtypeBits / MIN_TILING_BITS_SIZE_PER_CORE)`，capped by 平台 AIV 核数（et）。
- 每核连续段：`blockFormer = ceil(ceil(dim0/coreNum)/CACHE_LINE_BYTE_LENGTH)·CACHE_LINE_BYTE_LENGTH`（cache line 对齐），`blockNum/blockTail` 分主核/尾核（et）。
- 核内 UB 循环：`ubFormer = maxElemNum`，`ubLoopOfFormerBlock/ubLoopOfTailBlock/ubTail*`（et）——**两级切分（核间连续段 + 核内 UB 块循环）**，与 add 家族同一范式。
- schMode∈{0,1} 由 base 库 innerKey 决定（`innerKey = computeKey*COMPUTE_KEY_OFFSET + scheduleKey`，et）。

### 4.2 tilingkey 语义
- `tilingKey = GET_TPL_TILING_KEY(1, dType)`（`gelu_tiling_arch35.cpp:120`）：第一维固定 1（单 computeKey），第二维 dtype∈{FP16=1, BF16=2, FP32=3}（`gelu_struct.h:21-23`）。
- TPL 声明：schMode∈{0,1} × dType∈{FP16,BF16,FP32} 共 6 实例（`gelu_struct.h:28-36`）。
- UT 仅断言 tiling 成功，无 key/字段数值断言（`test_gelu_tiling.cpp:121`）。

### 4.3 BLOCK 候选（Triton 转化用）
- 两级切分的忠实映射 = persistent-core + 核内连续段循环（mapping §1 行 2 既有结论）：`grid = min(cdiv(numel, BLOCK), vector_core_num)`。
- ubFormer 语义 = 单核单次 UB 块大小；Triton 侧 BLOCK 即此概念，初选取值与对齐预算在 P2 校核（fp32 时 VL=VECTOR_REG_WIDTH/4，`gelu_dag.h:37-38`——Reg 链按向量寄存器宽度循环，说明 BLOCK 内再被硬件 VL 细分，BLOCK 取 cache-line 倍数即可）。

### 4.4 tilingdata 字段
- base 库 ElewiseBaseTiling 输出（blockFormer/blockTail/ubFormer/ubLoop*/ubTail* 等，et）；kernel 壳经 `ElementwiseSch16B<schMode, DAG> sch(tilingData)` 消费（`gelu_apt.cpp:34,38,42`）。调度器本体不在仓（维持 inferred，同 add 缺省）。

## 5. 片上内存

- DAG `MemCfg = MemOptCfg<MemLevel::LEVEL_2>`（UB 级，`gelu_dag.h:84`）。
- 数据面：fp16/bf16 输入 1 份 U + fp32 中间 1 份 T + 输出 1 份 U；fp32 时 2 份。UB 预算校核留给 P2（backend_facts/hardware_platforms.md：A2 192KB / 950 248KB）。
- 无 workspace、无 GM 中转（§2.2）。

## 6. 流水编排（计算 DAG —— 核心语义资产，gelu_dag.h）

### 6.1 DAG 结构（:74-86）
```
GeluDAG<U, T=float>::OpDag =
  CopyIn<U> → Cast<T,U,NONE> → GeluCustom<T> → Cast<U,T,RINT> → CopyOut<U>
```
- 输入 dtype U ∈ {half, bf16, float}；**计算 dtype T 恒为 float**（模板默认参数 :74）。
- Cast 进 = `CAST_MODE_NONE`（:27，half/bf16→fp32 无舍入）；Cast 出 = `CAST_MODE_RINT`（:28，fp32→U，RINT=RTNE）。
- kernel 壳 3 路 constexpr 分派仅改 U（`gelu_apt.cpp:33-43`）——三条实例共享同一份 fp32 GeluCustom。

### 6.2 GeluCustom Reg 链（:32-71，仅 `__CCE_AICORE__` 设备侧）
逐 VL（向量寄存器宽度）循环（:37-39, 52），每趟 7 条原语（:55-65）：

| # | 原语 | 语义 | 累算结果 |
|---|---|---|---|
| 1 | `LoadAlign(vregInput, src+i·vl)` | UB 对齐加载 | x |
| 2 | `Mul(sqr, x, x)` | x² | x² |
| 3 | `Mul(cub, sqr, x)` | x³ | x³ |
| 4 | `Axpy(cub, x, 1/0.044715)` | dst += src·a → x³ + x/0.044715 | (x + 0.044715x³)/0.044715 |
| 5 | `Muls(cub, cub, −1.595769121·0.044715)` | 标量乘 | −√(8/π)(x + 0.044715x³)（步骤 4 的 1/0.044715 被 ×0.044715 抵消——**一次 Axpy 融合掉两项加法**） |
| 6 | `Exp(cub)` | e^t | e^{−√(8/π)(x+0.044715x³)} |
| 7 | `Adds(cub, 1.0)` + `Div(out, x, cub)` | 1+e^t；x/(1+e^t) | y |
| 8 | `StoreAlign(dst+i·vl, out, mask)` | 写回 | — |

- mask 由 `Reg::UpdateMask<T>(count)` 尾块更新（:53）——VL 级尾处理，无语义影响。
- **`if constexpr (std::is_same_v<T, float>)`（:49）**：计算体只对 fp32 实例化；half/bf16 实例为空操作。由于 DAG 保证 T=float（:74），该 guard 是编译期防呆——语义结论：**gelu 在本仓只有 fp32 计算域一种实现，低精度全靠进出 Cast**。

### 6.3 aclnn 层策略
- 无多 kernel 分解：直通 `l0op::Gelu`（aclnn_gelu.cpp:116），Contiguous/ViewCopy 仅处理非连续（:112,120）——Triton 侧一律单 kernel，调用方 `.contiguous()` 即可。

## 7. 数值精度

- **中间域恒 fp32**（§6.1）；fp16/bf16 的精度行为 = 升 fp32 计算 + RTNE 回写（Cast RINT，:28,80）。golden 同构（golden.py:36-44 先升 fp32 再 astype 回）。
- 常数：`NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715`（:25）、`TANH_APPROX_FACTOR = 1 / 0.044715`（:26）——注意是编译期 double 表达式折成 float；Triton 侧直接用 `1.5957691` 字面量即可（差异 <1e-7，fp32 域内不可辨），或照抄两常数表达式。
- 精度隐患（sigmoid 形特有，manifest 注意点 3）：大正 x → e^t 下溢 0 → y=x（正确）；大负 x → e^t 上溢 inf → x/inf → −0（tanh 形 0.5x(1+(−1))=0 一致）；**x = −inf 时 cub = e^{+inf}=inf，y = −inf/inf = NaN？**实际 −inf/inf 按 IEEE = NaN，而 tanh 形 0.5·(−inf)·0 = NaN 亦 NaN——行为一致，但 fp16 值域含 inf 的 ST case（atk json id 4）是验收观察点。有限大负数支路无风险（e^700+ 才溢出 fp32，|x|>88 时 x³ 项早已使指数 <−10⁴ → e^t→inf？x=−10 时 t=−1.5958·(−10−44.7)≈+87.3，e^87.3≈7e37 已近 fp32 上限 3.4e38；x=−11 即溢出 inf → y=−11/inf=−0，而真值 gelu(−11)≈−11·Φ(−11)≈−10⁻²⁸≈−0——**方向正确，精度无损**）。结论：sigmoid 形在全值域与 tanh 形数值一致，溢出支路恰好给出正确极限值。
- PrecisionReduceFlag(true)（def.cpp:40）——允许框架在图模式做精度降低优化，aclnn 直调不受影响。

## 8. 测试与标杆

- golden：`torch.nn.GELU(approximate="tanh")`，fp16/bf16 升 fp32（golden.py:30-45）。
- ST（atk_aclnnGelu.json）：acc=default、perf=not_key；dtype fp16/fp32/bf16；shape 小（[1,1]/[1] 等）；值域 [-7,7]、8、inf。
- UT：tiling 成功性（test_gelu_tiling.cpp:121）。
- 无仓内 benchmark 脚本；验收标杆 = torch fp32 golden，容差按 acc=default（具体阈值不在仓，P4 冒烟用通用 dtype 默认容差）。

## 9. 偏差与风险清单

1. docs 产品支持表 ≠ def.cpp（950-only）——以 def 为准（§2.1）。
2. 调度器 ElementwiseSch16B 本体不在仓——§4.1 分核事实由 base 库 et 锚定，执行骨架 inferred（同 add 先例，不影响 Triton 转化）。
3. sigmoid 形 inf 支路已论证数值一致（§7），但 e^t 中间溢出依赖硬件 Exp 产生 +inf 且 Div 处理 inf——ST inf case 通过即佐证，卡片记为「官方形态背书」。
4. **bf16 平台门槛 DAV_2201/Regbase**（aclnn_gelu.cpp:28）——卡片 §3 平台边界必记。
5. 无 approximate 属性——本仓仅 tanh 形；若上游需要 erf 形（approximate="none"）是另一个算子语义，卡片登记即可。

## P1 质量门

- [x] 三层语义齐备：kernel 本体（§6 Reg 链逐条）、aclnn 层（§2.2/§6.3）、测试层（§8）
- [x] 全部关键论断带 file:line 锚点；base 库 tiling 在仓 → §4 为源码事实非 inferred
- [x] 无 broadcast/reduction/gather；❌ 项 = 0（Reg 链全落在 mapping §3 ✅ 行）
- [x] 低价值预判复核通过（manifest 结论不变）：进入 P2，**P2.5 skipped: low-value**
