# P0 材料清单 — gelu

> 生成时间：2026-09-17｜来源目录：`src/ops-nn/activation/gelu/`
> **kernel 类型判定：AIV-only 纯 elementwise**（`op_kernel/gelu_apt.cpp` 注册 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，无 Cube、无 matmul；TilingPrepare 取 `GetCoreNumAiv()`，`op_host/arch35/gelu_tiling_arch35.cpp`）。**壳算子特征**：kernel 入口是 3 路 dtype constexpr 分派壳 → `ElementwiseSch16B<schMode, GeluOp::GeluDAG<T>::OpDag>`；计算 DAG（`gelu_dag.h`）在仓内＝核心语义资产；调度器 ElementwiseSch16B 本体不在仓，**但 elewise 家族 base 库 tiling 计算器已在 `model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp`（209 行）**——§4 分核/BLOCK 事实有源码锚点。
>
> **🔻 低价值预判：低价值 Vector 简单算子（P0 动作 6 全中）——P2.5 skipped: low-value；P4 降为单组冒烟或跳过。**
> - ❌ 不可达项预判 = 0：GeluCustom 的 Reg 链（LoadAlign/Mul/Axpy/Muls/Exp/Adds/Div）是纯算术 elementwise 原语，全部落在 mapping §3 ✅ 行（Mul→`*`、Exp→`tl.exp`、Div→`/`、Muls→标量广播）；不涉及 ❌ 行的「寄存器直写 E2B/寄存器 reshape」。
> - 无广播、无规约、无 gather/scatter；shape 推导委托 `InferShape4Elewise`（输出=输入，逐元素）。
> - 公式在基座先验内：tanh 近似 GELU（sigmoid 形态 `x/(1+e^{-1.5957691(x+0.044715x³)})`），PyTorch `nn.GELU(approximate="tanh")` 同名语义。
> - 无 benchmark 语义超集：ST json 仅 dtype/shape/值域变化，golden 即 `torch.nn.GELU(approximate="tanh")`（tests/assets/golden.py）。
> - 联动结论（README P2.5 跳过纪律）：① 本行即「P2.5 skipped: low-value」显著记录；② P3 卡片 confidence 保持 **inferred**，不得升 measured(skeleton)；③ 证据上限 [DOC]/[SRC]；④ 不产 p25_report.md。P4 按既有策略降为单组冒烟。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/gelu_def.cpp` | ✅ | Input x REQUIRED {BF16,F16,F32} ND；Output y 同型同 format；AICore 配置仅 "ascend950"；`PrecisionReduceFlag(true)`；`ExtendCfgInfo("opFile.value", "gelu_apt")` | §2、§3 |
| GEIR 原型 | `op_graph/gelu_proto.h` | ✅ | `REG_OP(Gelu).INPUT(x, {DT_BF16,DT_FLOAT16,DT_FLOAT}).OUTPUT(y, 同)` —— 与 def.cpp 一致 | §3 |
| shape 推导 | `op_host/gelu_infershape.cpp` | ✅ | 委托 base 库 `InferShape4Elewise`：输出 shape = 输入 shape（elementwise 直通） | §2.1 |
| tiling host 侧（arch35/950） | `op_host/arch35/gelu_tiling_arch35.cpp` | ✅ | `ElewiseBaseTiling elewiseBaseTiling(tilingContext); DoTiling<GeluOp::GeluDAG<half>::OpDag>(*tiling)`；`tilingKey = GET_TPL_TILING_KEY(1, dType)`；`ASCEND_WORKSPACE = 0`；TilingPrepare：coreNum=`GetCoreNumAiv`、ubSize 取 `CoreMemType::UB` | §4.1.3 |
| tiling key 声明 | `op_kernel/arch35/gelu_struct.h` | ✅ | `TPL_FP16=1, TPL_BF16=2, TPL_FP32=3`；schMode∈{0,1}；tilingKey 两维（固定 1 + dtype） | §4.1.3 |
| kernel 入口（壳） | `op_kernel/gelu_apt.cpp` | ✅ | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；3 路 constexpr 分派 TPL_FP16/BF16/FP32 → `ElementwiseSch16B<schMode, GeluOp::GeluDAG<half/bfloat16_t/float>::OpDag>` | §4.1、§6 |
| **计算 DAG（核心语义）** | `op_kernel/arch35/gelu_dag.h` | ✅ | 常量 `NEG_SQRT_EIGHT_OVER_PI = -1.595769121*0.044715`、`TANH_APPROX_FACTOR = 1/0.044715`；`CAST_MODE_NONE=0 / CAST_MODE_RINT=1`；GeluCustom Reg 链见下「语义速览」；DAG = Bind(CopyIn<U>→Cast<T,U,NONE>→GeluCustom<T>→Cast<U,T,RINT>→CopyOut<U>)，MemOptCfg LEVEL_2（UB）；`if constexpr (std::is_same_v<T, float>)` | §1、§6、§7 |
| **base 库 tiling 实现** | `model/opbase-master-src-op_common/src/op_common/atvoss/elewise/elewise_tiling.cpp`（209 行） | ✅（在仓） | coreNum=按 `MIN_TILING_BITS_SIZE_PER_CORE` 折算、capped by 平台 AIV 核数；`blockFormer = ceil(ceil(dim0/coreNum)/CACHE_LINE_BYTE_LENGTH)*CACHE_LINE_BYTE_LENGTH`（cache line 对齐均分）；blockNum/blockTail；ubFormer=maxElemNum；ubLoopOfFormerBlock/ubLoopOfTailBlock/ubTail*；`innerKey = computeKey*COMPUTE_KEY_OFFSET + scheduleKey` | §4.1、§4.1.3 |
| aclnn 接口 | `op_api/aclnn_gelu.{h,cpp}` | ✅ | 两段式；`DTYPE_SUPPORT_LIST={FLOAT,FLOAT16,BF16}`；bf16 需 DAV_2201 或 Regbase 平台；空 tensor 早退；非连续输入 `Contiguous` 化、非连续输出 `ViewCopy` 回写；workspace=0 | §2 对外接口 |
| 文档 | `README.md`；`docs/aclnnGelu.md` | ✅ | 公式 `0.5*self*(1+tanh(√(2/π)*(self+0.044715*self³)))`（README）；dtype FLOAT/FLOAT16/BFLOAT16、ND、0~8 维；产品支持 950PR/950DT、A3、A2（910b）、Atlas 推理（310p）；200I/500 A2 不支持 | §1.1、§3 |
| golden | `tests/assets/golden.py` | ✅ | kernel golden = `torch.nn.GELU(approximate="tanh")`，fp16/bf16 先升 fp32 计算再回原 dtype——与 DAG 精度策略同构 | §测试设计 |
| ST 用例 | `tests/st/aclnnGelu/atk_aclnnGelu.json` | ✅ | acc=default / perf=not_key；fp16/fp32/bf16；值域 [-7,7] 为主，含 `range_values: "inf"` 边界 case（fp16）、`range_values: 8` | §测试设计 |
| UT | `tests/ut/op_host/arch35/test_gelu_tiling.cpp`（121 行） | ✅ | 仅断言 `tiling_func==GRAPH_SUCCESS`，无 tilingkey/字段数值断言 | §测试设计 |

## 计算 DAG 语义速览（gelu_dag.h，P1 详抽的对象）

**全程 fp32 中间域**（half/bf16 输入先 Cast NONE 升 fp32，计算后 Cast RINT 回写），Reg 链单趟完成 sigmoid 形态 tanh-GELU：

| 步骤 | Reg 原语 | 语义 |
|---|---|---|
| Load | LoadAlign | UB 对齐加载 x（fp32 域） |
| 1 | `Mul(x²)` | t = x·x |
| 2 | `Mul(x³)` | t = t·x = x³ |
| 3 | `Axpy(x, 1/0.044715)` | t = x³ + x·(1/0.044715) = (x + 0.044715x³)/0.044715 —— **常数融合技巧**：与下一步相乘后 1/0.044715 被 NEG_SQRT_EIGHT_OVER_PI 中的 ×0.044715 抵消 |
| 4 | `Muls(NEG_SQRT_EIGHT_OVER_PI)` | t = −1.595769121·(x + 0.044715x³)（= −√(8/π)·(x+0.044715x³)，√(8/π)≈1.5957691） |
| 5 | `Exp` | t = e^t |
| 6 | `Adds(1.0)` | t = 1 + e^(…) |
| 7 | `Div(x, t)` | y = x / (1 + e^{−√(8/π)(x+0.044715x³)}) —— 即 x·sigmoid(√(8/π)(x+0.044715x³)) ≡ tanh 近似 |
| Store | StoreAlign | 写回 |

等价性：0.5x(1+tanh(z)) = x·sigmoid(2z)，z = √(2/π)(x+0.044715x³)，2z = √(8/π)(x+0.044715x³) ≈ 1.5957691(·)。P1 需确认 Axpy 参数序与 `is_same_v<T,float>` 分支的完整语义。

## 缺省项与注意点（P1 必须处理）

1. **调度器 ElementwiseSch16B 本体不在仓**：两级切分（blockFormer/ubFormer）语义由 base 库 `elewise_tiling.cpp` 锚定，kernel 侧执行骨架维持 inferred（与 add 同款缺省，mapping §1 persistent-core 行已有既定结论可复用）。
2. **bf16 平台门槛**：aclnn 层 bf16 需 DAV_2201 或 Regbase——卡片 §3 记平台边界；Triton 侧 bf16 支持查 backend_facts/dtype_support.md。
3. **ST 含 "inf" 值域 case（fp16）**：exp(−1.5958·(x+0.044715x³)) 在 x 大正数时下溢 0、大负数时上溢 inf——sigmoid 形态的溢出行与 golden（torch tanh 形态）在 inf 输入下的行为差是验收注意点（tanh(±inf)=±1 良定义；sigmoid 形态 exp(+inf)=inf→x/inf=0? 需 P1/P4 核对「大负数」支路：x→−inf 时指数项 e^{+inf}=inf，y=x/inf→0，而 tanh 形态 y=x·0=−inf? 实际 0.5x(1+(−1))=0——两者一致为 0/极限 0，但中间 inf 是否产生 NaN 需实测关注；P4 冒烟若覆盖不了则在卡片 §9 标风险）。
4. **workspace = 0**：单 kernel 直通，无中转 buffer——Triton 侧天然满足。
5. **非连续输入由 aclnn 层 Contiguous 化**：Triton wrapper 沿用「假定连续 / 调用方 `.contiguous()`」惯例即可，卡片登记即可不需要复刻。
6. **docs 与 def 一致性**：docs/aclnnGelu.md 产品支持含 310p/Atlas 训练系列，def.cpp AICore 仅注册 ascend950——以 def.cpp 为准（旧平台走其他实现/仓），P1 标偏差。

## P0 质量门结论

五类材料齐全（proto/infershape/tiling/kernel/DAG/docs/tests/golden + op_common base 库 elewise tiling 在仓），计算 DAG 全文可读 → **通过，进入 P1**。体量小（in-repo 核心 <400 行），低价值判定成立：**P2.5 skipped: low-value**，P4 降单组冒烟，P3 卡片 confidence=inferred。
