# P0 材料清单 — add

> 生成时间：2026-09-03｜来源目录：`Triton-Ascend-KernelWiki/kernelwiki/artifacts/ops-math/math/add/`
> **kernel 类型判定：AIV-only 广播 elementwise**（op_kernel/arch35/add.cpp:24-52 仅注册 `__aicore__` 模板 + `BroadcastSch` 向量调度；TilingPrepare 取 `GetCoreNumAiv()`，add_tiling_arch35.cpp:182）。无 Cube、无 matmul。**壳算子特征**：kernel 本体是 4 路 dtype 分派壳，计算 DAG（add_dag.h）在算子仓内；调度器（`atvoss/broadcast/broadcast_sch.h`）本体仍不可得，**但 tiling 计算器与 infershape 实现已在 `model/opbase-master-src-op_common/src/op_common/`（2026-09-03 补入）**——add 卡 §4/§5 的 BLOCK/分核事实由此获得源码锚点。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/add_def.cpp`（77 行） | ✅ | Input x1/x2 L22-49；Output y L50-63；**无 alpha 属性**；dtype 集 {BF16,FP16,FLOAT,INT32,UINT8,INT8,INT64,BOOL,COMPLEX32,COMPLEX64} L24-27；ND + UnknownShapeFormat L32-35；仅注册 ascend950 L72；DynamicRank/Shape=true、PrecisionReduceFlag=true L64-70 | §2、§3 |
| shape 推导 | `op_host/add_infershape.cpp`（26 行） | ✅ | 委托 `Ops::Base::InferShape4Broadcast`（L20，base 库）→ 输出 = 标准 broadcast 形状 | §2.1 |
| tiling host 侧（arch35/950） | `op_host/arch35/add_tiling_arch35.cpp`（190 行） | ✅ | IsMixedDtype L37-43；CheckDtype（同型 or fp16/bf16+fp32 组合，mix 输出必须 fp32）L45-69；**10 分支 dtype 分派** DoOpTiling L71-131；tilingkey=`GET_TPL_TILING_KEY(schMode)`（schMode 来自 base 库）L90 等；workspace 无 L143-146；TilingPrepare：coreNum=AIV 核数、ubSize L173-185 | §4.1.3 |
| tilingdata 结构 | `op_host/arch35/add_tiling_arch35.h:31-34` | ✅ | AddCompileInfo 仅 2 字段（coreNum/ubSize） | §4.1.3 |
| tiling key 声明 | `op_kernel/arch35/add_struct.h:21-23` | ✅ | `BRC_TEMP_SCH_MODE_KEY_DECL(schMode)`——key 只有 schMode 一个维度（具体模式枚举在 base 库） | §4.1.3 |
| kernel 入口 | `op_kernel/arch35/add.cpp`（52 行） | ✅ | 4 路 constexpr 分派：bool→AddBoolCompute / mix-dtype→AddMixDtypeCompute / fp 三型→AddWithCastCompute / 整型→AddWithoutCastCompute L24-51；全部经 `BroadcastSch<schMode, OpDag>` 执行 | §4.1、§6 |
| **计算 DAG（核心语义）** | `op_kernel/arch35/add_dag.h`（93 行） | ✅ | 四类模板见下「语义速览」；CAST_NONE_MODE=0 / CAST_RINT_MODE=1 L27-28；MemCfg=MemLevel::LEVEL_2（UB）L39/56/72/87 | §1、§6、§7 |
| binary 配置 | `op_host/config/ascend950/add_binary.json` | ✅ | **13 个 bin 实例**：BF16、BF16_F32、BOOL、CPX32、CPX64、FP16、FP16_F32、F32_BF16、F32_FP16、F32、INT32、INT64、INT8、UINT8；shape=[-2]（rank 无关动态）、FormatAgnostic | §3 |
| **base 库 tiling 实现** | `model/opbase-master-src-op_common/src/op_common/atvoss/broadcast/broadcast_tiling.cpp`（660 行）+ `broadcast_tiling_base.cpp`（47 行）+ `broadcast_tiling_noncontiguous.cpp`（667 行） | ✅（2026-09-03 补入） | DimensionCollapse 合轴 L151-269；NonContiguous 合轴计划 KEEP/MERGE/ABSORB L271-333, 344-492；GetBlockSplitFactor（ubFormer/ubOuter/ubTail/fusedProduct）L494-536；DoBrodcastTiling：maxElemNum→blockFormer=ceil(fusedProduct/coreNum)→preferMultiCore 回调→blockTail L538-607；innerKey=computeKey×OFFSET+scheduleKey L604-605；BROADCAST_MAX_DIMS=8 L632-634 | §4.1、§4.1.3 |
| **base 库 infershape 实现** | `.../op_common/op_host/infershape_broadcast_util.cpp`（165 行） | ✅（补入） | BroadcastDim 真值表（1/相等/-1 动态）L17-40；InferShape4Broadcast L127/144 | §2.1 |
| **base 库其他家族** | `.../op_common/atvoss/elewise/elewise_tiling.cpp`（209 行）、`atvoss/reduce/reduce_tiling.cpp`（1293 行）+ `batch_invariant.h` | ✅（补入） | reduce/elewise 家族 tiling，供 backlog 复用 | backlog |
| aclnn 接口 | `op_api/aclnn_add.{h,cpp}`、`aclnn_add_v3.{h,cpp}`、`add.{h,cpp}` | ✅ | V3 与 V1 差异待 P1 确认（文档区有 aclnndev 说明）；均无 alpha | §2 对外接口 |
| AICPU 兜底 | `op_kernel_aicpu/add_aicpu.{h,cpp}` + def | ✅ | 不参与 NPU 主路径，P1 仅登记 | §3 |
| GEIR 图模式 | `op_graph/add_proto.h`、`add_graph_infer.cpp` | ✅ | 登记用 | §3 |
| 文档 | `README.md`；`docs/aclnnAdd&aclnnInplaceAdd.md`、`aclnnAddV3&aclnnInplaceAddV3.md`、`aclnnAdds.md`、`aclnnInplaceAdds.md` | ✅ | README 公式 y=x₁+x₂ L24-28；产品支持表（950/A3/A2 等）L6-15 | §1.1、§3 |
| ST 用例 | `tests/st/arch35/ttk_kernel_add_st.csv` | ✅ | 广播形态丰富：异 rank 组合（(75,17716)+(2,1,17716)）、(1,) 标量广播、6/7/8 维高 rank、(17453,)+(1,1,1,2,1)、625 万长 1D 等；含 mix-dtype 与 bool/complex | §测试设计 |
| UT | `tests/ut/op_host/test_add_infershape.cpp`、`op_host/arch35/test_add_tiling.cpp`、`op_api/test_aclnn_add*.cpp`、`op_kernel_aicpu/test_add.cpp` | ✅ | host/api 层异常分支 | §测试设计 |
| 调用示例 | `examples/test_aclnn_add.cpp`、`test_aclnn_inplace_add.cpp`、`test_geir_add.cpp` | ✅ | — | §测试设计 |

## 计算 DAG 语义速览（add_dag.h，P1 详抽的对象）

| 模板（dtype 范围） | 计算链 | 精度策略 |
|---|---|---|
| `AddWithCastCompute<T>`（T=half/bf16/**float**） | CopyInBrc→Cast→**fp32 Add**→Cast<T, RINT>→CopyOut | 输入升 fp32，相加后 RTNE 回 T（add_dag.h:43-58） |
| `AddMixDtypeCompute<T1,T2>`（fp16/bf16 × fp32） | 双输入各自 Cast→fp32 Add→**fp32 直出**（不回 cast） | 输出恒 fp32（与 CheckDtype 一致） |
| `AddWithoutCastCompute<T>`（int8/uint8/int32/int64/complex） | CopyInBrc→Add<T>→CopyOut | 原生精度直加 |
| `AddBoolCompute<int8>` | Add→**Mins(const 1)**→CopyOut | min(x1+x2, 1)，即布尔或 |

## 缺省项与注意点（P1 必须处理）

1. **alpha 属性不存在**：Ascend C Add 公式 y=x₁+x₂，benchmark `3_Add.py` 的 `torch.add(x, y, alpha=alpha)` 中 alpha∈{0.5,-1,2,1.5,3,0.25,-0.5} 大量 ≠1——**benchmark 语义超集**，生成 kernel 须融合 `out = x + alpha·y`，卡片 §9 记录。
2. **~~调度/tiling 三件套在 base 库~~（2026-09-03 解决大半）**：opbase-master-src-op_common 提供了 tiling 计算器（broadcast_tiling*.cpp）与 infershape（infershape_broadcast_util.cpp）实现；仍缺 kernel 侧 BroadcastSch 调度器本体与 broadcast_tiling.h 头文件（struct 字段、CACHE_LINE 常量值从 .cpp 反推）→ §4 升级为源码锚点，§6 调度器维持 inferred。
3. **README 声明 dtype 集与 def.cpp 不一致**：README 参数表含 INT16/FLOAT64/COMPLEX128/STRING，def.cpp 与 add_binary.json 实际仅 10 种——以 def.cpp + binary json 为准，P1 标偏差。
4. benchmark 输入值域大（偶数 case x~N(μ∈[-100,100], σ∈[1,25])，y~randn），fp16/bf16 大数相加 + alpha 放缩的精度行为是验收重点。
5. ST csv 含 benchmark 未覆盖的形态（高维异 rank、(1,) 标量、625 万元素），卡片 §9 以 benchmark case 为验收域、ST 为扩展参考。
6. 源码未附标杆脚本；benchmark 侧 torch golden 即标杆（NPUKernelBench）。

## P0 质量门结论

五类材料齐全（proto/infershape/tiling/kernel/dag/docs/tests/examples/binary 配置 + op_common base 库 tiling/infershape 实现），计算 DAG 在仓内可读 → 通过，进入 P1。体量远小于 mhc（in-repo 核心 ~490 行 + base 库 1400 行 vs mhc_pre 数千行）；base 库依赖仅剩 kernel 侧调度器（§6 inferred），§4/§5 已有源码锚点。
