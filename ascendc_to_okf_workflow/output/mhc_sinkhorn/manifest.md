# P0 材料清单 — mhc_sinkhorn

> 生成时间：2026-09-03｜来源目录：`src/mhc/mhc_sinkhorn/` + `src/mhc/mhc_sinkhorn_common/`
> **kernel 类型判定：AIV-only**（`op_kernel/mhc_sinkhorn_apt.cpp:26` `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`；tilingkey 声明 `ASCENDC_TPL_AIV_ONLY`，`arch35/mhc_sinkhorn_tiling_key.h:22-25`）。无 Cube/Mmad，纯 Vector + Reg 级实现。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/mhc_sinkhorn_def.cpp` | ✅ | `Input("h_res")` L23-28；`Output("y"/"norm_out"/"sum_out")` L29-46；`Attr eps/num_iters/out_flag` L47-49；`AICore().AddConfig("ascend950")` L51-59 | §2 原型设计、§3 规格约束 |
| aclnn 接口 | `op_host/op_api/aclnn_mhc_sinkhorn.h` L30-38（两段式）；`aclnn_mhc_sinkhorn.cpp`（263 行） | ✅ | `aclnnMhcSinkhornGetWorkspaceSize(x, eps, numIters, output, normOut, sumOut, ...)` | §2 对外接口 |
| L0 接口 | `op_host/op_api/mhc_sinkhorn.cpp` L32-49 | ✅ | `MhcSinkhornAiCore(x, outFlag, eps, numIters, ...)`，`ADD_TO_LAUNCHER_LIST_AICORE` | §2 |
| shape 推导 | `op_host/mhc_sinkhorn_infershape.cpp` | ✅ | 3/4 维判定 L73-92；n0==n1 校验 L94-99；norm/sum size 公式 L106-123；unknownRank 处理 L66-71 | §2.1 原型约束、Infershape |
| tiling host 侧 | `op_host/mhc_sinkhorn_tiling.cpp` | ✅ | 属性校验 L73-105（num_iters∈[1,100]、out_flag∈{0,1}）；shape 校验 L152-263（n∈{4,6,8}）；`SplitByCoreNum` L265-276；tilingkey L308-315；`SetBlockDim(usedCoreNum)` L325-334 | §4.1 分核、§4.1.3 tilingkey |
| 分核公共实现 | `../mhc_sinkhorn_common/op_host/arch35/mhc_sinkhorn_base_tiling.cpp` | ✅ | `BaseSplitCores` L33-46（forward/backward 取 UB 占用更小者）；`ForwardSplitCore` L122-145；`ForwardCalOccupySize` L100-120 | §4.1.1 分核策略、§5 UB 预算 |
| 平台配置 | `op_host/config/ascend950/mhc_sinkhorn_binary.json`、`mhc_sinkhorn_simplified_key.ini` | ✅ | binary json：input h_res / outputs y,norm_out,sum_out 全 fp32 ND shape [-2]；ini `default=0` | §3 规格约束 |
| kernel 入口 | `op_kernel/mhc_sinkhorn_apt.cpp` | ✅ | TEMPLATE_MODE 0/1 分支 L30-40；`REGISTER_TILING_DEFAULT(MhcSinkhornTilingData)` | §4.1.3 |
| kernel 实现 | `op_kernel/arch35/mhc_sinkhorn_arch35.h`（556 行） | ✅ | 类 `MhcSinkhornSimd<T, OUT_FLAG>` L41-42；Init L86-122；CopyIn/CopyOut L124-147；CalcSoftmax L149-245；CalcColNorm L247-306；CalcRowNorm L308-384；ScatterOutFromNorm L386-440；Process L442-553 | §5 片上内存、§6 流水、计算流程 |
| tiling 数据结构 | `op_kernel/arch35/mhc_sinkhorn_struct.h` L18-31 | ✅ | 11 字段（eps/num_iters/out_flag/n/usedCoreNum/tNormCoreLoop/tUbFactor/tUbFactorTail/tTailCoreLoop/tUbTailTail/tNormCore） | §4.1.2/§4.1.3 |
| tilingkey 声明 | `op_kernel/arch35/mhc_sinkhorn_tiling_key.h` L20-25 | ✅ | `TEMPLATE_MODE ∈ {0,1}`，AIV_ONLY | §4.1.3 |
| 文档 | `README.md`（功能+公式 L14-28，约束 L79-83）；`docs/aclnnMhcSinkhorn.md`（447 行，约束 L139-162） | ✅ | Sinkhorn-Knopp 迭代公式；n∈{4,6,8}；3/4 维；norm/sum 元素数公式 | §1.1、§2.1、§3 |
| 测试 | `test/ut/op_host/op_api/test_aclnn_mhc_sinkhorn.cpp` | ✅ | 正常 shape {1,128,4,4}/{1024,6,6} L38-62；异常 {1,0,6,6} L78、fp16 dtype L100、5 维 L121 | §测试设计 |
| 调用示例 | `examples/test_aclnn_mhc_sinkhorn.cpp` | ✅ | x_shape={1024,4,4} L311；norm_out={40*4*8*1024} L320；sum_out={40*8*1024} L324（num_iters=20） | §测试设计 |
| GEIR proto | `op_graph/mhc_sinkhorn_proto.h` | ✅ | 图模式原型（51 行，未详读） | §2 |

## 缺省项

- 无 arch22 目录：**仅 arch35（ascend950）一套实现**，无架构差异小节。
- 无独立性能测试报告：性能标杆需回放验证阶段自建。
- README 接口描述（`init_matrix/max_iter/epsilon/h_res`）与 def.cpp 实际 IR（`h_res` 输入 + `y/norm_out/sum_out` 输出 + 属性）**不一致**，以 def.cpp/aclnn 头文件为准（P1 中标注偏差）。

## P0 质量门结论

五类材料齐全 → 通过，进入 P1。
