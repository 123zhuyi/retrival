# P0 材料清单 — mhc_pre


> **2026-09-14 新流程复验**：本清单在新流程（backend_facts 三层检索）下复用，未改动。
> 材料边界不变；kernel 类型判定（CV 融合 MIX_AIC_1_2）经 apt.cpp:33 抽验仍成立。

> 生成时间：2026-09-03｜来源目录：`src/mhc/mhc_pre/`
> **kernel 类型判定：CV 融合 MIX_AIC_1_2**（`op_kernel/mhc_pre_apt.cpp:33` `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`；tilingkey 声明 `ASCENDC_TPL_MIX_AIC_1_2`，`op_kernel/arch35/mhc_pre_tiling_key.h:33-50`）。Cube（Basic API Mmad）+ Vector（Reg 级）协同，且强校验 `aivNum == aicNum * 2`（`op_host/op_tiling/arch35/mhc_pre_tiling.cpp:575-579`）。**与 mhc_sinkhorn（AIV-only）不同，本算子含 tl.dot 级矩阵乘，是 CV 融合样例。**

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/mhc_pre_def.cpp` | ✅ | Input x/phi/alpha/bias/gamma L23-47；Output hin/h_post/h_res L49-63 + 可选 inv_rms/h_mix/h_pre L65-79；Attr out_flag/norm_eps/hc_eps/op_impl_mode L94-97；三平台注册 ascend910b/910_93/950 L81-92 | §2、§3 |
| aclnn 接口 | `op_host/op_api/aclnn_mhc_pre.h`（66 行）、`aclnn_mhc_pre_v2.h`（64 行，多 opImplMode）、`aclnn_mhc_pre.cpp`（1034 行） | ✅ | V1 两段式 `aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gammaOptional, normEps, hcEps, hIn, hPost, hRes, invRmsOptional, hMixOptional, hPreOptional, ...)`；V2 增加 opImplMode 选 FP32/HF32 | §2 对外接口 |
| shape 推导 | `op_host/mhc_pre_infershape.cpp` | ✅ | rank 3/4 判定 L107-114；alpha [2]/[3] → hasResi L121-127；phi 首维三态 2n / n!+2n / n²+2n L146-163；输出 shape 推导 L166-201 | §2.1 原型约束 |
| tiling host 侧（arch35/950） | `op_host/op_tiling/arch35/mhc_pre_tiling.cpp`（939 行） | ✅ | 常量区 L57-114；校验链 L532-560；`CanUseMk` L641-649；`SelectMkStage` L651-658；BS/ND/MK 三套 tiling L660-789；模式路由 `TilingProcess` L811-833；UB 预算校验 L835-871；tilingkey L914-917 | §4.1、§4.1.3 |
| tiling host 侧（910b/910_93） | `op_host/op_tiling/mhc_pre_membase_tiling.cpp`（574 行）+ `.h` | ✅ | 老平台两阶段 tiling，tilingkey ∈ {0,10,11}（power+res / fac+res / fac 无 res）L533-541 | §3 架构差异 |
| tilingdata 结构 | `op_host/op_tiling/arch35/mhc_pre_tiling.h:31-60` | ✅ | 24 字段（coreNum/outFlag/hasGamma/hasResi/chunkTSize/v1ChunkDSize/totalLength/nD/fusionSize/N/D/normEps/hcEps/scaleMean/mL1Size/kUbSize/kL1Size/fusionAlign/cubeBlockDimM/cubeBlockDimK/multCoreSplitKSize/mkWorkspace*3/mkUseGmStage/stage2*/implMode） | §4.1.3 |
| kernel 入口（950） | `op_kernel/mhc_pre_apt.cpp` | ✅ | TILING_MODE × RESI_MODE 六分支 L35-61；MK 模式同 launch 内两阶段（pipe.Destroy 后重建 TPipe）L44-54 | §4.1.3、§6 |
| kernel 入口（910b/910_93） | `op_kernel/mhc_pre.cpp` | ✅ | 高阶 matmul API 两阶段 MhcPreStage1/Stage2，TILING_KEY_IS(0/10/11) L44-89 | §3 架构差异 |
| kernel 公共 | `op_kernel/arch35/mhc_pre_common.h` | ✅ | InitParams/MNConfig L27-69；跨核 flag 常量（X_READY=8/X_CONSUMED=9/MM_READY=10，SUBBLOCK_OFFSET=16）L93-99；L1 布局 512KB、双缓冲 128×256 L80-83 | §5、§6 |
| kernel BS 模式 | `op_kernel/arch35/mhc_pre_split_bs.h`（243 行） | ✅ | 主循环 Process L54-115；K=1024 顺序部分和 + SetAtomicAdd L171-211；跨核握手 L67-99 | §4.1.2、§6 |
| kernel ND 模式 | `op_kernel/arch35/mhc_pre_split_nd.h`（514 行） | ✅ | 解码路径；V0 全量 cast x→fp32 workspace + Σx² L360-443；K 轴按核切分 matmul L234-275；AIV1 UB 内归并 partial L473-510 | §4.1.2、§6 |
| kernel MK 模式 | `op_kernel/arch35/mhc_pre_m_k_split.h`（992 行） | ✅ | Part1 AIC/AIV L604-783；Part2 归并+后处理 L798-981；X gamma+cast+Σx² VF L94-282；4 路展开顺序归并 L283-545 | §4.1.2、§6 |
| kernel cube 服务 | `op_kernel/arch35/mhc_pre_cube_compute.h`（329 行） | ✅ | Nd2Nz 搬入 L59-134；L0 乒乓 Mmad L136-266（FP32 mmKStep=32 保序 L246-247）；Fixpipe→UB L268-291 / CO1→GM L293-309；HF32 模式 L105-111 | §6、§7 |
| kernel vector 服务 | `op_kernel/arch35/mhc_pre_vector_compute.h`（917 行） | ✅ | alpha 广播展开 L131-150；Σx²+cast VF L369-414；invRms=1/sqrt(mean+eps) L416-437；分段后处理 VFPostSegments L58-129；sigmoid hPre/hPost L582-656；hIn 累加+CAST_RINT L658-791 | §1、§6、§7 |
| 文档 | `README.md`；`docs/aclnnMhcPre.md`（696 行）；`docs/aclnnMhcPreV2.md`（698 行） | ✅ | 功能+公式 README L14-46；950 约束 n∈{4,6,8}、D∈[16,16384] 16 对齐 L171-173；A3/A2 约束 n=4、D≤100000 128 对齐、impl_mode=0 L174-177 | §1.1、§2.1、§3 |
| 测试 | `tests/ut/op_host/op_api/test_aclnn_mhc_pre.cpp`（393 行）、`test_mhc_pre_infershape.cpp`、`test_mhc_pre_tiling.cpp`（1928 行） | ✅ | 正常/异常用例（rank、n、phi 首维、alpha 长度、gamma shape 等）L267-321 | §测试设计 |
| 调用示例 | `examples/test_aclnn_mhc_pre.cpp`、`test_aclnn_mhc_pre_v2.cpp` | ✅ | x=[1·2048, 4, 2560]、phi=[24, 10240]、alpha=[3]（L277-281），fp16 | §测试设计 |

## 缺省项与注意点

- **无 config 平台 json/simplified_key.ini**（op_host 下无 config 目录），binary 配置不走本仓文件。
- **factorial（n!+2n）模式仅在 infershape 与 910b/910_93 路径存在**（infershape.cpp:146-163、mhc_pre.cpp:59-74）；arch35 tiling 把 hasResi 时的 phi 首维硬校验为 n²+2n（tiling.cpp:500-511），950 上 n! 形态会被拒绝——P1 标注为偏差。
- 源码未附标杆脚本，P4 需自建 torch 参考。
- 性能阈值常量（M_K_MAX_VALIDATED_TOTAL_LENGTH=10240 等）来自 msprof 实测注释（tiling.cpp:94-98、653-654），转化时作为超参候选依据保留。

## P0 质量门结论

五类材料齐全（proto/aclnn/infershape/tiling/kernel/docs/tests/examples）→ 通过，进入 P1。
