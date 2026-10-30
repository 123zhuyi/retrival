# FlashAttentionScoreGrad — 源文件清单 (S1)

来源仓库：`kernelwiki/artifacts/ops-transformer/`（CANN ops-transformer，固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`，2026-07-09）。
算子目录：`attention/flash_attention_score_grad/`。共 ~155 个文件。下表按角色与中心度归纳，所有路径相对算子目录。`centrality`：core=核心计算/tiling，supporting=辅助，indirect=测试/示例/构建。

## op_host（Host 侧 tiling/定义/inferShape）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_host/flash_attention_score_grad_def.cpp` | 508 | 定义 | core | `FlashAttentionScoreGrad` OpDef：输入输出 dtype/layout、属性、AICore 配置（ascend950/ascend910b/ascend910_93）L20-503 |
| `op_host/flash_attention_score_grad_tiling.cpp` | 436 | tiling入口 | core | `TilingFlashAttentionGradScore` 架构分发 L372-395、`TilingPrepareForFlashAttentionScoreGrad` 平台参数读取 L397-429、空 tensor `RunEmptyTiling`/`RunEmptyTilingRegbase` L68-234、`CheckBaseInput`/`CheckParams` L281-370 |
| `op_host/flash_attention_score_grad_infershape.cpp` | 204 | inferShape | supporting | 输出 dq/dk/dv/dpse/dq_rope/dk_rope/dsink 形状推导 |
| `op_host/arch32/flash_attention_score_grad_tiling_common.h` | 379 | tiling公共 | core | 枚举（`InputIndex` L104-130、`AttrIndex` L142-156、`SparseMode` L187-197、`PseType` L168-174、`InputLayout` L73-79、`AttenMaskCompressMode` L132-140）、`PreParams`/`PreSfmgParams`/`PostParams` TilingData 字段 L272-377 |
| `op_host/arch32/flash_attention_score_grad_tiling_common.cpp` | 351 | tiling公共 | supporting | 形状/dtype 校验函数实现 |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2.cpp` | 2553 | tiling | core | 通用模板 tiling（核间切 B/N2/G/S1，核内切 S1/S2） |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.cpp` | 2117 | tiling | core | BN2 模板 tiling（核间切 B/N2，核内切 G/S1/S2） |
| `op_host/arch32/flash_attention_score_grad_tiling_ngs1s2_bn.cpp` | 1187 | tiling | core | NGS1S2_BN 模板 tiling（核内切 N2.i） |
| `op_host/arch32/flash_attention_score_grad_tiling_bngs1s2_b.cpp` | 1143 | tiling | core | B 模板 tiling（核间切 B，核内 B.i） |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp` | 2142 | tiling | core | SameAB 模板 tiling（Cube 为主核，_sab 后缀） |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic.cpp` | 439 | tiling | core | Basic/MLA 模板 tiling |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_basic_det.cpp` | 548 | tiling | core | 确定性计算模板 tiling |
| `op_host/arch32/flash_attention_score_grad_tiling_unpadded_attension.cpp` | 62 | tiling | supporting | unpadded 路由 |
| `op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.h` / `_basic.h` / `_basic_det.h` / `_sab.h` | 156/101/114/181 | tiling头 | supporting | 各模板 TilingData 结构体声明 |
| `op_host/arch35/flash_attention_score_grad_tiling_common_regbase.h` | 271 | tiling公共 | core | arch35 枚举（与 arch32 对齐，额外含 `D_SCALE_DS_IDX`/`D_SCALE_P_IDX` L125-126、`BATCH_MAX_SIZE=2048` L54） |
| `op_host/arch35/flash_attention_score_grad_tiling_common_regbase.cpp` | 132 | tiling公共 | supporting | arch35 形状/dtype 校验 |
| `op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.cpp` | 4529 | tiling | core | arch35 统一 tiling（BN2/BN2S2/BN2GS1S2/确定性 在此分模板参数隔离）；基础块常量 L93-126（`BN2_MAX_S=128` L94、`BN2S2_MAX_S=1024` L95、`REGBASE_POST_BASE=128*128` L106、`ALIGN128` L93、HIFP8 准入阈值 L119-126） |
| `op_host/arch35/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_regbase.h` | 459 | tiling头 | supporting | arch35 TilingData 结构体 |

## op_kernel（Kernel 侧）

| 文件 | 行数 | 角色 | 中心度 | 关键符号 / 职责 |
|------|------|------|--------|----------------|
| `op_kernel/flash_attention_score_grad.cpp` | 903 | kernel入口 | core | kernel 函数 `flash_attention_score_grad<UB0,UB1,Block,IsSameAB,DataType,Layout,Sparse,MatmulCfg,Mm12IsNZOut,Mm345IsNZOut,HasDropOut,HasPse,HasAttenMask,EnableL1Reuse,TNDS1Pingpong,S1TemplateType,S2TemplateType,DTemplateType,IsDeterministic,HasRope>` L488-903；`KERNEL_TYPE_MIX_AIC_1_2` L501；模板分发 `if constexpr` 树 L508-901；INVOKE_FAG_* 宏 L59-484 |
| `op_kernel/flash_attention_score_grad_apt.cpp` | 79 | kernel入口 | core | ascend950 专用入口（def.cpp L270 `opFile=flash_attention_score_grad_apt`） |
| `op_kernel/arch32/flash_attention_score_grad_tiling.h` | 5025 | tiling数据 | core | arch32 全部 TilingData 结构体（`FlashAttentionScoreGradTilingData`、`...S1s2Bn2gs1s2`、`...S1s2Bn2`、`...Ungs1s2Bbn`、`...Ubngs1s2Bb`、`...SameAb`、`...Mla`、`...BasicDet`、空 tensor `EmptyTensorTilingData`）及 matmul tiling 字段（mm1/mm2/mm3/mm4 TilingData） |
| `op_kernel/arch32/flash_attention_score_grad_template_tiling_key.h` | 1734 | dispatch | core | arch32 tiling key 编码（`GET_TPL_TILING_KEY` 宏、bit 字段、`STemplateType`/`DTemplateType` 对齐枚举 Aligned16/32/48/64/128/192、NotAligned/NotAligned30/60/72/88） |
| `op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2.h` | 2816 | kernel模板 | core | 通用模板 kernel 类（`FlashAttentionScoreGradS1s2Bn2gs1s2`），3 阶段 `Process`/`ComputeMM1`/`ComputeVec`/`ComputeMMDqkv` |
| `op_kernel/arch32/flash_attention_score_grad_s1s2_bn2.h` | 4000 | kernel模板 | core | BN2 模板 kernel 类 |
| `op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h` | 4310 | kernel模板 | core | SameAB 模板（Cube 为主核） |
| `op_kernel/arch32/flash_attention_score_grad_ngs1s2_bn.h` | 1632 | kernel模板 | core | NGS1S2_BN 模板 |
| `op_kernel/arch32/flash_attention_score_grad_bngs1s2_b.h` | 1590 | kernel模板 | core | B 模板 |
| `op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_basic.h` / `_basic_det.h` | 276/263 | kernel模板 | core | Basic/MLA、确定性 模板 |
| `op_kernel/arch32/flash_attention_score_grad_pre.h` / `_post.h` / `_sfmg.h` | 829/2175/305 | kernel辅助 | core | Pre（dq/dk/dv 清零、dropMask bit2bool）、Post（cast FP32→输出、muls scale）、Sfmg（FlashSoftmaxGrad 预处理） |
| `op_kernel/arch32/flash_attention_score_grad_empty_tensor.h` | 86 | kernel模板 | supporting | 空 tensor 清零 kernel |
| `op_kernel/arch32/flash_attention_score_grad_constant_propagation.h` | 69 | kernel辅助 | supporting | 常量传播优化 |
| `op_kernel/arch32/basic_modules/` | — | kernel辅助 | supporting | `cube_op*.h`/`vec_op*.h`/`cube_modules/`/`vec_modules/`（Cube/Vector 基础算子封装、地址计算 `addr_compute_det.h`、`vec_sfmg.h` FlashSoftmaxGrad 向量算子） |
| `op_kernel/arch35/flash_attention_score_grad_kernel.h` | 403 | kernel模板 | core | arch35 非确定性统一 kernel 入口 |
| `op_kernel/arch35/flash_attention_score_grad_kernel_base.h` | 2353 | kernel模板 | core | arch35 kernel 基类（BN2/BN2S2/BN2GS1S2 共用，模板参数隔离） |
| `op_kernel/arch35/flash_attention_score_grad_kernel_deter.h` | 899 | kernel模板 | core | arch35 确定性 kernel |
| `op_kernel/arch35/flash_attention_score_grad_kernel_quant.h` | 886 | kernel模板 | core | arch35 FP8 量化 kernel |
| `op_kernel/arch35/flash_attention_score_grad_block_cube.h` / `_block_vec.h` | 1465/976 | kernel阶段 | core | Cube 阶段（`ComputeMM1` 等 matmul）、Vector 阶段（`ComputeVec`：dp/ds/dq/dk/dv 向量算子） |
| `op_kernel/arch35/flash_attention_score_grad_block_cube_quant.h` / `_block_vec_quant.h` | 669/385 | kernel阶段 | core | FP8 量化 Cube/Vector 阶段 |
| `op_kernel/arch35/flash_attention_score_grad_entry_regbase.h` | 475 | kernel入口 | core | arch35 kernel 入口封装 |
| `op_kernel/arch35/flash_attention_score_grad_s1s2_bn2_regbase.h` / `_s1s2_bn2gs1s2_regbase.h` / `_s1s2_bn2s2_regbase.h` / `_s1s2_bn2gs1s2_pre_regbase.h` / `_post_regbase.h` / `_presfmg_regbase.h` | 1412/2865/516/266/213/495 | kernel模板 | core | arch35 各分模板实现 |
| `op_kernel/arch35/flash_attention_score_grad_template_tiling_key.h` | 1234 | dispatch | core | arch35 tiling key 编码（`GET_TPL_TILING_KEY` 宏，12 字段） |
| `op_kernel/arch35/flash_attention_score_grad_tiling_data_regbase.h` | 592 | tiling数据 | core | arch35 TilingData 结构体（`coreNum` L88 等） |
| `op_kernel/arch35/flash_attention_score_grad_common.h` / `common.h` / `deter.h` | 163/405/2321 | kernel辅助 | core | 公共定义、确定性计算工具 |
| `op_kernel/arch35/matmul_modules/` | — | kernel辅助 | core | `fag_custom_matmul_policy.h`（MatmulPolicyType NORMAL/UNSPLITK）、`matmul_config.h`、`copy_cube_in/*`（preload/align/rope 三种 cube 输入拷贝）、`cube_in_buffer/`、`cube_out_buffer/`（mm1/mm3 输出缓冲）、`fag_flag_data.h` |
| `op_kernel/arch35/vector_api/` | — | kernel辅助 | core | `vf_softmax_grad_front_cast*.h`（softmax 反向，按对齐 256/512/768 + F16/F32 特化）、`vf_muls_sel_simple_softmax*.h`、`cast_softmax_grad.h`、`dropout.h`、`pse_atten_mask_muls_simple_softmax.h`、`vf_anti_quant_*`（FP8 反量化）、`vf_ds_abs_reduce_max.h`、`vf_broadcast_sub_mul.h` |

## op_api / docs / examples / tests

| 文件 | 行数 | 角色 | 中心度 | 职责 |
|------|------|------|--------|------|
| `op_api/aclnn_flash_attention_score_grad.cpp` | 2665 | api | core | aclnn 两段式接口：`aclnnFlashAttentionScoreGrad`(L1541)/`V2`(L1803)/`V3`(L2346)、`aclnnFlashAttentionUnpaddingScoreGrad`(L1637)/`V2`(L1898)/`V3`(L2080)/`V4`(L2151)/`V5`(L2352-) 及各 `GetWorkspaceSize` |
| `op_api/aclnn_flash_attention_score_grad.h` / `flash_attention_score_grad.cpp` / `.h` | 249/171/37 | api | supporting | 接口声明、注册 |
| `docs/FAG算子设计介绍.md` | 404 | doc | core | 6 步反向公式 L9-14、3 阶段入口 L20-23、arch32/arch35 tiling L29-72、流水 L74-110、多模板 L111-367、编程视角 L368-405 |
| `docs/aclnnFlashAttentionScoreGrad.md` | 749 | doc | core | V1 接口（A2 训练 √，950PR ×） |
| `docs/aclnnFlashAttentionScoreGradV2.md` | 786 | doc | core | V2：新增 pseType（A2 √，950PR ×） |
| `docs/aclnnFlashAttentionScoreGradV3.md` | 852 | doc | core | V3：新增 sinkInOptional/dsinkOut（950PR 暂不支持 sink；A2 √） |
| `docs/aclnnFlashAttentionScoreGradV4.md` | 1006 | doc | core | V4：合并 V2+UnpaddingV2、多等/不等长序列、Dropout 调整（950PR √，A2 ×） |
| `docs/aclnnFlashAttentionUnpaddingScoreGrad*.md` (V1-V5) | 773-920 | doc | core | 变长/TND 接口族 |
| `examples/test_aclnn_flash_attention_score_grad_*.cpp` | 271-304 | example | indirect | V2/V4(fp16/fp32/sparse/varlen) 调用样例 |
| `tests/pytest/` (`cpu_impl.py` 1983、`npu_impl.py` 373、`test_*.py`) | — | test | indirect | CPU/NPU 参考实现与用例 |
| `tests/ut/op_host/arch32|arch35/test_flash_attention_score_grad_tiling.cpp` | 517/1192 | test | indirect | tiling 单测 |
| `CMakeLists.txt` / `op_host/CMakeLists.txt` / `tests/**/CMakeLists.txt` | — | build | indirect | 构建配置 |

## 共享依赖（算子目录外）

- `attention/common/`、`attention/flash_attention_score/`（前向算子，共享 softmax/mask 工具与设计）、仓库级 `tiling_base/`、`register/`、`kernel_operator.h` 等 CANN 框架头。本算子 tiling 通过 `TilingRegistryArch::DoTilingImpl`（tiling.cpp L394）分发到 arch32/arch35 各注册 tiling 类。
