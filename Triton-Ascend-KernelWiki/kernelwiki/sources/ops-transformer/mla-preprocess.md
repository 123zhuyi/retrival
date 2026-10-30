---
id: ops-transformer-mla-preprocess
title: CANN ops-transformer MlaPreprocess source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/mla_preprocess/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer MlaPreprocess 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/mla_preprocess/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`（与 [[ops-transformer-flash-attention-score]] 同源同提交）。`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。无 arch 子目录（单架构 A2/A3）。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/mla_preprocess/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-29
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

完整清单（含行数、role、centrality、关键符号、行号区域）见 `artifacts/mla-preprocess/source_file_inventory.md`。关键文件 SHA-256：

| 源文件（相对算子目录） | SHA-256 |
|---|---|
| `README.md` | `ee3b0e00742a48fbcb7a9ac6df046916e8c624044ef7e048088bc0d6bf5ce32e` |
| `docs/aclnnMlaPreprocess.md` | `9cc3ea3e90c1a13d272749160600d22c69fdfa0173c6583ee7f9bc397c56c059` |
| `op_host/mla_preprocess_def.cpp` | `59aa04b833d7be9e27b7fcdee1167f3b21966431ae8e9932a20f8300929d460d` |
| `op_host/mla_preprocess_infershape.cpp` | `4a86d754d252eb70c87d681c7bb6eb47c38ba6f309e212e987c7daa23dd4f9e8` |
| `op_host/mla_preprocess_tiling.cpp` | `d029a7fad6431a1d4f130a145d503fa91161f9c1d2f44c3dbdd6c3736c5a5c88` |
| `op_host/mla_preprocess_tiling.h` | `3d0f2c6adecab68ba9f6ebd6ff93eec9ae2ec25e9c42140d4bb5f2dc0386065e` |
| `op_host/mla_preprocess_tilingdata.h` | `cd63a365c0ebb2788c756e305558eec2d2a4320f3e2fe82b7507201de25c4f68` |
| `op_kernel/mla_preprocess.cpp` | `742fed9146ac78693d74bb48d6325ab74e5d579e38246fe64fb4c63535e668cb` |
| `op_kernel/mla_preprocess_fp16.h` | `76860e49f3e9a0f03d28c2b820bb22de21138fe12907e21cf5deb22419f862cd` |
| `op_kernel/mla_preprocess_bf16.h` | `9ff99ca4483551f6329104c8de501f6a62ca2c194767a56fa7ca548bf41701a1` |
| `op_kernel/mla_preprocess_no_quant.h` | `05704a08e0891053c718134a0b69da4dbda6ec6ec9ad0f36c427f3eba02a38f1` |
| `op_kernel/mla_kernel_utils.h` | `cdb4ee7c156907cd02e03f1adbd6019d5dba1a0e4cb5f28f9623a6b70efc75e2` |
| `op_kernel/mla_mma.h` | `9024e68902ff832a9d0bb15832ec992265ed92a770e93567aa51ad9f50f62cda` |
| `op_kernel/mla_simd.h` | `c393aca1041b0a32ce74f61ae1e5d23b32da3b5134acda1a7a84f1f0f3a06673` |

`op_kernel/*_iterator.h`（gm_to_l1/gm_to_ub/l0c_to_gm/l1_to_l0 等 9 个 buffer 搬运迭代器）SHA-256 见 `source_file_inventory.md`。

## 3. 关键代码区域摘要

### 3.1 计算公式与数据流（README L10-54）
MLA 前处理融合：`x -> RmsNormQuant₁ -> ×W^DQKV(mm1) -> 拆 Q/K`。
- Q 通路：`RmsNormQuant₂ -> ×W^UQ(mm2) -> 拆 nope/rope`；nope `×W^UK(mm3) -> q^N`；rope `RoPE -> q^R`。
- K 通路：`RmsNorm₃ -> Cache -> k^N`；`RoPE -> Cache -> k^R`。
- `q^N = RmsNormQuant(x)·W^DQKV·W^UK`（数学 2-matmul 形式，MLA 权重吸收 W^UQ·W^UK 折叠；kernel 实现 3 matmul，吸收与否 `needs_verification`）。

### 3.2 算子定义（def.cpp L73-119）
- 输入：input/gamma_0/beta_0/quant_scale_0/quant_offset_0/wdqkv/descale_0/bias_0/gamma_1/beta_1/quant_scale_1/quant_offset_1/wuq/descale_1/bias_1/gamma_2/cos/sin/wuk/kv_cache/kv_cache_rope/slot_mapping/ctkv_scale/q_nope_scale（多为 REQUIRED，NZ/ND 混合格式）。
- 输出：q_out/kv_cache_out/q_rope_out/kr_cache_out。
- 属性：wdq_dim(128)/q_rope_dim(64)/k_rope_dim(64)/epsilon(1e-5)/q_rotary_coeff(2)/k_rotary_coeff(2)/transepose_wdq/wuq/wuk(true)/cache_mode(0)/quant_mode(0)/do_rms_norm(true)/wdkv_split_count(1)（多项为预留参数，只支持单值）。
- AICore config：ascend910b + ascend910_93（仅 A2/A3）。

### 3.3 tiling 与 dispatch（tiling.h/tilingdata.h/mla_preprocess.cpp）
- CV 分离：Cube 做 3 matmul + RoPEConcat（`RopeConcatTiling`）；Vector 做 RmsNormQuant + EinSumQuant（`RmsNormQuantTiling`/`EinSumQuantTiling`）。kernel 先 `ProcessCube()` 再 `ProcessVector()`（mla_preprocess.cpp L52-53）。
- dtype dispatch：FP16 -> `_fp16.h`；BF16 -> `_bf16.h` + `_no_quant.h`（L17-22）。
- tiling key = cacheMode × 3 输出格式(ND/NZ)（L46+，步长 4，8 组合/cacheMode）。
- 3 matmul tiling mm1/mm2/mm3（基本块 m0/k0/n0 + loop + swizzle + ping-pong）。
- workspace 3 等大 section（s1/s2/s3，L41-43）。

### 3.4 约束（README L363-371）
tokenNum 0-256，hiddenSize 2048-10240（256 倍数），headNum {16,32,64,128}，blockNum 192，blockSize 128。BF16 权重 -> input BF16 且 hiddenSize 仅 6144。

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用：
- `wiki/kernels/mla-preprocess.md`
- `artifacts/mla-preprocess/`（source_file_inventory.md、tiling_dispatch_tables.md、pipeline-diagram.md、migration-table.md、mla-preprocess-triton-skeleton.py）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数数值、workspace 字节预算、MLA 权重吸收实现、quantMode 1/2/3（未实现）等需结合硬件/源码进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
