---
id: ops-transformer-incre-flash-attention
title: CANN ops-transformer IncreFlashAttention source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/incre_flash_attention/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer IncreFlashAttention 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/incre_flash_attention/` 目录抽取的不可变来源信息。仓库快照固定提交为 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`。本仓库为本地制品目录，`fixture: true` 表示来源摘录用于知识库分析，不做整仓复制。

## 1. 仓库与算子定位

- 仓库根: `kernelwiki/artifacts/ops-transformer/`
- 算子目录: `attention/incre_flash_attention/`
- 固定提交: `d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期: 2026-07-29
- 许可证: CANN Open Software License Agreement Version 2.0

## 2. 核心源文件清单与 SHA-256

| 源文件（相对仓库根） | SHA-256 | 作用 |
|------|------|------|
| `attention/incre_flash_attention/README.md` | `91148eb65185da8e9f3095afc6a58be83a849f29cdb1fa4e5e7fa67f2bceb296` | 算子总览、公式、参数、约束（基础版） |
| `attention/incre_flash_attention/docs/IFA算子设计介绍.md` | `66948064d6b6914cca0c5d655c3f8eb1e2feddb1e13c03ec9d45d9613570b5ba` | 实现原理、模板设计、切分、FlashDecode、AntiQuant-MSD、PageAttention、GQA、TilingKey |
| `attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md` | `44e4cc9ef651e844bb6660620d8b49b3a49b5be7a241c5e4cf6c7247f16e61b4` | V4 aclnn API（V4 新增 KV 左 padding） |
| `attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV3.md` | `b03f0c79886f48f7f8a322659c165a9cd5343a60c27fbbc3880193419b16cb6f` | V3 aclnn API |
| `attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV2.md` | `2af5acda05677d7d51c00ed72d9abd49a3ffdba1c7bda6527ca8fb028e0aa2c9` | V2 aclnn API |
| `attention/incre_flash_attention/op_host/incre_flash_attention_def.cpp` | `fe0be89bbe3fe2c4614875d6e368d4a8808bbceb409daf78ea00353041a2de24` | 算子定义（输入输出 dtype/layout/属性/AICore config） |
| `attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp` | `c84dbe37632ed428c080918ce4ca124d43640fbcfd05e69c212b9453f2b46a74` | arch32 tiling 主逻辑（arch 分发/分核/FlashDecode/MLA/PA/workspace/GenTilingKey） |
| `attention/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp` | `a8590eddd3487f9a2e74611d0608be3cd4d97783cfe7dd83ecb297ad9b54fc4d` | arch35 tiling 主逻辑 |
| `attention/incre_flash_attention/op_host/incre_flash_attention_tiling.h` | `e68b2d10e40bc9dda0bb31f5cb31f006342db4100c9f2d5aa0ccc448358cdae7` | tiling 数据结构 |
| `attention/incre_flash_attention/op_kernel/incre_flash_attention_tilingkey.h` | `3d832c813a18be5b1ae6b783f9c7e6e0b3f5bd2bbb83508a57447d64be24ab85` | arch32 tilingkey 宏（十进制 16 位组装） |
| `attention/incre_flash_attention/op_kernel/incre_flash_attention_arch32.h` | `20cb2397d84cf5d5386ebc09b6aa79eacfd3f32d956309c1b4e7f3e2690ef63b` | arch32 kernel 主体（按 tiling key 分发模板） |
| `attention/incre_flash_attention/op_kernel/arch32/incre_flash_attention_split_Bbn2s2_Us2.h` | `a5ef1c63be0303e47a9af91933ee5c31f59c51ce1d43cb9c6044a3fe5a03390d` | C+V 基础模板 |
| `attention/incre_flash_attention/op_kernel/incre_flash_attention_allvec_new.h` | `1777ac030c0bba3bd1857efb7877f127d8fd94b6b14c3716eeca1ee27f1eac42` | 全 Vector 模板（位于 op_kernel/，非 arch32/） |
| `attention/incre_flash_attention/op_kernel/arch32/incre_flash_attention_preload.h` | `0fa3095cc72799a6c910e937fb42f1e131742fd36405c001967bbcfde024b72a` | matmul 基础 API 模板（CUBE 视角 + N-Buffer 流水） |
| `attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_template_tiling_key.h` | `970ffe440799c4c8c008416366f9b2ed5cb5054646994c4d4799a792a85d90b4` | arch35 模板分发（if constexpr） |
| `attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h` | `39223a41a750a95235febf1037affaf216c294d8732779f034583e5f2f3f78a9` | arch35 kernel 入口 |
| `attention/incre_flash_attention/op_kernel/arch35/vector_api/vf_softmax_const.h` | `1356e612b02493ea3541d070f6f0cf659470a2d0b42a282886c23c468d9bcc59` | arch35 在线 FlashSoftmax 实现 |

## 3. 关键代码区域摘要

### 3.1 计算公式（来源: README.md L30-40）

`Attention(Q,K,V) = Softmax(QK^T / √d) · V`。IFA query S1=1，`QK^T` 为 1×S2 行向量；采用 FlashSoftmax 在线刷新，FP32 中间累加。

### 3.2 FlashSoftmax 在线刷新（来源: IFA算子设计介绍.md L22-32）

1. 按 Skv 轴切分，每次处理一个 SkvSplit。
2. 从第二次循环起记录 `exp[i] = e^(max[i-1] - max[i])`。
3. `MM[PV][i] = MM[PV][i-1]·exp[i] + MM_new[PV][i]`。
4. 最终按行除以 softmax_sum 写回 attention_out。

### 3.3 FlashDecode 规约（来源: IFA算子设计介绍.md L73/119-138）

`BN2 < 0.4 × 总核数` 时对 S2 外切 SplitKv 份，总块数 `BN2 × SplitKv`，各核独立 Attention 后 lse 加权规约：`scale[i] = exp(lse[i]) / Sum(exp(lse[i]))`，`res = Σ split_res[i] · scale[i]`。

### 3.4 三模板设计（来源: IFA算子设计介绍.md L40-57）

- C+V 基础模板（`split_Bbn2s2_Us2.h`，perfMode 0）：Cube 做 matmul，顺序流水。
- 全 Vector 模板（`allvec_new.h`，perfMode 1）：matmul 由 vector 实现，降 CV 通信开销。
- matmul 基础 API 模板（`preload.h`，perfMode 3）：CUBE 视角 + N-Buffer 流水。
- `preload_dd`（perfMode 6）与 `preload_mla`（perfMode 5）设计介绍 L54/L56 标注**仅 FIA 调用，IFA 不调用**。

### 3.5 架构分发逻辑（来源: incre_flash_attention_def.cpp L294-408）

`ascend910b`/`ascend910_93` 用 opFile `incre_flash_attention`（arch32）；`ascend950` 用 opFile `incre_flash_attention_apt`（arch35，`jitCompile static_false,dynamic_false`）；`ascend310p` 单独 config（仅 FP16），opFile `incre_flash_attention`。

### 3.6 TilingKey 16 位十进制组装（来源: IFA算子设计介绍.md L189-244）

`GenTilingKey`（`tiling.cpp` L3409-3470）按十进制位组装：位0 layout / 位1 Q dtype / 位2 KV dtype / 位3 output dtype / 位5[bit0] splitKv(FlashDecode) / 位5[bit1] PA / 位5[bit2] antiquantMode / 位6 antiquantMode_ / 位15 perfMode(模板) / 位16 mode(IFA Base/SysPrefix)。

### 3.7 AntiQuant-MSD / PageAttention / GQA（来源: IFA算子设计介绍.md L140-173）

- AntiQuant-MSD（L140-155）：Q 展开多行 int8 -> `AA·B`（int8×int8=int32）-> reduce，避免对大 KV 矩阵反量化。
- PageAttention（L156-158）：block_table 驱动，MatMul 回调 GM->L1 拷贝。
- GQA（L160-173）：G 轴 UB 驱动切分，`g = target_ub_size / column_size`。

### 3.8 算子定义的 dtype/layout（来源: incre_flash_attention_def.cpp）

- query: FP16/BF16（S=1）；key/value: FP16/BF16/INT8（tensorlist）；attention_out: FP16/BF16/INT8
- 输入 layout: BSH/BNSD/BSND
- 属性: num_heads(必需), scale_value(默认1.0), input_layout(默认"BSH"), num_key_value_heads(默认0), block_size(默认0), inner_precise(默认1)
- 支持架构: ascend910b, ascend910_93, ascend950, ascend310p

## 4. 衍生引用

本来源页被以下 wiki 页和 artifact 引用:
- `wiki/kernels/incre-flash-attention.md`
- `artifacts/incre-flash-attention/`（source_file_inventory.md / migration-table.md / tiling_dispatch_tables.md / pipeline-diagram.md / incre-flash-attention-triton-skeleton.py / PROVENANCE.yaml）

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 和关键代码区域摘要，不包含完整源码。要复现分析，需 checkout 到固定提交 `d28d8f99a8858b71caa27a535da32bb5ba68ce2a` 并按上述路径读取原文件。tiling 参数具体数值、arch35 模板枚举、V4 在 950 支持范围、FlashDecode lse 精确公式等需结合源码/硬件进一步验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。
