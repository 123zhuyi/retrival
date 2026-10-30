---
id: ops-transformer-nsa-compress-with-cache
title: CANN ops-transformer NsaCompressWithCache source snapshot
type: source
repo: kunpengcompute/CANN-ops-transformer
path: attention/nsa_compress_with_cache/
commit: d28d8f99a8858b71caa27a535da32bb5ba68ce2a
retrieved_at: 2026-07-29
license: CANN Open Software License Agreement Version 2.0
fixture: true
---

# CANN ops-transformer NsaCompressWithCache 来源摘要

本页记录从 CANN ops-transformer 仓库 `attention/nsa_compress_with_cache/` 目录抽取的不可变来源信息。固定提交 `d28d8f9...`。`fixture: true`。支持 A2/A3 + KirinX90/Kirin9030。

## 1. 仓库与算子定位

- 仓库根：`kernelwiki/artifacts/ops-transformer/`
- 算子目录：`attention/nsa_compress_with_cache/`
- 固定提交：`d28d8f99a8858b71caa27a535da32bb5ba68ce2a`
- 抓取日期：2026-07-29
- 许可证：CANN Open Software License Agreement Version 2.0

## 2. 核心源文件 SHA-256

| 源文件 | SHA-256 |
|---|---|
| `README.md` | `82d127c308cf4731be3499f18f1b0e34cef4c0694023743bf824b04328e8dd8b` |
| `docs/aclnnNsaCompressWithCache.md` | `69ca072adafe81cc44768d0f6e140bd42b5226df46e70a92e78258b69a261093` |
| `op_host/nsa_compress_with_cache_def.cpp` | `b48e0085036c4ab4e3e9cb020b9814be81482af6c64202285334644ec6ba2d6a` |
| `op_host/nsa_compress_with_cache_tiling.h` | `d5b233357b2c79d7c2ce8e984d12fc032bc6906f26d1f889cc0a9c81943ef7e7` |
| `op_kernel/nsa_compress_with_cache.cpp` | `725b2fecb646e6daf0e68d66bac09de9d5cbce1108cf51c97278ffdc3b9c219e` |

## 3. 关键代码区域摘要

### 3.1 功能（README L17-29）
NSA 推理 KV 压缩：当 batch 的 seqLen 满足 `(seqLen-compressBlockSize)%stride==0`，对该 batch 后 compressBlockSize 个 token 用 weight 做加权求和，通过 slot_mapping 写回 output_cache。

### 3.2 算子定义（def.cpp L21-61）
- 输入：input/weight/slot_mapping/output_cache（REQUIRED，FP16/BF16，ND）+ act_seq_len/block_table（OPTIONAL）。
- 输出：output_cache（REQUIRED，原地更新）。
- 属性：compress_block_size/compress_stride/act_seq_len_type/page_block_size（均为必选 int）。
- AICore config：ascend910b/910_93 + kirinX90/9030（仅 FP16）。

### 3.3 tiling 与分核（tiling.h L24-50）
`coreStartBatchIdxList[50]`/`coreStartHeadIdxList[50]` 按 head×batch 分核。`headsNumPerCore`、`coresNumPerCompress`、`compressBlockSize`/`compressStride`。

### 3.4 约束（README L93-105）
compressBlockSize/stride 16 倍数，compressBlockSize≤64，headDim≤256（16 倍数），headNum≤64，pageBlockSize=64/128。

## 4. 衍生引用

- `wiki/kernels/nsa-compress-with-cache.md`
- `artifacts/nsa-compress-with-cache/`

## 5. 复现性说明

本来源页仅记录源文件路径、SHA-256 与关键代码区域摘要。tiling 参数数值、weight 初始化方式（`weightInitFlag`）等需结合硬件验证，相关结论在引用 wiki 页中标记 `confidence: inferred`。