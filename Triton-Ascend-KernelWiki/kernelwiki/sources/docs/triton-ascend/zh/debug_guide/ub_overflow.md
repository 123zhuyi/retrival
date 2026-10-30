---
id: doc-triton-ascend-zh-debug-guide-ub-overflow
title: UB Overflow 问题排查指南
repo: triton-lang/triton-ascend
url: https://github.com/triton-lang/triton-ascend/blob/e4c692be02c38f3d5e99ccaf314bf4148fa84413/docs/zh/debug_guide/ub_overflow.md
source_category: official-doc
tags:
- triton_ascend
- official_doc
- debug-guide
retrieved_at: '2026-07-15'
upstream_commit: e4c692be02c38f3d5e99ccaf314bf4148fa84413
upstream_path: docs/zh/debug_guide/ub_overflow.md
content_sha256: 1fd74ec73da8c8dc68aaeb28738ee8bfcd8f79309c78f4f885aed3e2dc2f537a
snapshot_body_sha256: f80b32784b469ff0ad147547225671b17ac5c07773fed10b1764e35de0d62292
document_language: zh
source_format: md
license: MIT
generated_by: scripts/ingest_triton_ascend_docs.py
symptoms:
- ub_overflow
hardware_features:
- unified_buffer
---

# UB Overflow 问题排查指南

## 概述

UB Overflow 是 Triton-Ascend 开发中常见的问题。本文档详细介绍 UB Overflow 的常见原因、解决方案和调试方法。

## 常见原因及解决方案

### 1. 使用了会增加 UB 开销的接口参数

某些接口在特定参数配置下会自动添加额外的处理逻辑，导致 UB 空间使用增加：

#### `tl.maximum`、`tl.minimum`、`tl.clamp` 接口的 `propagate_nan` 参数

**问题描述：**
当设置 `propagate_nan=tl.PropagateNAN.NONE` 时，系统会自动添加 NaN 值检测和处理逻辑。

**影响：**

- 显著增加 UB 空间使用
- 可能导致性能下降

**解决方案：**

- 如果输入数据不包含 NaN 值或不需要严格的 NaN 处理语义，考虑调整 `propagate_nan` 参数值
- 在 UB 空间紧张的场景下，优先选择不会触发额外 NaN 处理的参数配置

### 2. 中间变量过多

**问题：**
kernel 中定义了大量临时张量或中间计算结果

**解决方案：**

- 减少不必要的中间变量
- 复用已分配的缓冲区
- 将大计算拆分为多个小 kernel

### 3. shape 过大

**问题：**
处理高维/大 shape 张量

**解决方案：**

- 考虑将大 tensor 分块处理
- 修改分块策略，减小每块的大小

## 调试建议

1. **开启详细日志**
   - 使用 `TRITON_DEBUG=1` 查看详细的编译信息
   - 定位具体哪个算子导致 UB 溢出

2. **逐步排查**
   - 通过注释掉部分代码，定位导致问题的具体操作

3. **参考文档**
   - 查看各接口文档中的"特殊限制说明"部分
   - 了解可能增加 UB 开销的参数配置

4. **优化策略**
   - 优先处理占用 UB 空间较大的算子
   - 考虑重新设计算法以减少中间变量
   - 考虑修改分块策略
