---
kind: index
type: section_index
title: ops/ 算子参考卡
description: 从 Triton-Ascend-KernelWiki 迁移的 48 张算子语义 / tiling / dispatch 参考卡（operator_spec）。当前为静态分析级知识（inferred），后续挑高频算子改写成成功案例。
---

# ops/ 算子参考卡

本目录收录 48 张算子级参考卡，来源为 A 的 `kernelwiki/wiki/kernels/`。每张卡记录算子语义、输入输出、tiling key / dispatch、Agent actions 与 Triton 骨架，置信度 `inferred`（源码静态分析，未硬件实测）。

案例卡与 `runbooks/` 条目经「实例化 ⇄ 已知实例」双向回链，后续由 `triton-knowledge-*-ingest` skill 按需增补为真正的成功案例。
