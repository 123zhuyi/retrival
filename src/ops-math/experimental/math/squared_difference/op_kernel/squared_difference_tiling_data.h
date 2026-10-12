/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file squared_difference_tiling_data.h
 * \brief SquaredDifference tiling data struct (shared by host & kernel)
 */

#ifndef _SQUAREDDIFFERENCE_TILING_DATA_H_
#define _SQUAREDDIFFERENCE_TILING_DATA_H_

// 合轴后最大维度数
constexpr int32_t SD_MAX_DIM = 8;

// DataCopyPad blockCount 的接口上限。
constexpr int64_t SD_DATACOPY_MAX_BLOCK_COUNT = 4095;

constexpr int64_t SD_UB_BLOCK_SIZE = 32;

// 运行模式
constexpr int32_t SD_MODE_ONEDIM = 0; // 合轴后 1 维（same-shape / 标量广播）
constexpr int32_t SD_MODE_BRC = 1;    // 合轴后 >1 维（多维广播）

// dtype 键（host 与 kernel 约定）
constexpr int32_t SD_DT_FP32 = 0;
constexpr int32_t SD_DT_FP16 = 1;
constexpr int32_t SD_DT_BF16 = 2;
constexpr int32_t SD_DT_INT32 = 3;
constexpr int32_t SD_DT_INT64 = 4;

struct SquaredDifferenceTilingData {
    int32_t mode;        // SD_MODE_*
    int32_t dtypeKey;    // SD_DT_*
    int32_t shapeLen;    // 合轴后维度数
    int32_t ubSplitAxis; // 多维分支：切分轴（= shapeLen-2）

    int64_t outDims[SD_MAX_DIM];    // 合轴后输出各维大小
    int64_t x1Strides[SD_MAX_DIM];  // 合轴后 x1 各维 stride（广播轴=0）
    int64_t x2Strides[SD_MAX_DIM];  // 合轴后 x2 各维 stride（广播轴=0）
    int64_t outStrides[SD_MAX_DIM]; // 合轴后输出各维 stride（连续）

    // OneDim 分支
    int64_t totalLength; // 输出总元素数
    int32_t x1Scalar;    // 1=x1 合轴后为标量（广播）
    int32_t x2Scalar;    // 1=x2 同上

    // 通用 Tiling
    int64_t ubFormer;    // OneDim: 每 tile 元素数；BRC: 每 tile 的 M（切分轴行数）
    int64_t ubOuter;     // 切分轴方向的 tile 数
    int64_t ubTail;      // 切分轴方向最后一个 tile 的大小
    int64_t innerDim;    // BRC: N（内轴元素数，连续）
    int64_t alignInner;  // BRC: 单 tile 内轴 pitch，向上对齐到 32B（保证行子偏移 32B 对齐）
    int64_t maxTileElem; // 单 tile 最大元素数（BRC = ubFormer*alignInner）

    // BRC 内轴 N 切分（N 单行超 UB 预算时启用；N 放得下时 nFormer=N, nOuter=1）
    int64_t nFormer; // 每 N-tile 的列数
    int64_t nOuter;  // N 方向的 tile 数
    int64_t nTail;   // 最后一个 N-tile 的列数

    // 多核切分
    int32_t blockPolicy;    // 0=legacy 核数；1=大 BRC 满核策略（仅 telemetry）
    int64_t blockNum;       // 实际启用核数 B，非空时 1 <= B <= min(coreNum, fusedProduct)
    int64_t blockBase;      // 商 q=floor(fusedProduct/blockNum)
    int64_t blockRemainder; // 余数 r=fusedProduct%blockNum
    int64_t fusedProduct;   // BRC: ubOuter * outerProd * nOuter；OneDim: ubOuter

    // BRC 快速路径（单轴广播，对齐 TBE：inner 合并广播轴 + 紧凑展开 + 连续搬移）
    // brcKind: 0=旧路径；2=尾维广播（kind=2）
    int32_t brcKind;
    int64_t broadcastLen; // 广播轴长度 outDims[b]
    int64_t innerSrc;     // 广播输入的连续源大小(每个外层位置) = inner / broadcastLen
    int32_t brcWhich;     // 1=x1 广播，2=x2 广播
    int64_t outerTotal;   // 外层位置总数 = prod(outDims[0..b-1])
    int64_t srcTileElems; // 快速路径：单 tile 的广播源元素数（x2 搬入缓冲大小）

    // int64 广播（广播轴=M，N 切分）路径：整广播轴一个 tile，读广播源一次、标量跨 M 广播
    int32_t bcastOnM; // 1=启用该路径
};

#endif // _SQUAREDDIFFERENCE_TILING_DATA_H_
