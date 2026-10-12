/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file as_strided_struct.h
 * \brief define tiling data of AsStrided
 */

#ifndef OP_KERNEL_AS_STRIDED_STRUCT_H_
#define OP_KERNEL_AS_STRIDED_STRUCT_H_

#include <cstdint>

const int64_t TILING_ARRAY_LEN = 10;
const int64_t TILING_NDDMA_LEN = 5;

struct AsStridedTilingData {
    int64_t storageOffset = 0;
    uint32_t blockNum = 0;
    uint32_t loopsTailCore = 0;
    uint32_t tilingAxisIdx = 0;
    uint32_t outerAxisFactor = 0;
    uint32_t innerAxisFactor = 0;
    uint32_t outerAxisNum = 0;
    uint32_t innerAxisNum = 0;
    uint32_t loopsPerCore = 0;
    uint32_t ubFactor = 0;
    uint32_t ubFactorTail = 0;
    uint32_t ubSize = 0;
    uint32_t innerAxisFactorTail = 0;
    uint32_t axisOutTotalFactor = 0;
    uint32_t en32BAligned = 0;

    int32_t innerAxis[TILING_ARRAY_LEN] = {0};
    int32_t outStrideArr[TILING_ARRAY_LEN] = {0};
    int32_t outLoopArr[TILING_ARRAY_LEN] = {0};
    uint32_t nddmaLoop[TILING_NDDMA_LEN] = {0};
    uint32_t nddmaTailLoop[TILING_NDDMA_LEN] = {0};
    uint64_t nddmaSrcStride[TILING_NDDMA_LEN] = {0};
    uint32_t nddmaDstStride[TILING_NDDMA_LEN] = {0};
    int32_t gmOutStride[TILING_ARRAY_LEN] = {0};
    int32_t gmShape[TILING_ARRAY_LEN] = {0};
    int32_t gmInStride[TILING_ARRAY_LEN] = {0};
};

struct AsStridedSimtTilingData {
    uint32_t outDimNum{0};
    uint32_t blockNum{0};
    int64_t storageOffset{0};
    int64_t mainBlockFactor{0};
    int64_t tailBlockFactor{0};
    uint32_t sizeArr[TILING_ARRAY_LEN] = {0};
    uint32_t strideArr[TILING_ARRAY_LEN] = {0};
    uint32_t outSizeStride[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // 输出shape每维相邻元素步长
};

struct AsStridedEmptyTilingData {
    uint32_t blockNum{0};
};

struct AsStridedZeroStrideTilingData {
    uint32_t blockNum{0};
    uint32_t ubSizePlatForm{0};
    int64_t storageOffset{0};
    int64_t mainBlockFactor{0};
    int64_t tailBlockFactor{0};
};

struct UbParam {
    uint32_t innerAxisFactor{1};       // UB切分轴的内部轴
    uint32_t innerAxisFactorTail{0};   // UB切分轴的内部尾轴
    uint32_t outerAxisFactor{0};       // UB切分轴的外部轴
    uint32_t ubFactor{0};
    uint32_t ubFactorTail{0};
    uint32_t loopsPerCore{0};
};

struct AsStridedWithGatherTilingData {
    int64_t storageOffset{0};
    uint64_t ubSizePlatForm{0};
    uint32_t tilingAxisIdx{0};     // ub切分轴
    uint32_t preSize{1};
    uint32_t blockNum{0};
    uint32_t mainBlockCnt{0};
    uint32_t outDimNum{0};
    uint32_t inUbSize = 0;         // x need ub size, B
    uint32_t inDataLen = 0;        // move x len, element
    uint32_t blockAxisIdx{0};
    uint32_t coreCurAxisFactor{0};         // 左→右到当前轴，轴的前缀积
    uint32_t coreInnerAxisFactor{0};       // 核切分轴的内部轴
    uint32_t coreInnerAxisTailFactor{0};   // 核切分轴的内部尾轴
    uint32_t coreOuterAxisFactor = 0;       // 核切分轴的外部轴
    UbParam mainBlockUbParam;
    UbParam tailBlockUbParam;
    uint32_t sizeArr[TILING_ARRAY_LEN] = {0};
    uint32_t strideArr[TILING_ARRAY_LEN] = {0};
    uint32_t idxStrideArr[TILING_ARRAY_LEN] = {0};   // ub切分轴向左前缀积
};

struct AsStridedTilingParam {
    int64_t storageOffset = 0;
    uint32_t ubFactor = 0;
    uint32_t ubFactorTail = 0;
    uint32_t preSize = 1;
    uint32_t numCore = 0;
    uint32_t innerAxisFactor = 1;
    uint32_t outerAxisFactor = 0;
    uint32_t tilingFlag = 0;
    uint32_t tilingAxisIdx = 0;
    uint32_t curAxisFactor = 0;
    uint32_t outerAxisNum = 0;
    uint32_t innerAxisNum = 0;
    uint32_t innerAxisFactorTail = 0;
    uint32_t blockNum = 0;
    uint32_t axisOutTotalFactor = 0;
    uint32_t ubSize = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t sizeofDtype = 0;
    uint32_t loopsPerCore = 0;
    uint32_t ubUseFactor = 0;
    int64_t tilingKey = 0;
    uint32_t en32BAligned = 0;
    bool movealignFlag = false;
    bool dualCutFlag = false;
    uint32_t outDimNum = 0;
    uint32_t inputSize = 0; // input元素数
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    int32_t innerAxis[TILING_ARRAY_LEN] = {0};
    uint32_t nddmaDstStride[TILING_NDDMA_LEN]  = {1, 1, 1, 1, 1};
    uint32_t nddmaLoop[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    uint32_t nddmaTailLoop[TILING_NDDMA_LEN] = {1, 1, 1, 1, 1};
    int32_t outLoopArr[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    uint64_t nddmaSrcStride[TILING_NDDMA_LEN] = {0};
    int32_t outStrideArr[TILING_ARRAY_LEN] = {0};
    int32_t gmOutStride[TILING_ARRAY_LEN] = {0};
    int32_t gmShape[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int32_t gmInStride[TILING_ARRAY_LEN] = {0};
    uint32_t strideArr[TILING_ARRAY_LEN] = {0};
    uint32_t sizeArr[TILING_ARRAY_LEN] = {0};
    uint32_t outSizeStride[TILING_ARRAY_LEN] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
};

struct AsStridedUbGatherParam {
    int64_t storageOffset = 0;
    uint32_t preSize = 1;
    uint32_t tilingFlag = 0;        // 切分完成标志，0:未完成，1:完成
    uint32_t tilingAxisIdx = 0;     // ub切分轴
    uint32_t blockNum = 0;
    uint32_t blockNumMin = 0;       // 单核至少处理一个cacheline的开核数
    uint32_t mainBlockCnt = 0;
    uint64_t ubSizePlatForm = 0;
    uint32_t outDimNum = 0;
    uint32_t inUbSize = 0;
    uint32_t inDataLen = 0;

    uint32_t blockAxisIdx = 0;
    uint32_t coreCurAxisFactor = 0;         // 左→右到当前轴，轴的前缀积
    uint32_t coreInnerAxisFactor = 0;       // 核切分轴的内部轴
    uint32_t coreInnerAxisTailFactor = 0;   // 核切分轴的内部尾轴
    uint32_t coreOuterAxisFactor = 0;       // 核切分轴的外部轴

    UbParam mainBlockUbParam;
    UbParam tailBlockUbParam;
    uint32_t sizeArr[TILING_ARRAY_LEN] = {0};
    uint32_t strideArr[TILING_ARRAY_LEN] = {0};
    uint32_t idxStrideArr[TILING_ARRAY_LEN] = {0};   // ub切分轴向左前缀积
};

#endif  // OP_KERNEL_AS_STRIDED_STRUCT_H_