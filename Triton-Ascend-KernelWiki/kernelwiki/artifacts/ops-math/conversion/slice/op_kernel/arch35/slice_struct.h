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
 * \file slice_struct.h
 * \brief slice struct
 */
#ifndef SLICE_TILINGDATA_H_
#define SLICE_TILINGDATA_H_

#include <cstdint>

constexpr int64_t MAX_AXIS_NUM_FOR_STRIDESLICE = 8;
constexpr int64_t MAX_NDDMA_UB_SPLIT_AXIS_NUM = 5;
constexpr int64_t MAX_SIMT_UB_SPLIT_AXIS_NUM = 8;
constexpr int64_t NUMBER_TWO = 2;

struct SliceBaseTilingData {
    int8_t isBeginConst;
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t ubInLoopSteps;
    int32_t ubSize;
    int32_t ubFactor;
    int32_t ubTailFactor;
    int32_t ubTailTailFactor;
    int16_t realCoreNum;
    int16_t inputDims;
    int16_t blkIndex;
    int16_t ubIndex;
    int64_t outputShape[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t begin[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t inputSteps[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t rowsOffsetSteps[MAX_AXIS_NUM_FOR_STRIDESLICE];
};

struct StridedSliceMoveAlignParams2 {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
    uint16_t loop1Size;
    uint16_t loop2Size;
    uint32_t loop1SrcStride;
    uint16_t loop1DstStride;
    uint32_t loop2SrcStride;
    uint16_t loop2DstStride;
};

struct SliceMoveAlignLastDimTilingData {
    SliceBaseTilingData sliceBaseTilingData;
};

struct SliceMoveAlignParams {
    uint16_t blockCount;
    uint32_t blockLen;
    uint32_t srcStride;
    uint32_t dstStride;
};

struct SliceMoveAlignLast2DimTilingData {
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t ubInLoopSteps;
    int64_t ubOutLoopSteps;
    int32_t ubSize;
    int32_t ubFactor;
    int32_t ubTailFactor;
    int32_t ubTailTailFactor;
    int16_t realCoreNum;
    int8_t isBeginConst;
    SliceMoveAlignParams moveAlignParams;
    int64_t outputShape[NUMBER_TWO];
    int64_t begin[NUMBER_TWO];
    int64_t inputSteps[NUMBER_TWO];
};

struct SliceMoveAlignGatherTilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t ubOutLoopSteps;
    int32_t ubSizeInput;
    uint32_t lastOneInputDim;
    uint32_t outBlockLen;
    StridedSliceMoveAlignParams2 moveAlignParams;
};

struct StridedSliceTilingData2 {
    int64_t ubSize;
    int64_t ubSizeInput; // stride为负时使用
    int64_t coreNum;
    int64_t ubIndex;
    int64_t ubFactor;
    int64_t ubTailFactor;
    int64_t ubTailTailFactor;
    int64_t realCoreNum;
    int64_t inputDims;
    int64_t blkIndex;
    int64_t blkFactor;
    int64_t blkTailFactor;
    int64_t xDtypeSize;
    int64_t tilingKey;
    int64_t nddmaTotalNum;
    int64_t ubInLoopSteps;
    int64_t ubOutLoopSteps;
    uint32_t isShapeExceedUint32;
    uint32_t isEmptyTensor;
    int8_t isBeginConst;
    StridedSliceMoveAlignParams2 moveAlignParams;
    int64_t outputShape[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t begin[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t strides[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t rowsOffsetSteps[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t inputSteps[MAX_AXIS_NUM_FOR_STRIDESLICE];
    int64_t nddmaLoopSize[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopSrcStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t outputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM];
    int64_t inputShapeProd[MAX_SIMT_UB_SPLIT_AXIS_NUM];
};

struct SliceTilingData {
    StridedSliceTilingData2 stridedSliceTilingData;
};

struct SliceMoveAlignTilingData {
    SliceBaseTilingData sliceBaseTilingData;
    StridedSliceMoveAlignParams2 moveAlignParams;
    int64_t ubOutLoopSteps;
};

struct SliceNDDMATilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t ubOutLoopSteps;
    int64_t nddmaTotalNum;
    int64_t nddmaLoopSize[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopSrcStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
};

struct SliceNDDMALastDimTilingData {
    SliceBaseTilingData sliceBaseTilingData;

    int64_t nddmaLoopSrcStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
    int64_t nddmaLoopDstStride[MAX_NDDMA_UB_SPLIT_AXIS_NUM];
};

struct SliceTwoDimSmallSapeTilingData {
    int8_t isBeginConst;
    int16_t realCoreNum;
    int16_t mainCoreNum;
    uint32_t ubSize;
    uint32_t blockLen;
    uint32_t blkFactor;
    uint64_t lastOneInputDim;
    uint64_t lastOneOutputDim;
    uint64_t lastOneDimOffset;
};

#endif // SLICE_TILINGDATA__H_
