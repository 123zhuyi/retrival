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
 * \file split_v_tiling_data.h
 * \brief tiling data struct
 */
#ifndef OP_KERNEL_SPLIT_V_TILING_DATA_H_
#define OP_KERNEL_SPLIT_V_TILING_DATA_H_
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

const int maxSplitNum = 61;
const uint32_t splitVSameLenFullRowDma = 0xFFFFFFFFU;
const uint32_t splitVSameLenInnerCopyFullRowPack = 0;
const uint32_t splitVSameLenInnerCopyMidTilePack = 1;
const uint32_t splitVSameLenInnerCopyInnerTilePack = 2;
const uint32_t splitVSameLenInnerCopySegmentInnerPack = 3;
const uint32_t splitVSameLenInnerCopySplitChunkPack = 4;
const uint32_t splitVUnevenInnerAlignedMidTilePack = 0;
const uint32_t splitVUnevenInnerSegmentPack = 1;
const uint32_t splitVUnevenInnerSplitChunkPack = 2;

#define SPLIT_V_PURE_COPY_SCHEDULE_FIELDS \
    uint32_t mode;                        \
    uint32_t splitPitch;                  \
    uint32_t realCoreNum;                 \
    uint32_t formerCoreRows;              \
    uint32_t tailCoreRows;                \
    uint32_t formerNum;                   \
    uint32_t outerTile;                   \
    uint32_t colTileLength;               \
    uint32_t colTilePitch

struct SplitVTilingData {
    uint64_t totalLength;
    uint64_t outerLength;
    uint64_t midLength;
    uint64_t innerLength;

    int64_t splitDim;
    int64_t splitNum;
    int64_t sizeSplits[maxSplitNum];

    uint64_t formerLoop;
    uint64_t tailLoop;
    uint64_t formerNum;

    uint32_t innerTileLength;
    uint32_t innerTileNum;
    uint32_t innerLastTileLength;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t splitTileLength;
};

struct SplitVTilingDataPureCopy {
    uint64_t totalLength;
    uint64_t formerLength;
    uint64_t tailLength;
    uint32_t tileLength;
    uint32_t formerTileNum;
    uint32_t tailTileNum;
    uint32_t formerLastTileLength;
    uint32_t tailLastTileLength;
};

struct SplitVTilingDataOneRowPureCopy {
    uint64_t totalLength;
    uint32_t splitNum;
    uint32_t sizeSplits[maxSplitNum];

    uint64_t totalTaskNum;
    uint64_t formerTaskNum;
    uint64_t tailTaskNum;
    uint64_t formerNum;
    uint32_t chunkLength;
};

struct SplitVTilingDataSameLen {
    uint64_t totalLength;
    uint64_t outerLength;
    uint64_t midLength;
    uint64_t innerLength;

    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t splitNum;

    uint64_t formerLoop;
    uint64_t tailLoop;
    uint64_t formerNum;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t splitTileLength;
};

struct SplitVTilingDataSameLenCompact {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t splitNum;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t formerNum;

    uint32_t rowTransLen;
    uint32_t splitTransLen;
    uint32_t chunkSplitNum;
    uint32_t colChunkNum;
};

struct SplitVTilingDataSameLenCompact32Bit {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitSize;
    uint32_t splitNum;
    uint32_t viewFactor;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t formerNum;

    uint32_t viewRowLength;
    uint32_t viewSplitSize;
    uint32_t viewRowPitch;
    uint32_t viewSplitPitch;
};

struct SplitVTilingDataSameLenPureCopy8Bit {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t splitNum;

    SPLIT_V_PURE_COPY_SCHEDULE_FIELDS;
};

struct SplitVTilingDataSameLenInnerCopy {
    uint64_t totalLength;
    uint64_t outerLength;
    uint64_t midLength;
    uint64_t innerLength;

    uint32_t splitSize;
    uint32_t tailSplitSize;
    uint32_t splitNum;
    uint32_t mode;

    uint32_t outerTile;
    uint32_t outerTileNum;
    uint32_t outerTail;

    uint32_t midTile;
    uint32_t midTileNum;
    uint32_t midTail;

    uint32_t innerTile;
    uint32_t innerTileNum;
    uint32_t innerTail;

    uint32_t chunkElems;
    uint32_t chunkElemsAligned;
    uint32_t chunkNumMax;

    uint64_t totalTaskNum;
    uint64_t formerTaskNum;
    uint64_t tailTaskNum;
    uint64_t formerNum;
};

struct SplitVTilingDataUnevenInnerAlignedMid {
    uint64_t totalLength;
    uint64_t outerLength;
    uint64_t midLength;
    uint64_t innerLength;

    uint32_t splitNum;
    uint32_t mode;
    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitOffsets[maxSplitNum];

    uint32_t outerTile;
    uint32_t outerTileNum;
    uint32_t outerTail;

    uint32_t midTile;
    uint32_t midTileNum;
    uint32_t midTail;

    uint32_t chunkElems;
    uint32_t chunkElemsAligned;
    uint32_t chunkNumMax;

    uint64_t totalTaskNum;
    uint64_t formerTaskNum;
    uint64_t tailTaskNum;
    uint64_t formerNum;
};

struct SplitVTilingDataUnevenCompact {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitNum;
    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitStarts[maxSplitNum];
    uint32_t maxSplitSize;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t formerNum;

    uint32_t mode;
    uint32_t rowTransLen;
    uint32_t splitTransLen;
    uint32_t virtualSplitSize;
    uint32_t virtualSplitNum;
    uint32_t colChunkSize;
    uint32_t colChunkNum;
};

struct SplitVTilingDataUnevenCompact32Bit {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitNum;
    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitStarts[maxSplitNum];
    uint32_t maxSplitSize;
    uint32_t viewFactor;

    uint32_t outerTile;
    uint32_t outerTail;
    uint32_t outerTileNum;
    uint32_t formerOuterTileNum;
    uint32_t tailOuterTileNum;
    uint32_t formerNum;

    uint32_t viewRowLength;
    uint32_t maxViewSplitSize;
    uint32_t viewRowPitch;
    uint32_t maxViewSplitPitch;
};

struct SplitVTilingDataUnevenPureCopy16Bit {
    uint64_t totalLength;
    uint64_t outerLength;
    uint32_t rowLength;
    uint32_t splitNum;
    uint32_t sizeSplits[maxSplitNum];
    uint32_t splitStarts[maxSplitNum];
    uint32_t maxSplitSize;

    SPLIT_V_PURE_COPY_SCHEDULE_FIELDS;
};
#endif // OP_KERNEL_SPLIT_V_TILING_DATA_H_
