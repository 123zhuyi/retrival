/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua<@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file broadcast_v2.h
 * \brief
 */
#ifndef BROADCASTV2_H
#define BROADCASTV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "broadcast_v2_tiling_data.h"
#include "broadcast_v2_tiling_key.h"

namespace NsBroadcastV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class BroadcastV2 {
public:
    __aicore__ inline BroadcastV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const BroadcastV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    AscendC::TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    AscendC::TBuf<QuePosition::VECCALC> tmpBuffer;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t dim;
    uint32_t tmpSize;
    uint32_t axis;
    uint32_t num;
    uint32_t bLength;
    uint32_t isReuseSource;
    uint32_t sLength;
};

template <typename T>
__aicore__ inline void BroadcastV2<T>::Init(GM_ADDR x, GM_ADDR y, const BroadcastV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreIdx = AscendC::GetBlockIdx();
    this->dim = tilingData->dim;
    this->tmpSize = tilingData->tmpSize;
    this->axis = tilingData->axis;
    this->num = tilingData->num;
    this->bLength = tilingData->bLength;
    this->isReuseSource = tilingData->isReuseSource;
    this->sLength = (this->dim == 1) ? 1u : static_cast<uint32_t>(tilingData->sLength);

    uint32_t inGlobalBufferIndex;
    uint32_t outGlobalBufferIndex;

    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
        inGlobalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
        outGlobalBufferIndex = tilingData->bigCoreDataNum * this->num * coreIdx;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;

        inGlobalBufferIndex = tilingData->bigCoreDataNum * tilingData->tailBlockNum +
                              tilingData->smallCoreDataNum * (coreIdx - tilingData->tailBlockNum);

        uint32_t bigCoresOutputSize = tilingData->bigCoreDataNum * this->num * tilingData->tailBlockNum;
        uint32_t precedingSmallCoresOutputSize =
            tilingData->smallCoreDataNum * this->num * (coreIdx - tilingData->tailBlockNum);
        outGlobalBufferIndex = bigCoresOutputSize + precedingSmallCoresOutputSize;
    }

    this->tileDataNum = tilingData->tileDataNum;
    uint32_t outCoreDataNum = this->coreDataNum * this->num;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + inGlobalBufferIndex, this->coreDataNum);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + outGlobalBufferIndex, outCoreDataNum);

    if (this->tileDataNum > 0) {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->tileDataNum * this->num * sizeof(T));
    }

    if (this->tmpSize > 0) {
        pipe.InitBuffer(tmpBuffer, this->tmpSize);
    }
}

template <typename T>
__aicore__ inline void BroadcastV2<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, inputGMX[progress * this->tileDataNum], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void BroadcastV2<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    uint32_t outOffset = progress * this->tileDataNum * this->num;
    uint32_t outLength = this->processDataNum * this->num;
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(outLength * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(outputGMY[outOffset], yLocal, copyParams);
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void BroadcastV2<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    uint32_t sLen = this->sLength;
    uint32_t rowsThisTile = (this->dim == 2 && sLen > 0) ? (this->processDataNum / sLen) : 0;
    bool hasTmp = (this->tmpSize > 0);
    uint32_t alignUnitElems = 32u / sizeof(T);
    if (alignUnitElems == 0)
        alignUnitElems = 1;
    bool useHw = false;
    if (this->dim == 1) {
        uint32_t outLen = this->processDataNum * this->num;
        useHw = ((this->processDataNum % alignUnitElems) == 0) && ((outLen % alignUnitElems) == 0);
    } else {
        if (this->axis == 0) {
            uint32_t outRows = rowsThisTile * this->num;
            useHw = ((sLen % alignUnitElems) == 0) && ((rowsThisTile % alignUnitElems) == 0) &&
                    ((outRows % alignUnitElems) == 0);
        } else {
            uint32_t outCols = sLen * this->num;
            useHw = ((this->bLength % alignUnitElems) == 0) && ((rowsThisTile % alignUnitElems) == 0) &&
                    ((outCols % alignUnitElems) == 0);
        }
    }

    AscendC::LocalTensor<uint8_t> tmpLocal;
    if (useHw && hasTmp) {
        tmpLocal = tmpBuffer.Get<uint8_t>();
    }

    if (useHw) {
        if (this->dim == 1) {
            const uint32_t srcShape[] = {this->processDataNum};
            const uint32_t dstShape[] = {this->processDataNum * this->num};
            if (hasTmp) {
                AscendC::BroadCast<T, 1, 0>(yLocal, xLocal, dstShape, srcShape, tmpLocal);
            } else {
                AscendC::BroadCast<T, 1, 0>(yLocal, xLocal, dstShape, srcShape);
            }
        } else {
            const uint32_t srcShape[] = {rowsThisTile, sLen};
            if (this->axis == 0) {
                const uint32_t dstShape[] = {rowsThisTile * this->num, sLen};
                if (hasTmp) {
                    AscendC::BroadCast<T, 2, 0>(yLocal, xLocal, dstShape, srcShape, tmpLocal);
                } else {
                    AscendC::BroadCast<T, 2, 0>(yLocal, xLocal, dstShape, srcShape);
                }
            } else { // axis == 1
                const uint32_t dstShape[] = {rowsThisTile, sLen * this->num};
                if (hasTmp) {
                    AscendC::BroadCast<T, 2, 1>(yLocal, xLocal, dstShape, srcShape, tmpLocal);
                } else {
                    AscendC::BroadCast<T, 2, 1>(yLocal, xLocal, dstShape, srcShape);
                }
            }
        }
    } else {
        if (this->dim == 1) {
            AscendC::LocalTensor<T> srcVec = xLocal[0];
            for (uint32_t k = 0; k < this->num; ++k) {
                AscendC::LocalTensor<T> dstVec = yLocal[k * this->processDataNum];
                AscendC::DataCopy(dstVec, srcVec, this->processDataNum);
            }
        } else {
            if (this->axis == 0) {
                for (uint32_t r = 0; r < rowsThisTile; ++r) {
                    AscendC::LocalTensor<T> inRow = xLocal[r * sLen];
                    for (uint32_t k = 0; k < this->num; ++k) {
                        uint32_t outRowIdx = k * rowsThisTile + r;
                        AscendC::LocalTensor<T> outRow = yLocal[outRowIdx * sLen];
                        AscendC::DataCopy(outRow, inRow, sLen);
                    }
                }
            } else {
                uint32_t outRowStride = sLen * this->num;
                for (uint32_t r = 0; r < rowsThisTile; ++r) {
                    AscendC::LocalTensor<T> inRow = xLocal[r * sLen];
                    uint32_t outRowBaseElem = r * outRowStride;
                    for (uint32_t k = 0; k < this->num; ++k) {
                        AscendC::LocalTensor<T> outSeg = yLocal[outRowBaseElem + k * sLen];
                        AscendC::DataCopy(outSeg, inRow, sLen);
                    }
                }
            }
        }
    }
    if (useHw && hasTmp) {
        tmpBuffer.FreeTensor(tmpLocal);
    }
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void BroadcastV2<T>::Process()
{
    int32_t loopCount = this->tileNum;
    if (loopCount == 0) {
        return;
    }
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        if (this->processDataNum == 0) {
            continue;
        }
        if (this->dim == 2) {
            ASSERT((this->processDataNum % this->sLength) == 0 && "processDataNum not aligned to sLength");
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsBroadcastV2
#endif // BROADCASTV2_H