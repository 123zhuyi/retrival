/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file reduce_min_v2.h
 * \brief
 */
#ifndef _REDUCE_MIN_V2_H
#define _REDUCE_MIN_V2_H

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_min_v2_tiling_data.h"
#include "reduce_min_v2_tiling_key.h"

namespace NsReduceMinV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DATA_CACHE_CLEAN_NEED = 64;
constexpr int32_t SLOT_STRIDE = DATA_CACHE_CLEAN_NEED / sizeof(float);
template <typename T>
class ReduceMinV2 {
public:
    __aicore__ inline ReduceMinV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const ReduceMinV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void ReduceMinV2Axes0();
    __aicore__ inline void ReduceMinV2Axes1();
    __aicore__ inline void ReduceMinV2AxesAll();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::GlobalTensor<float> workGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFloat;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBase;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMin;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colMin;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;

    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t globalOffset;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;

    uint32_t keyType;
    uint32_t axes;
    uint32_t rows;
    uint32_t cols;
    uint32_t keepdims;
};

template <typename T>
__aicore__ inline void ReduceMinV2<T>::Init(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const ReduceMinV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->blockIdx = AscendC::GetBlockIdx();
    ;
    this->blockNum = AscendC::GetBlockNum();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * this->blockIdx;
    this->tileDataNum = tilingData->tileDataNum;
    this->axes = tilingData->axes;
    this->rows = tilingData->rows;
    this->cols = tilingData->cols;
    this->keyType = tilingData->dataTypeId;

    if (this->blockIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (this->blockIdx - tilingData->tailBlockNum);
    }

    uint32_t totalElements = this->rows * this->cols;
    if (globalBufferIndex >= totalElements) {
        this->coreDataNum = 0;
        xGm.SetGlobalBuffer((__gm__ T*)x, 0);
    } else {
        uint32_t available = totalElements - globalBufferIndex;
        uint32_t bindLen = (available < this->coreDataNum) ? available : this->coreDataNum;
        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, bindLen);
        this->coreDataNum = bindLen;
    }
    uint32_t outputSize = 0;
    if (axes == 0) {
        outputSize = this->cols;
    } else if (axes == 1) {
        outputSize = this->rows;
    } else {
        outputSize = 1;
    }
    zGm.SetGlobalBuffer((__gm__ T*)z, outputSize); // 所有核共享输出 GM
    uint32_t workGmSize = 0;
    if (axes == 0) {
        workGmSize = this->cols * SLOT_STRIDE;
    } else if (axes == 1) {
        workGmSize = this->rows * SLOT_STRIDE;
    } else {
        workGmSize = 1 * SLOT_STRIDE;
    }
    workGm.SetGlobalBuffer((__gm__ float*)workspace, workGmSize);
    if (AscendC::GetBlockIdx() == 0) {
        AscendC::InitGlobalMemory(workGm, workGmSize, std::numeric_limits<float>::infinity());
    }
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(tmpFloat, tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBase, 64 * sizeof(float));
    pipe.InitBuffer(tmpBuffer, workGmSize * sizeof(float));
    if (axes == 0) {
        pipe.InitBuffer(colMin, this->cols * sizeof(float));
    } else if (axes == 1) {
        pipe.InitBuffer(rowMin, this->rows * sizeof(float));
    }
    this->globalOffset = globalBufferIndex;
}

template <typename T>
__aicore__ inline void ReduceMinV2<T>::CopyIn(int32_t progress)
{
    uint32_t curTileLen = (progress == tileNum - 1) ? tailDataNum : tileDataNum;
    AscendC::LocalTensor<T> xLocal = inQueueInput.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], curTileLen);
    inQueueInput.EnQue(xLocal);
}
template <typename T>
__aicore__ inline void ReduceMinV2<T>::Process()
{
    if (this->axes == 0) {
        ReduceMinV2Axes0();
    } else if (this->axes == 1) {
        ReduceMinV2Axes1();
    } else {
        ReduceMinV2AxesAll();
    }
}

template <typename T>
__aicore__ inline void ReduceMinV2<T>::ReduceMinV2Axes0()
{
    AscendC::SyncAll();
    const uint32_t colNum = this->cols;
    const uint32_t rowNum = this->rows;
    const uint32_t totalElements = rowNum * colNum;

    const uint32_t tileNum = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->globalOffset;

    AscendC::LocalTensor<float> localMin = colMin.Get<float>();
    for (uint32_t c = 0; c < colNum; ++c) {
        localMin.SetValue(c, std::numeric_limits<float>::infinity());
    }

    for (uint32_t t = 0; t < tileNum; ++t) {
        uint32_t curTileLen = (t == tileNum - 1) ? lastTileLen : tileLen;

        CopyIn(t);
        AscendC::LocalTensor<T> tileLocal = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<float> tileFloat = tmpFloat.Get<float>();

        if (keyType == 1) {
            AscendC::Cast(tileFloat, tileLocal, AscendC::RoundMode::CAST_NONE, curTileLen);
        } else {
            for (uint32_t i = 0; i < curTileLen; ++i) {
                tileFloat.SetValue(i, tileLocal.GetValue(i));
            }
        }

        for (uint32_t i = 0; i < curTileLen; ++i) {
            uint32_t globalIdx = globalOffset + t * tileLen + i;
            if (globalIdx >= totalElements) {
                continue;
            }
            uint32_t rowIdx = globalIdx / colNum;
            uint32_t colIdx = globalIdx % colNum;

            if (colIdx < colNum) {
                float prev = localMin.GetValue(colIdx);
                float curv = tileFloat.GetValue(i);
                if (prev > curv)
                    localMin.SetValue(colIdx, curv);
            }
        }

        inQueueInput.FreeTensor(tileLocal);
    }

    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    uint32_t totalWorkSize = colNum * SLOT_STRIDE;
    AscendC::Duplicate(tmpBuf, std::numeric_limits<float>::infinity(), totalWorkSize);
    for (uint32_t c = 0; c < colNum; ++c) {
        tmpBuf.SetValue(c * SLOT_STRIDE, localMin.GetValue(c));
    }
    AscendC::SetAtomicMin<float>();
    AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
    AscendC::SetAtomicNone();
    AscendC::SyncAll();

    if (this->blockIdx == 0) {
        AscendC::DataCopy(tmpBuf, workGm, totalWorkSize);
        for (uint32_t c = 0; c < colNum; ++c) {
            float val = tmpBuf.GetValue(c * SLOT_STRIDE);
            zGm.SetValue(c, static_cast<T>(val));
        }
    }
}

template <typename T>
__aicore__ inline void ReduceMinV2<T>::ReduceMinV2Axes1()
{
    AscendC::SyncAll();
    const uint32_t colNum = this->cols;
    const uint32_t rowNum = this->rows;
    const uint32_t totalElements = rowNum * colNum;

    const uint32_t tileNum = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->globalOffset;

    AscendC::LocalTensor<float> localRowMin = rowMin.Get<float>();
    for (uint32_t r = 0; r < rowNum; ++r) {
        localRowMin.SetValue(r, std::numeric_limits<float>::infinity());
    }
    for (uint32_t t = 0; t < tileNum; ++t) {
        uint32_t curTileLen = (t == tileNum - 1) ? lastTileLen : tileLen;

        CopyIn(t);
        AscendC::LocalTensor<T> tileLocal = inQueueInput.DeQue<T>();
        AscendC::LocalTensor<float> tileFloat = tmpFloat.Get<float>();
        if (keyType == 1) {
            AscendC::Cast(tileFloat, tileLocal, AscendC::RoundMode::CAST_NONE, curTileLen);
        } else {
            for (uint32_t i = 0; i < curTileLen; ++i) {
                tileFloat.SetValue(i, tileLocal.GetValue(i));
            }
        }
        for (uint32_t i = 0; i < curTileLen; ++i) {
            uint32_t globalIdx = globalOffset + t * tileLen + i;
            if (globalIdx >= totalElements) {
                continue;
            }
            uint32_t rowIdx = globalIdx / colNum;
            uint32_t colIdx = globalIdx % colNum;

            if (rowIdx < rowNum) {
                float prev = localRowMin.GetValue(rowIdx);
                float curv = tileFloat.GetValue(i);
                if (prev > curv)
                    localRowMin.SetValue(rowIdx, curv);
            }
        }

        inQueueInput.FreeTensor(tileLocal);
    }

    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    uint32_t totalWorkSize = rowNum * SLOT_STRIDE;
    AscendC::Duplicate(tmpBuf, std::numeric_limits<float>::infinity(), totalWorkSize);
    for (uint32_t r = 0; r < rowNum; ++r) {
        tmpBuf.SetValue(r * SLOT_STRIDE, localRowMin.GetValue(r));
    }
    AscendC::SetAtomicMin<float>();
    AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
    AscendC::SetAtomicNone();
    AscendC::SyncAll();

    if (this->blockIdx == 0) {
        AscendC::DataCopy(tmpBuf, workGm, totalWorkSize);
        for (uint32_t r = 0; r < rowNum; ++r) {
            float val = tmpBuf.GetValue(r * SLOT_STRIDE);
            zGm.SetValue(r, static_cast<T>(val));
        }
    }
}
template <typename T>
__aicore__ inline void ReduceMinV2<T>::ReduceMinV2AxesAll()
{
    const uint32_t loopCount = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t blockLen = this->coreDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->globalOffset;

    AscendC::LocalTensor<float> localMin = tmpBase.Get<float>();
    float initVal = std::numeric_limits<float>::infinity();
    localMin.SetValue(0, initVal);
    for (uint32_t t = 0; t < loopCount; ++t) {
        uint32_t curTileLen = (t == loopCount - 1) ? lastTileLen : tileLen;

        CopyIn(t);
        AscendC::LocalTensor<T> tileLocal = inQueueInput.DeQue<T>();

        AscendC::LocalTensor<float> tileFloat = tmpFloat.Get<float>();
        if (keyType == 1) {
            AscendC::Cast(tileFloat, tileLocal, AscendC::RoundMode::CAST_NONE, curTileLen);
        } else {
            for (uint32_t i = 0; i < curTileLen; ++i) {
                tileFloat.SetValue(i, tileLocal.GetValue(i));
            }
        }
        float tileMin = std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < curTileLen; ++i) {
            float curv = tileFloat.GetValue(i);
            if (curv < tileMin)
                tileMin = curv;
        }
        if (localMin.GetValue(0) > tileMin)
            localMin.SetValue(0, tileMin);

        inQueueInput.FreeTensor(tileLocal);
    }
    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    AscendC::Duplicate(tmpBuf, std::numeric_limits<float>::infinity(), 8);
    tmpBuf.SetValue(0, localMin.GetValue(0));

    AscendC::SetAtomicMin<float>();
    AscendC::DataCopy(workGm[0], tmpBuf, 8);
    AscendC::SetAtomicNone();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        workGm[0]);
    AscendC::SyncAll();

    float globalMin = workGm.GetValue(0);
    if (this->blockIdx == 0) {
        zGm.SetValue(0, static_cast<T>(globalMin));
    }
}
} // namespace NsReduceMinV2
#endif // ReduceMinV2_H