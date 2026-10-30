/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Zhi<@hitLeechi>
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
 * \file reduce_prod_v2.h
 * \brief
 */
#ifndef _REDUCE_PROD_V2_H
#define _REDUCE_PROD_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "reduce_prod_v2_tiling_data.h"
#include "reduce_prod_v2_tiling_key.h"

namespace NsReduceProdV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SLOT_STRIDE = 64 / sizeof(float);
template <typename T>
class ReduceProdV2 {
public:
    __aicore__ inline ReduceProdV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const ReduceProdV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void ReduceProdV2Axes0();
    __aicore__ inline void ReduceProdV2Axes1();
    __aicore__ inline void ReduceProdV2Axes2();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueInput;
    // AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOutput;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::GlobalTensor<float> workGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFloat;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBase;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowSum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> colSum;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuffer;

    uint32_t blockIdx;
    uint32_t blockNum;
    uint32_t GlobalOffset;

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
__aicore__ inline void ReduceProdV2<T>::Init(
    GM_ADDR x, GM_ADDR z, GM_ADDR workspace, const ReduceProdV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreIdx = AscendC::GetBlockIdx();
    this->blockIdx = coreIdx;
    this->blockNum = AscendC::GetBlockNum();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * coreIdx;
    this->tileDataNum = tilingData->tileDataNum;
    this->axes = tilingData->axes;
    this->rows = tilingData->rows;
    this->cols = tilingData->cols;
    this->keyType = tilingData->dataTypeId;

    if (coreIdx < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -=
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) * (coreIdx - tilingData->tailBlockNum);
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
    if (axes == 0)
        outputSize = this->cols;
    else if (axes == 1)
        outputSize = this->rows;
    else
        outputSize = 1;
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
        AscendC::InitGlobalMemory(workGm, workGmSize, (float)0.0f);
    }
    pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileDataNum * sizeof(T));
    uint32_t outputSizeAlign32 = (outputSize * sizeof(T) + 31) / 32 * 32;

    pipe.InitBuffer(tmpFloat, tileDataNum * sizeof(float));
    pipe.InitBuffer(tmpBase, 64 * sizeof(float));
    pipe.InitBuffer(tmpBuffer, workGmSize * sizeof(float));
    if (axes == 0) {
        pipe.InitBuffer(colSum, this->cols * sizeof(float));
    } else if (axes == 1) {
        pipe.InitBuffer(rowSum, this->rows * sizeof(float));
    }
    this->GlobalOffset = globalBufferIndex;
}

template <typename T>
__aicore__ inline void ReduceProdV2<T>::CopyIn(int32_t progress)
{
    uint32_t curTileLen = (progress == tileNum - 1) ? tailDataNum : tileDataNum;
    AscendC::LocalTensor<T> xLocal = inQueueInput.AllocTensor<T>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], curTileLen);
    inQueueInput.EnQue(xLocal);
}
template <typename T>
__aicore__ inline void ReduceProdV2<T>::Process()
{
    if (this->axes == 0) {
        ReduceProdV2Axes0();
    } else if (this->axes == 1) {
        ReduceProdV2Axes1();
    } else {
        ReduceProdV2Axes2();
    }
}

template <typename T>
__aicore__ inline void ReduceProdV2<T>::ReduceProdV2Axes0()
{
    const uint32_t colNum = this->cols;
    const uint32_t rowNum = this->rows;
    const uint32_t totalElements = rowNum * colNum;
    const uint32_t tileNum = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->GlobalOffset;

    AscendC::LocalTensor<float> localSum = colSum.Get<float>();
    for (uint32_t c = 0; c < colNum; ++c) {
        localSum.SetValue(c, 0.0f);
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
        AscendC::Ln(tileFloat, tileFloat, curTileLen);
        for (uint32_t i = 0; i < curTileLen; ++i) {
            uint32_t globalIdx = globalOffset + t * tileLen + i;
            if (globalIdx >= totalElements) {
                continue;
            }
            uint32_t rowIdx = globalIdx / colNum;
            uint32_t colIdx = globalIdx % colNum;

            if (colIdx < colNum) {
                float prev = localSum.GetValue(colIdx);
                float curv = tileFloat.GetValue(i);
                localSum.SetValue(colIdx, curv + prev);
            }
        }

        inQueueInput.FreeTensor(tileLocal);
    }

    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    uint32_t totalWorkSize = colNum * SLOT_STRIDE;
    AscendC::Duplicate(tmpBuf, 0.0f, totalWorkSize);
    for (uint32_t c = 0; c < colNum; ++c) {
        tmpBuf.SetValue(c * SLOT_STRIDE, localSum.GetValue(c));
    }
    AscendC::SetAtomicAdd<float>();
    AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
    AscendC::SetAtomicNone();
    AscendC::SyncAll();
    Barrier();

    if (this->blockIdx == 0) {
        AscendC::DataCopy(tmpBuf, workGm, totalWorkSize);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::Exp(tmpBuf, tmpBuf, totalWorkSize);
        for (uint32_t c = 0; c < colNum; ++c) {
            float val = tmpBuf.GetValue(c * SLOT_STRIDE);
            zGm.SetValue(c, static_cast<T>(val));
        }
    }
}

template <typename T>
__aicore__ inline void ReduceProdV2<T>::ReduceProdV2Axes1()
{
    const uint32_t colNum = this->cols;
    const uint32_t rowNum = this->rows;
    const uint32_t totalElements = rowNum * colNum;
    const uint32_t tileNum = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->GlobalOffset;

    AscendC::LocalTensor<float> localRowSum = rowSum.Get<float>();
    for (uint32_t r = 0; r < rowNum; ++r) {
        localRowSum.SetValue(r, 0.0f);
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
        AscendC::Ln(tileFloat, tileFloat, curTileLen);
        for (uint32_t i = 0; i < curTileLen; ++i) {
            uint32_t globalIdx = globalOffset + t * tileLen + i;
            if (globalIdx >= totalElements) {
                continue;
            }
            uint32_t rowIdx = globalIdx / colNum;
            uint32_t colIdx = globalIdx % colNum;

            if (rowIdx < rowNum) {
                float prev = localRowSum.GetValue(rowIdx);
                float curv = tileFloat.GetValue(i);
                localRowSum.SetValue(rowIdx, curv + prev);
            }
        }

        inQueueInput.FreeTensor(tileLocal);
    }

    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    uint32_t totalWorkSize = rowNum * SLOT_STRIDE;
    AscendC::Duplicate(tmpBuf, 0.0f, totalWorkSize);
    for (uint32_t r = 0; r < rowNum; ++r) {
        tmpBuf.SetValue(r * SLOT_STRIDE, localRowSum.GetValue(r));
    }
    AscendC::SetAtomicAdd<float>();
    AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
    AscendC::SetAtomicNone();
    AscendC::SyncAll();
    Barrier();

    if (this->blockIdx == 0) {
        AscendC::DataCopy(tmpBuf, workGm, totalWorkSize);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::Exp(tmpBuf, tmpBuf, totalWorkSize);
        for (uint32_t r = 0; r < rowNum; ++r) {
            float val = tmpBuf.GetValue(r * SLOT_STRIDE);
            zGm.SetValue(r, static_cast<T>(val));
        }
    }
}
template <typename T>
__aicore__ inline void ReduceProdV2<T>::ReduceProdV2Axes2()
{
    const uint32_t loopCount = this->tileNum;
    const uint32_t tileLen = this->tileDataNum;
    const uint32_t blockLen = this->coreDataNum;
    const uint32_t lastTileLen = this->tailDataNum;
    const uint32_t globalOffset = this->GlobalOffset;

    AscendC::LocalTensor<float> localSum = tmpBase.Get<float>();
    float initVal = 0.0f;
    localSum.SetValue(0, initVal);
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
        AscendC::Ln(tileFloat, tileFloat, curTileLen);
        float tileSum = 0.0f;
        for (uint32_t i = 0; i < curTileLen; ++i) {
            float curv = tileFloat.GetValue(i);
            tileSum += curv;
        }
        localSum.SetValue(0, tileSum + localSum.GetValue(0));
        inQueueInput.FreeTensor(tileLocal);
    }
    AscendC::LocalTensor<float> tmpBuf = tmpBuffer.Get<float>();
    uint32_t totalWorkSize = SLOT_STRIDE;
    AscendC::Duplicate(tmpBuf, 0.0f, totalWorkSize);
    tmpBuf.SetValue(0, localSum.GetValue(0));
    AscendC::SetAtomicAdd<float>();
    AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
    AscendC::SetAtomicNone();
    AscendC::DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        workGm[0]);
    AscendC::SyncAll();
    Barrier();

    if (this->blockIdx == 0) {
        AscendC::DataCopy(tmpBuf, workGm, totalWorkSize);
        // MTE2_V  GM->UB --vector
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::Exp(tmpBuf, tmpBuf, totalWorkSize);
        // V_MTE3 vector --UB->GM
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::DataCopy(workGm, tmpBuf, totalWorkSize);
        float globalSum = workGm.GetValue(0);
        zGm.SetValue(0, static_cast<T>(globalSum));
    }
}
} // namespace NsReduceProdV2
#endif // ReduceProdV2_H