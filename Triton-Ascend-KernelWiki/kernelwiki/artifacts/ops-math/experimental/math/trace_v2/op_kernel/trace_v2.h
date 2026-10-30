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
 * \file trace_v2.h
 * \brief
 */
#ifndef TRACE_V2_H
#define TRACE_V2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "trace_v2_tiling_data.h"
#include "trace_v2_tiling_key.h"

namespace NsTraceV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TraceV2 {
public:
    __aicore__ inline TraceV2(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TraceV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();

private:
    TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    TBuf<QuePosition::VECCALC> tmpBuf0;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf2;

    TraceV2TilingData tiling;
    uint64_t rowLength;
    uint64_t columnLength;
    uint64_t diagLen;
    uint64_t fullBlockLength;
    uint64_t tailBlockLength;
    uint64_t fullBlockNum;
    uint64_t tailBlockNum;
    uint64_t typeSize;
    uint64_t matrixOrder;
    uint64_t blockIdx;
};

template <typename T>
__aicore__ inline void TraceV2<T>::Init(GM_ADDR x, GM_ADDR y, const TraceV2TilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->tiling = *tilingData;
    this->blockIdx = AscendC::GetBlockIdx();
    matrixOrder =
        (this->blockIdx >= tilingData->fullBlockNum) ? tilingData->tailBlockLength : tilingData->fullBlockLength;
    if (matrixOrder == 0) {
        return;
    }
    xGm.SetGlobalBuffer((__gm__ T*)x, tilingData->rowLength * tilingData->columnLength);
    yGm.SetGlobalBuffer((__gm__ T*)y, 1);
    uint32_t bufSize = matrixOrder * sizeof(T);
    uint32_t alignSize = 32;
    uint32_t allocSize = (bufSize + alignSize - 1) / alignSize * alignSize;
    pipe.InitBuffer(inQueueX, BUFFER_NUM, allocSize);
    pipe.InitBuffer(tmpQueue, BUFFER_NUM, allocSize);
    pipe.InitBuffer(outQueueY, BUFFER_NUM, allocSize);
    if constexpr (AscendC::Std::is_same<T, int32_t>::value || AscendC::Std::is_same<T, int16_t>::value) {
        pipe.InitBuffer(tmpBuf0, allocSize);
        pipe.InitBuffer(tmpBuf1, allocSize);
        pipe.InitBuffer(tmpBuf2, sizeof(float));
    }
}

template <typename T>
__aicore__ inline void TraceV2<T>::CopyIn()
{
    AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    uint64_t diagStart = 0;
    if (AscendC::GetBlockIdx() < tiling.fullBlockNum) {
        diagStart = (uint64_t)AscendC::GetBlockIdx() * tiling.fullBlockLength;
    } else {
        diagStart = tiling.fullBlockNum * tiling.fullBlockLength +
                    ((uint64_t)AscendC::GetBlockIdx() - tiling.fullBlockNum) * tiling.tailBlockLength;
    }
    uint64_t globalOffset = diagStart * (tiling.columnLength + 1);
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(matrixOrder), static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>((tiling.columnLength) * sizeof(T)), 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, xGm[globalOffset], copyParams, padParams);
    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void TraceV2<T>::CopyOut()
{
    AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::SetAtomicAdd<T>();
    AscendC::DataCopy(yGm, yLocal, 32);
    AscendC::SetAtomicNone();
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TraceV2<T>::Compute()
{
    AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    AscendC::LocalTensor<T> workLocal = tmpQueue.AllocTensor<T>();
    AscendC::ReduceSum<T>(yLocal, xLocal, workLocal, matrixOrder);
    outQueueY.EnQue<T>(yLocal);
    inQueueX.FreeTensor(xLocal);
    tmpQueue.FreeTensor(workLocal);
}

template <typename T>
__aicore__ inline void TraceV2<T>::Process()
{
    if (matrixOrder > 0) {
        CopyIn();
        Compute();
        CopyOut();
    }
}

} // namespace NsTraceV2
#endif // TraceV2_H