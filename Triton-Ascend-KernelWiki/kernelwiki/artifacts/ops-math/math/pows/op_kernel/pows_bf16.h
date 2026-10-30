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
 * \file pows_bf16.h
 * \brief
 */
#ifndef Pows_BFLOAT16_ALIGN_H
#define Pows_BFLOAT16_ALIGN_H

#include "pows_base.h"

namespace Pows {
using namespace AscendC;

template <typename T>
class PowsBfp16 : public PowsBase<T>
{
public:
    __aicore__ inline PowsBfp16(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, const PowsTilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;

private:
    __aicore__ inline void CopyInX(const int64_t& index, const int64_t& dataLength);
    __aicore__ inline void ComputePows(const int64_t& dataLength);
    __aicore__ inline void CopyOut(const int64_t& index, const int64_t& dataLength);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX1;
    TQue<QuePosition::VECOUT, 1> outQueue;

    TBuf<QuePosition::VECCALC> tempBuf1;
    TBuf<QuePosition::VECCALC> tempBuf2;
};

template <typename T>
__aicore__ inline void PowsBfp16<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const PowsTilingData* tilingData)
{
    this->BaseInit(x, y, gelu, tilingData);
    *(reinterpret_cast<T*>(&(this->scalar)) + 1) = this->x2Gm.GetValue(0);

    pipe.InitBuffer(inQueueX1, bufferNum, this->m_tilingData.bufSize * sizeof(T));
    pipe.InitBuffer(outQueue, 1, this->m_tilingData.bufSize * sizeof(T));

    pipe.InitBuffer(tempBuf1, this->m_tilingData.bufSize * sizeof(float));
    pipe.InitBuffer(tempBuf2, this->m_tilingData.bufSize * sizeof(float));
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::Process()
{
    if (this->blockIdx >= this->m_tilingData.realCoreNum) {
        return;
    }

    if (this->isLastCore) { // process last core
        ProcessLastCore();
    } else {
        ProcessPerCore();
    }
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::ProcessPerCore()
{
    // process core
    for (int64_t idx = 0; idx < this->m_tilingData.mainCoreLoopNum; idx++) {
        CopyInX(idx, this->m_tilingData.dataLength);
        ComputePows(this->m_tilingData.dataLength);
        CopyOut(idx, this->m_tilingData.dataLength);
    }

    if (this->m_tilingData.mainCoreTailLength > 0) {
        CopyInX(this->m_tilingData.mainCoreLoopNum, this->m_tilingData.mainCoreTailLength);
        ComputePows(this->m_tilingData.mainCoreTailLength);
        CopyOut(this->m_tilingData.mainCoreLoopNum, this->m_tilingData.mainCoreTailLength);
    }
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::ProcessLastCore()
{
    for (int64_t idx = 0; idx < this->m_tilingData.tailCoreLoopNum; idx++) {
        CopyInX(idx, this->m_tilingData.dataLength);
        ComputePows(this->m_tilingData.dataLength);
        CopyOut(idx, this->m_tilingData.dataLength);
    }
    if (this->m_tilingData.tailCoreTailLength > 0) {
        CopyInX(this->m_tilingData.tailCoreLoopNum, this->m_tilingData.tailCoreTailLength);
        ComputePows(this->m_tilingData.tailCoreTailLength);
        CopyOut(this->m_tilingData.tailCoreLoopNum, this->m_tilingData.tailCoreTailLength);
    }
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::CopyInX(const int64_t& index, const int64_t& dataLength)
{
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    this->CopyInXBase(index, dataLength, ubX1);
    inQueueX1.EnQue(ubX1);
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::ComputePows(const int64_t& dataLength)
{
    LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
    LocalTensor<float> ubx_fp32 = tempBuf1.Get<float>();
    Cast(ubx_fp32, ubX1, RoundMode::CAST_NONE, dataLength);
    inQueueX1.FreeTensor(ubX1);

    // after cast to fp32 , input buffer release, to use as tmp buffer wihle do pows compute.
    LocalTensor<float> result_fp32 = tempBuf2.Get<float>();
    this->ComputePowsBase(ubx_fp32, result_fp32, dataLength);

    LocalTensor<T> out = outQueue.AllocTensor<T>();
    Cast(out, result_fp32, RoundMode::CAST_RINT, dataLength);
    outQueue.EnQue(out);
}

template <typename T>
__aicore__ inline void PowsBfp16<T>::CopyOut(const int64_t& index, const int64_t& dataLength)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    this->CopyOutBase(index, dataLength, outLocal);
    outQueue.FreeTensor(outLocal);
}
} // namespace Pows
#endif // Pows_BFLOAT16_ALIGN_H