/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_int.h
 * \brief
 */

#ifndef RANGE_INT_H
#define RANGE_INT_H

#include "range_base.h"
#include "op_kernel/math_util.h"

namespace Range {
using namespace AscendC;

template <typename T>
class RangeInt : public RangeBase<T> {
public:
    __aicore__ inline RangeInt(){};
    __aicore__ inline void Init(
        GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace, const RangeTilingDataInt* tilingData);
    __aicore__ inline void Process(const RangeTilingDataInt* tilingData);

private:
    __aicore__ inline void Compute(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void CopyOut(int64_t loopIdx, int64_t dataCount);
    __aicore__ inline void GenSequence(int64_t dataCount);

private:
    TPipe pipe_;
    constexpr static int64_t DB_BUFFER = 2;
    TQue<QuePosition::VECOUT, DB_BUFFER> calcuDataQueue_;
    TBuf<QuePosition::VECCALC> sequenceBuf_;

    GlobalTensor<T> outputGm_;

    int64_t curNumOfCore_{0};
    int64_t curPerOfCore_{0};
    int64_t curTailOfCore_{0};
    int64_t loopCount_{0};
    int64_t coreOffset_{0};

    int64_t start_{0};
    int64_t step_{0};
};

template <typename T>
__aicore__ inline void RangeInt<T>::Init(
    GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out, GM_ADDR workspace, const RangeTilingDataInt* tilingData)
{
    if (GetBlockIdx() + 1 == tilingData->usedCoreNum) {
        curNumOfCore_ = tilingData->numOfTailCore;
        curPerOfCore_ = tilingData->perOfTailCore;
        curTailOfCore_ = tilingData->tailOfTailCore;
        loopCount_ = tilingData->loopOfTailCore;
    } else {
        curNumOfCore_ = tilingData->numOfPerCore;
        curPerOfCore_ = tilingData->perOfPerCore;
        curTailOfCore_ = tilingData->tailOfPerCore;
        loopCount_ = tilingData->loopOfPerCore;
    }

    // SetBuffer
    coreOffset_ = GetBlockIdx() * tilingData->numOfPerCore;
    outputGm_.SetGlobalBuffer((__gm__ T*)out + coreOffset_, curNumOfCore_);

    // InitBuffer
    pipe_.InitBuffer(calcuDataQueue_, DB_BUFFER, Ops::Base::CeilAlign(curPerOfCore_, ONCE_ALGN_NUM_INT32) * sizeof(T));
    pipe_.InitBuffer(sequenceBuf_, Ops::Base::CeilAlign(curPerOfCore_, ONCE_ALGN_NUM_INT32) * sizeof(T));

    // GetValue
    start_ = tilingData->start;
    step_ = tilingData->delta;
}

template <typename T>
__aicore__ inline void RangeInt<T>::GenSequence(int64_t dataCount)
{
    LocalTensor<T> sequenceUb = sequenceBuf_.Get<T>();

    // 生成vci连续序列
    const T firstValue = static_cast<T>(0);
    uint32_t vciCount = static_cast<uint32_t>(dataCount);
    CreateVecIndex(sequenceUb, firstValue, vciCount);

    // 实现随机序列乘上step倍数，进行序列常驻
    const T stepScalarValue = static_cast<T>(step_);
    int32_t calCount = static_cast<int32_t>(dataCount);
    Muls(sequenceUb, sequenceUb, stepScalarValue, calCount);
}

template <typename T>
__aicore__ inline void RangeInt<T>::Compute(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<T> xCalcuUb = calcuDataQueue_.AllocTensor<T>();
    LocalTensor<T> sequenceUb = sequenceBuf_.Get<T>();
    int32_t calCount = static_cast<int32_t>(dataCount);

    // 等差数列的基本公式，a_n = a_1 + (n - 1) * d, 实现乘加操作
    const T stepScalarValue = static_cast<T>(step_);
    const T startScalarValue = static_cast<T>(start_);
    auto curOffset = static_cast<T>(coreOffset_ + loopIdx * curPerOfCore_);
    const T offsetScalarValue = startScalarValue + curOffset * stepScalarValue;
    Adds(xCalcuUb, sequenceUb, offsetScalarValue, calCount);

    calcuDataQueue_.EnQue<T>(xCalcuUb);
}

template <typename T>
__aicore__ inline void RangeInt<T>::CopyOut(int64_t loopIdx, int64_t dataCount)
{
    LocalTensor<T> xCalcuUb = calcuDataQueue_.DeQue<T>();
    DataCopyExtParams copyParamsYOut{
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(outputGm_[loopIdx * curPerOfCore_], xCalcuUb, copyParamsYOut);
    calcuDataQueue_.FreeTensor(xCalcuUb);
}

template <typename T>
__aicore__ inline void RangeInt<T>::Process(const RangeTilingDataInt* tilingData)
{
    if (GetBlockIdx() >= tilingData->usedCoreNum) {
        return;
    }

    GenSequence(curPerOfCore_);
    for (int64_t n = 0; n < loopCount_ - 1; n++) {
        Compute(n, curPerOfCore_);
        CopyOut(n, curPerOfCore_);
    }
    {
        Compute(loopCount_ - 1, curTailOfCore_);
        CopyOut(loopCount_ - 1, curTailOfCore_);
    }
}
} // namespace Range

#endif // RANGE_INT_H