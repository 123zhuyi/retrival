/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file eye.h
 * \brief eye
 */

#ifndef ASCENDC_EYE_H_
#define ASCENDC_EYE_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace Eye {
using namespace AscendC;

constexpr uint32_t USED_THREAD = 512;

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void EyeSimt(
    __gm__ T* y, U allAxis, U perBatchAxis, uint32_t magic, uint32_t shift,
    int64_t numRows, int64_t numColumns);

template <typename T, typename U> class EyeKernel {
public:
    __aicore__ inline EyeKernel(const EyeForAscendCTilingData& tilingData, TPipe& pipe)
        : td_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void DumpGmZero();

private:
    GlobalTensor<T> y_;
    TQue<QuePosition::VECOUT, 1> zeroBuf_;
    TPipe &pipe_;

    U allAxis_{ 0 };
    U perBatchAxis_{ 0 };
    U blockIdx_{ 0 };
    U blockNum_{ 0 };
    uint32_t shift_{ 0 };
    uint32_t magic_{ 0 };
    int64_t loopNum_{ 0 };
    int64_t tailLoopLength_{ 0 };
    int64_t curCoreData_{ 0 };
    const EyeForAscendCTilingData &td_;
};

template <typename T, typename U>
__aicore__ inline void EyeKernel<T, U>::Init(GM_ADDR y)
{
#if defined(__CCE_KT_TEST__)
    perBatchAxis_ = std::min(td_.numRows, td_.numColumns);
#else
    perBatchAxis_ = min(td_.numRows, td_.numColumns);
#endif
    allAxis_ = static_cast<U>(td_.batch) * perBatchAxis_;

    y_.SetGlobalBuffer((__gm__ T *)(y));
    pipe_.InitBuffer(zeroBuf_, 1, td_.loopLength * sizeof(T));

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    curCoreData_ = blockIdx_ != (td_.usedCoreNum - 1) ? td_.normBlockData : td_.tailBlockData;
    loopNum_ = curCoreData_ / td_.loopLength;
    tailLoopLength_ = curCoreData_ - loopNum_ * td_.loopLength;
    if constexpr (sizeof(U) == sizeof(uint32_t)) {
        GetUintDivMagicAndShift(magic_, shift_, perBatchAxis_);
    }
}

template <typename T, typename U> 
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void EyeSimt(
    __gm__ T* y, U allAxis, U perBatchAxis, uint32_t magic, uint32_t shift, int64_t numRows, int64_t numColumns)
{
    for (U i = blockIdx.x * blockDim.x + threadIdx.x; i < allAxis; i += gridDim.x * blockDim.x) {
        if constexpr (sizeof(U) == sizeof(uint32_t)) {
            uint32_t t1 = __umulhi(i, magic);
            t1 = t1 + i;
            uint32_t batchIdx = t1 >> shift;
            uint32_t rowIdx = i - batchIdx * perBatchAxis;
            y[batchIdx * static_cast<U>(numRows) * static_cast<U>(numColumns) +
               rowIdx * static_cast<U>(numColumns) + rowIdx] = static_cast<T>(1);
        } else {
            U batchIdx = i / perBatchAxis;
            U rowIdx = i - batchIdx * perBatchAxis;
            y[batchIdx * static_cast<U>(numRows) * static_cast<U>(numColumns) +
               rowIdx * static_cast<U>(numColumns) + rowIdx] = static_cast<T>(1);
        }
    }
}

template <typename T, typename U> __aicore__ inline void EyeKernel<T, U>::DumpGmZero()
{
    LocalTensor<T> tmpLocal = zeroBuf_.AllocTensor<T>();
    Duplicate(tmpLocal, static_cast<T>(0), td_.loopLength);
    zeroBuf_.EnQue<T>(tmpLocal);
    LocalTensor<T> srcLocal = zeroBuf_.DeQue<T>();
    DataCopyExtParams copyParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(td_.loopLength * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    for (int64_t i = 0; i < loopNum_; i++) {
        DataCopyPad(y_[blockIdx_ * td_.normBlockData + i * td_.loopLength], srcLocal, copyParams);
    }
    if (tailLoopLength_ > 0) {
        copyParams.blockLen = tailLoopLength_ * sizeof(T);
        DataCopyPad(y_[blockIdx_ * td_.normBlockData + loopNum_ * td_.loopLength], srcLocal, copyParams);
    }
    zeroBuf_.FreeTensor(srcLocal);
}

template <typename T, typename U> __aicore__ inline void EyeKernel<T, U>::Process()
{
    if (blockIdx_ < td_.usedCoreNum) {
        DumpGmZero();
    }
    SyncAll();
    asc_vf_call<EyeSimt<T, U>>(dim3(USED_THREAD), (__gm__ T*)(y_.GetPhyAddr()),
        allAxis_, perBatchAxis_, magic_, shift_, td_.numRows, td_.numColumns);
}
} // namespace Eye

#endif // ASCENDC_EYE_H_
