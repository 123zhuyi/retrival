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
 * \file histogram_v2_simt_not_full_load_simt.h
 * \brief
 */
#ifndef HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_SIMT_H
#define HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_SIMT_H

#include "simt_api/asc_simt.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE = int32_t>
class HistogramV2SimtNotFullLoadGmAtomicAdd {
public:
    __aicore__ inline HistogramV2SimtNotFullLoadGmAtomicAdd(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMinMaxValue();

    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<OUT_TYPE> yGm_;

    COMPUTE_TYPE minValue_;
    COMPUTE_TYPE maxValue_;

    int32_t blockIdx_ = 0;
    int64_t bins_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;

    int64_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
};

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadGmAtomicAdd<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData)
{
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->bins_ = tilingData->bins;
    this->formerLength_ = tilingData->formerLength;
    this->tailLength_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE*>(y));
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCleanAtomic(
    __gm__ OUT_TYPE* yGmAddr, const int32_t blockIdx, const int32_t clearYCoreNum, const int64_t clearYIndexBase,
    const int64_t clearYDataLength)
{
    if (blockIdx >= clearYCoreNum) {
        return;
    }

    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < clearYDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t yIndex = clearYIndexBase + index;
        yGmAddr[yIndex] = static_cast<OUT_TYPE>(0);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtComputeAtomic(
    __gm__ X_TYPE* xGmAddr, __gm__ OUT_TYPE* yGmAddr, const int32_t blockIdx, const int32_t needXCoreNum,
    const int64_t xIndexBase, const int64_t coreDataLength, const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
    const int64_t bins)
{
    if (blockIdx >= needXCoreNum) {
        return;
    }

    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < coreDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t xIndex = xIndexBase + index;
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(xGmAddr[xIndex]);
        if (value >= minValue && value <= maxValue) {
            int64_t indexBin = static_cast<int64_t>((value - minValue) * bins / (maxValue - minValue));
            if (indexBin == bins) {
                indexBin -= 1;
            }
            asc_atomic_add(yGmAddr + indexBin, static_cast<OUT_TYPE>(1));
        }
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadGmAtomicAdd<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Process()
{
    if (blockIdx_ < GetBlockNum()) {
        ProcessMinMaxValue();

        int64_t xIndexBase = blockIdx_ * formerLength_;
        int64_t coreDataLength = (blockIdx_ == needXCoreNum_ - 1) ? tailLength_ : formerLength_;
        int64_t clearYIndexBase = blockIdx_ * clearYFactor_;
        int64_t clearYDataLength = (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;
        __gm__ OUT_TYPE* yGmAddr = (__gm__ OUT_TYPE*)yGm_.GetPhyAddr();
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();

        asc_vf_call<SimtCleanAtomic<X_TYPE, COMPUTE_TYPE, OUT_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, yGmAddr, blockIdx_, this->clearYCoreNum_, clearYIndexBase, clearYDataLength);
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        asc_vf_call<SimtComputeAtomic<X_TYPE, COMPUTE_TYPE, OUT_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yGmAddr, blockIdx_, needXCoreNum_, xIndexBase, coreDataLength, minValue_,
            maxValue_, bins_);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadGmAtomicAdd<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::ProcessMinMaxValue()
{
    minValue_ = static_cast<COMPUTE_TYPE>(minGm_.GetValue(0));
    maxValue_ = static_cast<COMPUTE_TYPE>(maxGm_.GetValue(0));

    if (minValue_ == maxValue_) {
        minValue_ = minValue_ - 1;
        maxValue_ = maxValue_ + 1;
    }
}

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_SIMT_H
