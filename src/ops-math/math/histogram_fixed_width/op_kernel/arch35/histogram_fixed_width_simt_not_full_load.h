/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HISTOGRAM_FIXED_WIDTH_SIMT_NOT_FULL_LOAD_H
#define HISTOGRAM_FIXED_WIDTH_SIMT_NOT_FULL_LOAD_H

#include "simt_api/asc_simt.h"
#include "histogram_fixed_width_tilingdata.h"
#include "histogram_fixed_width_tilingkey.h"
#include "histogram_fixed_width_simt_common.h"

namespace HistogramFixedWidthSIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE>
class HistogramFixedWidthSimtNotFullLoad {
public:
    __aicore__ inline HistogramFixedWidthSimtNotFullLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR range, GM_ADDR y,
                                const HistogramFixedWidthSimtTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SimtCompute(__gm__ X_TYPE* xGmAddr, const int64_t xIndexBase, const int64_t coreDataLength,
                                       const COMPUTE_TYPE rangeVal, const COMPUTE_TYPE invRange);
    __aicore__ inline void AtomicAddToGm(uint32_t ubLoop, int32_t yLocalNum);

    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> rangeGm_;
    GlobalTensor<int32_t> yGm_;
    LocalTensor<int32_t> yLocal_;

    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yQue_;

    COMPUTE_TYPE minValue_;
    COMPUTE_TYPE maxValue_;

    int32_t blockIdx_ = 0;
    int32_t bins_ = 0;
    uint32_t ubNumCanUse_ = 0;
    uint32_t ubLoopNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;

    uint32_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    uint32_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
};

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtNotFullLoad<X_TYPE, COMPUTE_TYPE>::Init(
    GM_ADDR x, GM_ADDR range, GM_ADDR y, const HistogramFixedWidthSimtTilingData* tilingData, TPipe* tPipe)
{
    this->pipe_ = tPipe;
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->bins_ = tilingData->bins;
    this->ubNumCanUse_ = tilingData->ubNumCanUse;
    this->ubLoopNum_ = tilingData->ubLoopNum;
    this->formerLength_ = tilingData->formerLength;
    this->tailLength_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->rangeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(range));
    this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    this->pipe_->InitBuffer(this->yQue_, 1, this->ubNumCanUse_ * sizeof(int32_t));
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_callee__ __aicore__ inline void DoBinNotFull(__ubuf__ int32_t* yLocalAddr, const COMPUTE_TYPE v,
                                                    const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
                                                    const COMPUTE_TYPE rangeVal, const COMPUTE_TYPE invRange,
                                                    const int32_t bins, const int32_t binsM1, const int64_t ubBase,
                                                    const int64_t ubEnd)
{
    if (v == INFINITY || v != v || (v == -INFINITY && minValue != -INFINITY)) {
        return;
    }
    int32_t binIndex;
    if (minValue == -INFINITY) {
        binIndex = binsM1;
    } else if (maxValue == INFINITY) {
        binIndex = 0;
    } else {
        COMPUTE_TYPE clamped = v;
        if (clamped < minValue) {
            clamped = minValue;
        } else if (clamped > maxValue) {
            clamped = maxValue;
        }
        if constexpr (std::is_integral<COMPUTE_TYPE>::value) {
            binIndex = static_cast<int32_t>((clamped - minValue) * bins / rangeVal);
        } else {
            binIndex = static_cast<int32_t>((clamped - minValue) * invRange * bins);
        }
        if (binIndex < 0) {
            binIndex = 0;
        } else if (binIndex >= bins) {
            binIndex = binsM1;
        }
    }
    if (binIndex >= ubBase && binIndex < ubEnd) {
        asc_atomic_add(yLocalAddr + (binIndex - ubBase), static_cast<int32_t>(1));
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeNotFull(
    __gm__ X_TYPE* xGmAddr, __ubuf__ int32_t* yLocalAddr, const int32_t blockIdx, const uint32_t needXCoreNum,
    const int64_t xIndexBase, const int64_t coreDataLength, const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
    const COMPUTE_TYPE rangeVal, const COMPUTE_TYPE invRange, const int32_t bins, const int64_t ubLoop,
    const int32_t ubNumCanUse)
{
    if (blockIdx >= static_cast<int32_t>(needXCoreNum)) {
        return;
    }

    int32_t binsM1 = bins - 1;
    int64_t ubBase = ubLoop * ubNumCanUse;
    int64_t ubEnd = (ubLoop + 1) * ubNumCanUse;
    int64_t base = static_cast<int64_t>(threadIdx.x);
    int32_t stride = static_cast<int32_t>(blockDim.x);
    constexpr int32_t UNROLL_FACTOR = 6;
    int64_t stepN = (int64_t)stride * UNROLL_FACTOR;
    int64_t limitN = coreDataLength - stepN + stride;

    for (; base < limitN; base += stepN) {
        COMPUTE_TYPE v0 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base));
        COMPUTE_TYPE v1 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride));
        COMPUTE_TYPE v2 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_2));
        COMPUTE_TYPE v3 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_3));
        COMPUTE_TYPE v4 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_4));
        COMPUTE_TYPE v5 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_5));
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v0, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v1, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v2, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v3, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v4, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v5, minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase,
                                           ubEnd);
    }
    for (; base < coreDataLength; base += stride) {
        DoBinNotFull<X_TYPE, COMPUTE_TYPE>(yLocalAddr, static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base)),
                                           minValue, maxValue, rangeVal, invRange, bins, binsM1, ubBase, ubEnd);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtNotFullLoad<X_TYPE, COMPUTE_TYPE>::Process()
{
    if (blockIdx_ < GetBlockNum()) {
        minValue_ = static_cast<COMPUTE_TYPE>(rangeGm_.GetValue(0));
        maxValue_ = static_cast<COMPUTE_TYPE>(rangeGm_.GetValue(1));
        COMPUTE_TYPE rangeVal = maxValue_ - minValue_;
        COMPUTE_TYPE invRange = static_cast<COMPUTE_TYPE>(1) / rangeVal;

        int64_t clearYIndexBase = blockIdx_ * clearYFactor_;
        int32_t clearYDataLength = (blockIdx_ == static_cast<int32_t>(clearYCoreNum_) - 1) ?
                                       static_cast<int32_t>(clearYTail_) :
                                       static_cast<int32_t>(clearYFactor_);
        __gm__ int32_t* yGmAddr = (__gm__ int32_t*)yGm_.GetPhyAddr();
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();

        asc_vf_call<SimtCleanY<X_TYPE, COMPUTE_TYPE, uint32_t>>(dim3{THREAD_NUM, 1, 1}, yGmAddr, blockIdx_,
                                                                clearYCoreNum_, clearYIndexBase, clearYDataLength);
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        int64_t xIndexBase = blockIdx_ * formerLength_;
        int64_t coreDataLength = (blockIdx_ == static_cast<int32_t>(needXCoreNum_) - 1) ? tailLength_ : formerLength_;
        SimtCompute(xGmAddr, xIndexBase, coreDataLength, rangeVal, invRange);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtNotFullLoad<X_TYPE, COMPUTE_TYPE>::SimtCompute(
    __gm__ X_TYPE* xGmAddr, const int64_t xIndexBase, const int64_t coreDataLength, const COMPUTE_TYPE rangeVal,
    const COMPUTE_TYPE invRange)
{
    for (uint32_t ubLoop = 0; ubLoop < ubLoopNum_; ubLoop++) {
        int32_t yLocalNum = (ubLoop == ubLoopNum_ - 1) ? (bins_ - (ubLoopNum_ - 1) * ubNumCanUse_) :
                                                         static_cast<int32_t>(ubNumCanUse_);
        yLocal_ = yQue_.template AllocTensor<int32_t>();
        Duplicate(yLocal_, static_cast<int32_t>(0), yLocalNum);
        yQue_.EnQue(yLocal_);
        yLocal_ = yQue_.template DeQue<int32_t>();
        __ubuf__ int32_t* yLocalAddr = (__ubuf__ int32_t*)yLocal_.GetPhyAddr();
        asc_vf_call<UbSimtComputeNotFull<X_TYPE, COMPUTE_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr, blockIdx_, needXCoreNum_, xIndexBase, coreDataLength,
            minValue_, maxValue_, rangeVal, invRange, bins_, ubLoop, static_cast<int32_t>(ubNumCanUse_));
        yQue_.EnQue(yLocal_);
        yLocal_ = yQue_.template DeQue<int32_t>();
        AtomicAddToGm(ubLoop, yLocalNum);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtNotFullLoad<X_TYPE, COMPUTE_TYPE>::AtomicAddToGm(uint32_t ubLoop,
                                                                                               int32_t yLocalNum)
{
    DataCopyExtParams dataCopyExtParamsAdd{static_cast<uint16_t>(1), static_cast<uint32_t>(yLocalNum * sizeof(int32_t)),
                                           0, 0, 0};

    SetAtomicAdd<int32_t>();
    DataCopyPad(yGm_[ubLoop * ubNumCanUse_], yLocal_, dataCopyExtParamsAdd);
    SetAtomicNone();
    yQue_.template FreeTensor<int32_t>(yLocal_);
}
} // namespace HistogramFixedWidthSIMT

#endif // HISTOGRAM_FIXED_WIDTH_SIMT_NOT_FULL_LOAD_H
