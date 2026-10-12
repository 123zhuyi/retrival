/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HISTOGRAM_FIXED_WIDTH_SIMT_FULL_LOAD_H
#define HISTOGRAM_FIXED_WIDTH_SIMT_FULL_LOAD_H

#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "histogram_fixed_width_tilingdata.h"
#include "histogram_fixed_width_tilingkey.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace HistogramFixedWidthSIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE>
class HistogramFixedWidthSimtFullLoad {
public:
    __aicore__ inline HistogramFixedWidthSimtFullLoad(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR range, GM_ADDR y,
                                const HistogramFixedWidthSimtTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> rangeGm_;
    GlobalTensor<int32_t> yGm_;
    LocalTensor<int32_t> yLocal_;
    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yQue_;
    int32_t blockIdx_ = 0;
    int32_t bins_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;
    uint32_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    uint32_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
    uint32_t needCoreNum_ = 0;
};

template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtCleanWsReduce(__gm__ int32_t* yGmAddr,
                                                                              const int32_t blockIdx,
                                                                              const uint32_t clearYCoreNum,
                                                                              const int64_t clearYIndexBase,
                                                                              const int32_t clearYDataLength)
{
    if (blockIdx >= static_cast<int32_t>(clearYCoreNum)) {
        return;
    }
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < clearYDataLength; i += static_cast<int32_t>(blockDim.x))
        yGmAddr[clearYIndexBase + i] = 0;
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_callee__ __aicore__ inline void DoBin(__ubuf__ int32_t* yLocalAddr, const COMPUTE_TYPE v,
                                             const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
                                             const COMPUTE_TYPE rangeVal, const COMPUTE_TYPE invRange,
                                             const int32_t bins, const int32_t binsM1)
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
    asc_atomic_add(yLocalAddr + binIndex, 1);
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeWsReduce(
    __gm__ X_TYPE* xGmAddr, __ubuf__ int32_t* yLocalAddr, const int64_t xIndexBase, const int64_t coreDataLength,
    const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue, const COMPUTE_TYPE rangeVal, const COMPUTE_TYPE invRange,
    const int32_t bins)
{
    int32_t binsM1 = bins - 1;
    int32_t stride = static_cast<int32_t>(blockDim.x);
    int32_t tid = static_cast<int32_t>(threadIdx.x);
    constexpr int64_t UNROLL_FACTOR = 6;
    constexpr int64_t STRIDE_OFFSET_2 = 2;
    constexpr int64_t STRIDE_OFFSET_3 = 3;
    constexpr int64_t STRIDE_OFFSET_4 = 4;
    constexpr int64_t STRIDE_OFFSET_5 = 5;
    int64_t stepN = (int64_t)stride * UNROLL_FACTOR;
    int64_t limitN = coreDataLength - stepN + stride;
    int64_t base = tid;

    for (; base < limitN; base += stepN) {
        COMPUTE_TYPE v0 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base));
        COMPUTE_TYPE v1 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride));
        COMPUTE_TYPE v2 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_2));
        COMPUTE_TYPE v3 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_3));
        COMPUTE_TYPE v4 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_4));
        COMPUTE_TYPE v5 = static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base + stride * STRIDE_OFFSET_5));
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v0, minValue, maxValue, rangeVal, invRange, bins, binsM1);
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v1, minValue, maxValue, rangeVal, invRange, bins, binsM1);
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v2, minValue, maxValue, rangeVal, invRange, bins, binsM1);
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v3, minValue, maxValue, rangeVal, invRange, bins, binsM1);
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v4, minValue, maxValue, rangeVal, invRange, bins, binsM1);
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, v5, minValue, maxValue, rangeVal, invRange, bins, binsM1);
    }
    for (; base < coreDataLength; base += stride) {
        DoBin<X_TYPE, COMPUTE_TYPE>(yLocalAddr, static_cast<COMPUTE_TYPE>(asc_ldcg(xGmAddr + xIndexBase + base)),
                                    minValue, maxValue, rangeVal, invRange, bins, binsM1);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtFullLoad<X_TYPE, COMPUTE_TYPE>::Init(
    GM_ADDR x, GM_ADDR range, GM_ADDR y, const HistogramFixedWidthSimtTilingData* tilingData, TPipe* tPipe)
{
    this->pipe_ = tPipe;
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->bins_ = tilingData->bins;
    this->formerLength_ = tilingData->formerLength;
    this->tailLength_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;
    this->needCoreNum_ = tilingData->needCoreNum;

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->rangeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(range));
    this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));

    this->pipe_->InitBuffer(this->yQue_, 1, this->bins_ * sizeof(int32_t));
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramFixedWidthSimtFullLoad<X_TYPE, COMPUTE_TYPE>::Process()
{
    if (blockIdx_ < GetBlockNum()) {
        COMPUTE_TYPE minValue = static_cast<COMPUTE_TYPE>(this->rangeGm_(0));
        COMPUTE_TYPE maxValue = static_cast<COMPUTE_TYPE>(this->rangeGm_(1));
        COMPUTE_TYPE minMaxLength = maxValue - minValue;
        COMPUTE_TYPE invRange = static_cast<COMPUTE_TYPE>(1) / minMaxLength;

        int64_t clearYIndexBase = blockIdx_ * this->clearYFactor_;
        int32_t clearYDataLength = blockIdx_ == static_cast<int32_t>(this->clearYCoreNum_) - 1 ?
                                       static_cast<int32_t>(this->clearYTail_) :
                                       static_cast<int32_t>(this->clearYFactor_);

        __gm__ int32_t* yGmAddr = (__gm__ int32_t*)yGm_.GetPhyAddr();
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();

        asc_vf_call<SimtCleanWsReduce<X_TYPE, COMPUTE_TYPE>>(dim3{THREAD_NUM, 1, 1}, yGmAddr, blockIdx_,
                                                             this->clearYCoreNum_, clearYIndexBase, clearYDataLength);

#ifndef __CCE_UT_TEST__
        SyncAll();
#endif

        if (blockIdx_ < static_cast<int32_t>(needXCoreNum_)) {
            this->yLocal_ = this->yQue_.template AllocTensor<int32_t>();
            Duplicate(this->yLocal_, static_cast<int32_t>(0), this->bins_);

            __ubuf__ int32_t* yLocalAddr = (__ubuf__ int32_t*)this->yLocal_.GetPhyAddr();

            int64_t xIndexBase = blockIdx_ * this->formerLength_;
            int64_t coreDataLength = blockIdx_ == static_cast<int32_t>(needXCoreNum_) - 1 ? this->tailLength_ :
                                                                                            this->formerLength_;

            asc_vf_call<UbSimtComputeWsReduce<X_TYPE, COMPUTE_TYPE>>(dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr,
                                                                     xIndexBase, coreDataLength, minValue, maxValue,
                                                                     minMaxLength, invRange, this->bins_);

            this->yQue_.EnQue(this->yLocal_);
            this->yLocal_ = this->yQue_.template DeQue<int32_t>();
            SetAtomicAdd<int32_t>();
            DataCopyExtParams dataCopyExtParamsAdd{static_cast<uint16_t>(1),
                                                   static_cast<uint32_t>(this->bins_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->yGm_, this->yLocal_, dataCopyExtParamsAdd);
            SetAtomicNone();

            this->yQue_.template FreeTensor<int32_t>(yLocal_);
        }
    }
}

} // namespace HistogramFixedWidthSIMT

#endif // HISTOGRAM_FIXED_WIDTH_SIMT_FULL_LOAD_H
