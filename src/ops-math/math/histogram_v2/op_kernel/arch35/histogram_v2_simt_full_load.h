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
 * \file histogram_v2_simt_full_load.h
 * \brief
 */
#ifndef HISTOGRAM_V2_SIMT_H
#define HISTOGRAM_V2_SIMT_H

#include "simt_api/asc_simt.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE = int32_t>
class HistogramV2SimtFullLoad {
public:
    __aicore__ inline HistogramV2SimtFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<OUT_TYPE> yGm_;

    LocalTensor<OUT_TYPE> yLocal_;

    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yQue_;

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
__aicore__ inline void HistogramV2SimtFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
    TPipe* tPipe)
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

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE*>(y));

    this->pipe_->InitBuffer(this->yQue_, 1, this->bins_ * sizeof(OUT_TYPE));
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtCompute(
    __gm__ X_TYPE* xGmAddr, __ubuf__ OUT_TYPE* yLocalAddr, const int64_t xIndexBase, const int64_t coreDataLength,
    const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue, const COMPUTE_TYPE minMaxLength, const int64_t bins)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < coreDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t xIndex = xIndexBase + index;
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(xGmAddr[xIndex]);
        if (value >= minValue && value <= maxValue) {
            int32_t index = static_cast<int32_t>((value - minValue) * bins / minMaxLength);
            if (index == bins) {
                index -= 1;
            }
            asc_atomic_add(yLocalAddr + index, static_cast<int32_t>(1));
        }
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Process()
{
    if (blockIdx_ < GetBlockNum()) {
        this->yLocal_ = this->yQue_.template AllocTensor<OUT_TYPE>();
        Duplicate(this->yLocal_, static_cast<OUT_TYPE>(0), this->bins_);

        if (blockIdx_ < clearYCoreNum_) {
            this->yQue_.EnQue(this->yLocal_);
            this->yLocal_ = this->yQue_.template DeQue<OUT_TYPE>();

            int64_t clearYIndexBase = blockIdx_ * this->clearYFactor_;
            int64_t clearYDataLength = blockIdx_ == this->clearYCoreNum_ - 1 ? this->clearYTail_ : this->clearYFactor_;

            DataCopyExtParams dataCopyExtParams{
                static_cast<uint16_t>(1), static_cast<uint32_t>(clearYDataLength * sizeof(OUT_TYPE)), 0, 0, 0};
            DataCopyPad(yGm_[clearYIndexBase], this->yLocal_, dataCopyExtParams);
        }

#ifndef __CCE_UT_TEST__
        SyncAll();
#endif

        if (blockIdx_ < needXCoreNum_) {
            COMPUTE_TYPE minValue = static_cast<COMPUTE_TYPE>(this->minGm_(0));
            COMPUTE_TYPE maxValue = static_cast<COMPUTE_TYPE>(this->maxGm_(0));

            if (minValue == maxValue) {
                minValue = minValue - 1;
                maxValue = maxValue + 1;
            }

            COMPUTE_TYPE minMaxLength = maxValue - minValue;

            int64_t xIndexBase = blockIdx_ * this->formerLength_;
            int64_t coreDataLength = blockIdx_ == needXCoreNum_ - 1 ? this->tailLength_ : this->formerLength_;

            __gm__ OUT_TYPE* yGmAddr = (__gm__ OUT_TYPE*)yGm_.GetPhyAddr();
            __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();
            __ubuf__ OUT_TYPE* yLocalAddr = (__ubuf__ OUT_TYPE*)this->yLocal_.GetPhyAddr();

            asc_vf_call<UbSimtCompute<X_TYPE, COMPUTE_TYPE, OUT_TYPE>>(
                dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr, xIndexBase, coreDataLength, minValue, maxValue,
                minMaxLength, this->bins_);

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);

            DataCopyExtParams dataCopyExtParamsAdd{
                static_cast<uint16_t>(1), static_cast<uint32_t>(this->bins_ * sizeof(OUT_TYPE)), 0, 0, 0};

            SetAtomicAdd<OUT_TYPE>();
            DataCopyPad(yGm_, this->yLocal_, dataCopyExtParamsAdd);
            SetAtomicNone();
        }

        this->yQue_.template FreeTensor<OUT_TYPE>(yLocal_);
    }
}

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_H
