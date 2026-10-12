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
 * \file histogram_v2_simt_not_full_load.h
 * \brief
 */
#ifndef HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_H
#define HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_H

#include "simt_api/asc_simt.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE = int32_t>
class HistogramV2SimtNotFullLoad {
public:
    __aicore__ inline HistogramV2SimtNotFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMinMaxValue();
    __aicore__ inline void SimtCompute(__gm__ X_TYPE* xGmAddr, const int64_t xIndexBase, const int64_t coreDataLength);
    __aicore__ inline void AtomicAddToGm(int64_t ubLoop, int64_t yLocalNum);

    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<OUT_TYPE> yGm_;
    LocalTensor<OUT_TYPE> yLocal_;

    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yQue_;

    COMPUTE_TYPE minValue_;
    COMPUTE_TYPE maxValue_;

    int32_t blockIdx_ = 0;
    int64_t bins_ = 0;
    int64_t ubNumCanUse_ = 0;
    int64_t ubLoopNum_ = 0;
    int64_t formerLength_ = 0;
    int64_t tailLength_ = 0;

    int64_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
};

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
    TPipe* tPipe)
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
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    this->yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ OUT_TYPE*>(y));

    this->pipe_->InitBuffer(this->yQue_, 1, this->ubNumCanUse_ * sizeof(OUT_TYPE));
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SimtClean(
    __gm__ OUT_TYPE* yGmAddr, const int32_t blockIdx, const int64_t clearYCoreNum, const int64_t clearYIndexBase,
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
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeNotFull(
    __gm__ X_TYPE* xGmAddr, __ubuf__ OUT_TYPE* yLocalAddr, const int32_t blockIdx, const int64_t needXCoreNum,
    const int64_t xIndexBase, const int64_t coreDataLength, const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue,
    const int64_t bins, const int64_t ubLoop, const int64_t ubNumCanUse)
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
            if (ubLoop * ubNumCanUse <= indexBin && indexBin < (ubLoop + 1) * ubNumCanUse) {
                int64_t indexBinNormal = indexBin - ubLoop * ubNumCanUse;
                asc_atomic_add(yLocalAddr + indexBinNormal, static_cast<OUT_TYPE>(1));
            }
        }
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::Process()
{
    if (blockIdx_ < GetBlockNum()) {
        ProcessMinMaxValue();

        int64_t clearYIndexBase = blockIdx_ * clearYFactor_;
        int64_t clearYDataLength = (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;
        __gm__ OUT_TYPE* yGmAddr = (__gm__ OUT_TYPE*)yGm_.GetPhyAddr();
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();

        asc_vf_call<SimtClean<X_TYPE, COMPUTE_TYPE, OUT_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, yGmAddr, blockIdx_, clearYCoreNum_, clearYIndexBase, clearYDataLength);
#ifndef __CCE_UT_TEST__
        SyncAll();
#endif
        int64_t xIndexBase = blockIdx_ * formerLength_;
        int64_t coreDataLength = (blockIdx_ == needXCoreNum_ - 1) ? tailLength_ : formerLength_;
        SimtCompute(xGmAddr, xIndexBase, coreDataLength);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::ProcessMinMaxValue()
{
    minValue_ = static_cast<COMPUTE_TYPE>(minGm_.GetValue(0));
    maxValue_ = static_cast<COMPUTE_TYPE>(maxGm_.GetValue(0));

    if (minValue_ == maxValue_) {
        minValue_ = minValue_ - 1;
        maxValue_ = maxValue_ + 1;
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::SimtCompute(
    __gm__ X_TYPE* xGmAddr, const int64_t xIndexBase, const int64_t coreDataLength)
{
    for (int64_t ubLoop = 0; ubLoop < ubLoopNum_; ubLoop++) {
        int64_t yLocalNum = (ubLoop == ubLoopNum_ - 1) ? (bins_ - (ubLoopNum_ - 1) * ubNumCanUse_) : ubNumCanUse_;
        yLocal_ = yQue_.template AllocTensor<OUT_TYPE>();
        Duplicate(yLocal_, static_cast<OUT_TYPE>(0), yLocalNum);
        yQue_.EnQue(yLocal_);
        yLocal_ = yQue_.template DeQue<OUT_TYPE>();
        __ubuf__ OUT_TYPE* yLocalAddr = (__ubuf__ OUT_TYPE*)yLocal_.GetPhyAddr();
        asc_vf_call<UbSimtComputeNotFull<X_TYPE, COMPUTE_TYPE, OUT_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalAddr, blockIdx_, needXCoreNum_, xIndexBase, coreDataLength,
            minValue_, maxValue_, bins_, ubLoop, ubNumCanUse_);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        AtomicAddToGm(ubLoop, yLocalNum);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE, typename OUT_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoad<X_TYPE, COMPUTE_TYPE, OUT_TYPE>::AtomicAddToGm(
    int64_t ubLoop, int64_t yLocalNum)
{
    DataCopyExtParams dataCopyExtParamsAdd{
        static_cast<uint16_t>(1), static_cast<uint32_t>(yLocalNum * sizeof(OUT_TYPE)), 0, 0, 0};

    SetAtomicAdd<OUT_TYPE>();
    DataCopyPad(yGm_[ubLoop * ubNumCanUse_], yLocal_, dataCopyExtParamsAdd);
    SetAtomicNone();
    yQue_.template FreeTensor<OUT_TYPE>(yLocal_);
}
} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_H
