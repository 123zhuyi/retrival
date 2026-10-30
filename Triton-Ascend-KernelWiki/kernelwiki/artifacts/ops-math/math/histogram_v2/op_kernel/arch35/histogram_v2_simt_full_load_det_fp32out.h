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
 * \file histogram_v2_simt_full_load_det_fp32out.h
 * \brief Deterministic UB full-load kernel with float32 output.
 *
 * Three-phase deterministic algorithm (no float atomicAdd on GM):
 *   Phase 1: clearY cores zero the output GM (as int32) → SyncAll
 *   Phase 2: each core UB-int32 atomic-accumulates → SetAtomicAdd<int32_t> → write GM → SyncAll
 *   Phase 3: clearY cores cast their GM int32 slice in place to GM float32
 */
#ifndef HISTOGRAM_V2_SIMT_FULL_LOAD_DET_FP32OUT_H
#define HISTOGRAM_V2_SIMT_FULL_LOAD_DET_FP32OUT_H

#include "simt_api/asc_simt.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE>
class HistogramV2SimtFullLoadDetFp32Out {
public:
    __aicore__ inline HistogramV2SimtFullLoadDetFp32Out(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    // y GM viewed as int32 for Phase 1/2, and as float for Phase 3 write
    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<int32_t> yGmInt_;   // alias of y as int32 (for Phase 1 clear and Phase 2 atomic add)
    GlobalTensor<float> yGmFloat_;   // alias of y as float (for Phase 3 write)

    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yIntQue_;

    int32_t blockIdx_ = 0;
    int64_t binsDet_ = 0;
    int64_t formerLengthDet_ = 0;
    int64_t tailLengthDet_ = 0;
    int64_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
};

// Phase 2: UB int32 SIMT accumulate — same as UbSimtComputeDet but accumulates into UB int32
template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeDetFull(
    __gm__ X_TYPE* xGmAddr, __ubuf__ int32_t* yLocalIntAddr, const int64_t xIndexBase, const int64_t coreDataLength,
    const COMPUTE_TYPE minValue, const COMPUTE_TYPE maxValue, const COMPUTE_TYPE minMaxLength, const int64_t bins)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < coreDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t xIndex = xIndexBase + index;
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(xGmAddr[xIndex]);
        if (value >= minValue && value <= maxValue) {
            int32_t indexBin = static_cast<int32_t>((value - minValue) * bins / minMaxLength);
            if (indexBin == bins) {
                indexBin -= 1;
            }
            asc_atomic_add(yLocalIntAddr + indexBin, static_cast<int32_t>(1));
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmCastDetFullFp32Out(
    __gm__ int32_t* yGmIntAddr, __gm__ float* yGmFloatAddr, const int64_t clearYIndexBase, const int64_t clearYDataLength)
{
    for (int32_t index = static_cast<int32_t>(threadIdx.x); index < clearYDataLength;
         index += static_cast<int32_t>(blockDim.x)) {
        int64_t yIndex = clearYIndexBase + index;
        int32_t yValue = yGmIntAddr[yIndex];
        yGmFloatAddr[yIndex] = static_cast<float>(yValue);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
    TPipe* tPipe)
{
    this->pipe_ = tPipe;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->binsDet_ = tilingData->bins;
    this->formerLengthDet_ = tilingData->formerLength;
    this->tailLengthDet_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    // Both views point to the same y GM address
    this->yGmInt_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    this->yGmFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    this->pipe_->InitBuffer(this->yIntQue_, 1, this->binsDet_ * sizeof(int32_t));
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }

    int64_t clearYIndexBaseDeter = blockIdx_ * clearYFactor_;
    int64_t clearYDataLengthDeter = (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;

    // --------------- Phase 1: clearY cores zero the output GM (as int32) ---------------
    if (blockIdx_ < clearYCoreNum_) {
        LocalTensor<int32_t> yLocalInt = yIntQue_.template AllocTensor<int32_t>();
        Duplicate(yLocalInt, static_cast<int32_t>(0), clearYDataLengthDeter);
        yIntQue_.EnQue(yLocalInt);

        yLocalInt = yIntQue_.template DeQue<int32_t>();
        DataCopyExtParams dataCopyExtParamsClear{
            static_cast<uint16_t>(1), static_cast<uint32_t>(static_cast<int64_t>(clearYDataLengthDeter) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(yGmInt_[clearYIndexBaseDeter], yLocalInt, dataCopyExtParamsClear);
        yIntQue_.template FreeTensor<int32_t>(yLocalInt);
    }

#ifndef __CCE_UT_TEST__
    SyncAll();
#endif

    // --------------- Phase 2: each X-processing core UB-int32 accumulate → GM int32 atomic add ---------------
    if (blockIdx_ < needXCoreNum_) {
        COMPUTE_TYPE minValue = static_cast<COMPUTE_TYPE>(minGm_(0));
        COMPUTE_TYPE maxValue = static_cast<COMPUTE_TYPE>(maxGm_(0));

        if (minValue == maxValue) {
            minValue = minValue - 1;
            maxValue = maxValue + 1;
        }

        COMPUTE_TYPE minMaxLength = maxValue - minValue;
        int64_t xIndexBaseDet = blockIdx_ * formerLengthDet_;
        int64_t coreDataLengthDet = (blockIdx_ == needXCoreNum_ - 1) ? tailLengthDet_ : formerLengthDet_;

        LocalTensor<int32_t> yLocalInt = yIntQue_.template AllocTensor<int32_t>();
        Duplicate(yLocalInt, static_cast<int32_t>(0), binsDet_);
        yIntQue_.EnQue(yLocalInt);
        yLocalInt = yIntQue_.template DeQue<int32_t>();

        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();
        __ubuf__ int32_t* yLocalIntAddr = (__ubuf__ int32_t*)yLocalInt.GetPhyAddr();

        asc_vf_call<UbSimtComputeDetFull<X_TYPE, COMPUTE_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalIntAddr, xIndexBaseDet, coreDataLengthDet, minValue, maxValue,
            minMaxLength, binsDet_);

        // Wait for SIMT VF compute to finish before DMA write
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // Atomic add UB int32 result to GM (int32 atomic, no float non-determinism)
        DataCopyExtParams dataCopyExtParamsAdd{
            static_cast<uint16_t>(1), static_cast<uint32_t>(static_cast<int64_t>(binsDet_) * sizeof(int32_t)), 0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(yGmInt_[0], yLocalInt, dataCopyExtParamsAdd);
        SetAtomicNone();
        yIntQue_.template FreeTensor<int32_t>(yLocalInt);
    }

#ifndef __CCE_UT_TEST__
    SyncAll();
#endif

    // --------------- Phase 3: clearY cores cast GM int32 slice in place to GM float ---------------
    if (blockIdx_ < clearYCoreNum_) {
        __gm__ int32_t* yGmIntAddr = (__gm__ int32_t*)yGmInt_.GetPhyAddr();
        __gm__ float* yGmFloatAddr = (__gm__ float*)yGmFloat_.GetPhyAddr();
        asc_vf_call<GmCastDetFullFp32Out>(
            dim3{THREAD_NUM, 1, 1}, yGmIntAddr, yGmFloatAddr, clearYIndexBaseDeter, clearYDataLengthDeter);
    }
}

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_FULL_LOAD_DET_FP32OUT_H
