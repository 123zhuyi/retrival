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
 * \file histogram_v2_simt_not_full_load_det_fp32out.h
 * \brief Deterministic UB not-full-load kernel with float32 output.
 *
 * Three-phase deterministic algorithm (no float atomicAdd on GM):
 *   Phase 1: clearY cores zero the output GM (as int32) → SyncAll
 *   Phase 2: each core loops over UB chunks, accumulates UB int32 → SetAtomicAdd<int32_t> → write GM → SyncAll
 *   Phase 3: clearY cores cast their GM int32 slice in place to GM float32
 */
#ifndef HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_DET_FP32OUT_H
#define HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_DET_FP32OUT_H

#include "simt_api/asc_simt.h"

namespace HistogramV2SIMT {
using namespace AscendC;

template <typename X_TYPE, typename COMPUTE_TYPE>
class HistogramV2SimtNotFullLoadDetFp32Out {
public:
    __aicore__ inline HistogramV2SimtNotFullLoadDetFp32Out(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
        TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessMinMaxValue();
    __aicore__ inline void Phase2SimtCompute(__gm__ X_TYPE* xGmAddr, const int64_t xIndexBase,
                                              const int64_t coreDataLength);

    // y GM viewed as int32 for Phase 1/2, and as float for Phase 3 write
    GlobalTensor<X_TYPE> xGm_;
    GlobalTensor<X_TYPE> minGm_;
    GlobalTensor<X_TYPE> maxGm_;
    GlobalTensor<int32_t> yGmInt_;   // alias of y as int32
    GlobalTensor<float> yGmFloat_;   // alias of y as float

    TPipe* pipe_;
    TQue<TPosition::VECOUT, 1> yIntQue_;

    COMPUTE_TYPE minValue_;
    COMPUTE_TYPE maxValue_;

    int32_t blockIdx_ = 0;
    int64_t binsDet_ = 0;
    int64_t ubNumCanUseDet_ = 0;
    int64_t ubLoopNumDet_ = 0;
    int64_t formerLengthDet_ = 0;
    int64_t tailLengthDet_ = 0;

    int64_t needXCoreNum_ = 0;
    int64_t clearYFactor_ = 0;
    int64_t clearYCoreNum_ = 0;
    int64_t clearYTail_ = 0;
};

// Phase 2 SIMT kernel: per UB chunk, accumulate into UB int32 for the [ubLoop*ubNumCanUse, (ubLoop+1)*ubNumCanUse) slice
template <typename X_TYPE, typename COMPUTE_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeNotFullDetFp32(
    __gm__ X_TYPE* xGmAddr, __ubuf__ int32_t* yLocalIntAddr, const int32_t blockIdx, const int64_t needXCoreNum,
    const int64_t xIndexBase, const int64_t coreDataLength, const COMPUTE_TYPE minValue,
    const COMPUTE_TYPE maxValue, const int64_t bins, const int64_t ubLoop, const int64_t ubNumCanUse)
{
    if (blockIdx >= needXCoreNum) {
        return;
    }

    for (int32_t idxDetFp32 = static_cast<int32_t>(threadIdx.x); idxDetFp32 < coreDataLength;
         idxDetFp32 += static_cast<int32_t>(blockDim.x)) {
        int64_t xIdxDetFp32 = xIndexBase + idxDetFp32;
        COMPUTE_TYPE value = static_cast<COMPUTE_TYPE>(xGmAddr[xIdxDetFp32]);
        if (value >= minValue && value <= maxValue) {
            int64_t indexBin = static_cast<int64_t>((value - minValue) * bins / (maxValue - minValue));
            if (indexBin == bins) {
                indexBin -= 1;
            }
            if (ubLoop * ubNumCanUse <= indexBin && indexBin < (ubLoop + 1) * ubNumCanUse) {
                int64_t indexBinNormal = indexBin - ubLoop * ubNumCanUse;
                asc_atomic_add(yLocalIntAddr + indexBinNormal, static_cast<int32_t>(1));
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmCastNotFullDetFp32Out(
    __gm__ int32_t* yGmIntAddr, __gm__ float* yGmFloatAddr, const int64_t clearYIndexBase, const int64_t clearYDataLength)
{
    for (int32_t castIdxDet = static_cast<int32_t>(threadIdx.x); castIdxDet < clearYDataLength;
         castIdxDet += static_cast<int32_t>(blockDim.x)) {
        int64_t yIdxDet = clearYIndexBase + castIdxDet;
        int32_t yValue = yGmIntAddr[yIdxDet];
        yGmFloatAddr[yIdxDet] = static_cast<float>(yValue);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Init(
    GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y, const HistogramV2SimtTilingData* __restrict tilingData,
    TPipe* tPipe)
{
    this->pipe_ = tPipe;
    this->blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    this->binsDet_ = tilingData->bins;
    this->ubNumCanUseDet_ = tilingData->ubNumCanUse;
    this->ubLoopNumDet_ = tilingData->ubLoopNum;
    this->formerLengthDet_ = tilingData->formerLength;
    this->tailLengthDet_ = tilingData->tailLength;
    this->needXCoreNum_ = tilingData->needXCoreNum;
    this->clearYFactor_ = tilingData->clearYFactor;
    this->clearYCoreNum_ = tilingData->clearYCoreNum;
    this->clearYTail_ = tilingData->clearYTail;

    this->xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(x));
    this->minGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(min));
    this->maxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_TYPE*>(max));
    // Both views point to the same y GM address
    this->yGmInt_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y));
    this->yGmFloat_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    this->pipe_->InitBuffer(this->yIntQue_, 1, this->ubNumCanUseDet_ * sizeof(int32_t));
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::ProcessMinMaxValue()
{
    minValue_ = static_cast<COMPUTE_TYPE>(minGm_.GetValue(0));
    maxValue_ = static_cast<COMPUTE_TYPE>(maxGm_.GetValue(0));

    if (minValue_ == maxValue_) {
        minValue_ = minValue_ - 1;
        maxValue_ = maxValue_ + 1;
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Phase2SimtCompute(
    __gm__ X_TYPE* xGmAddr, const int64_t xIndexBase, const int64_t coreDataLength)
{
    for (int64_t ubLoop = 0; ubLoop < ubLoopNumDet_; ubLoop++) {
        int64_t yLocalNum = (ubLoop == ubLoopNumDet_ - 1) ? (binsDet_ - (ubLoopNumDet_ - 1) * ubNumCanUseDet_) : ubNumCanUseDet_;
        LocalTensor<int32_t> yLocalInt = yIntQue_.template AllocTensor<int32_t>();
        Duplicate(yLocalInt, static_cast<int32_t>(0), yLocalNum);
        yIntQue_.EnQue(yLocalInt);
        yLocalInt = yIntQue_.template DeQue<int32_t>();
        __ubuf__ int32_t* yLocalIntAddr = (__ubuf__ int32_t*)yLocalInt.GetPhyAddr();

        asc_vf_call<UbSimtComputeNotFullDetFp32<X_TYPE, COMPUTE_TYPE>>(
            dim3{THREAD_NUM, 1, 1}, xGmAddr, yLocalIntAddr, blockIdx_, needXCoreNum_, xIndexBase, coreDataLength,
            minValue_, maxValue_, binsDet_, ubLoop, ubNumCanUseDet_);

        // Wait for SIMT VF compute to finish before DMA write
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // Atomic add UB int32 result to GM (int32 atomic, no float non-determinism)
        DataCopyExtParams dataCopyExtParamsAdd{
            static_cast<uint16_t>(1), static_cast<uint32_t>(static_cast<int64_t>(yLocalNum) * sizeof(int32_t)), 0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(yGmInt_[ubLoop * ubNumCanUseDet_], yLocalInt, dataCopyExtParamsAdd);
        SetAtomicNone();
        yIntQue_.template FreeTensor<int32_t>(yLocalInt);
    }
}

template <typename X_TYPE, typename COMPUTE_TYPE>
__aicore__ inline void HistogramV2SimtNotFullLoadDetFp32Out<X_TYPE, COMPUTE_TYPE>::Process()
{
    if (blockIdx_ >= GetBlockNum()) {
        return;
    }

    ProcessMinMaxValue();

    int64_t clearYIndexBaseDet = blockIdx_ * clearYFactor_;
    int64_t clearYDataLengthDet = (blockIdx_ == clearYCoreNum_ - 1) ? clearYTail_ : clearYFactor_;

    // --------------- Phase 1: clearY cores zero the output GM (as int32) ---------------
    if (blockIdx_ < clearYCoreNum_) {
        LocalTensor<int32_t> yLocalInt = yIntQue_.template AllocTensor<int32_t>();
        Duplicate(yLocalInt, static_cast<int32_t>(0), clearYDataLengthDet);
        yIntQue_.EnQue(yLocalInt);
        yLocalInt = yIntQue_.template DeQue<int32_t>();
        DataCopyExtParams dataCopyExtParamsClear{
            static_cast<uint16_t>(1), static_cast<uint32_t>(static_cast<int64_t>(clearYDataLengthDet) * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(yGmInt_[clearYIndexBaseDet], yLocalInt, dataCopyExtParamsClear);
        yIntQue_.template FreeTensor<int32_t>(yLocalInt);
    }

#ifndef __CCE_UT_TEST__
    SyncAll();
#endif

    // --------------- Phase 2: each X-processing core UB int32 accumulate → GM int32 atomic add ---------------
    if (blockIdx_ < needXCoreNum_) {
        int64_t xIndexBase = blockIdx_ * formerLengthDet_;
        int64_t coreDataLength = (blockIdx_ == needXCoreNum_ - 1) ? tailLengthDet_ : formerLengthDet_;
        __gm__ X_TYPE* xGmAddr = (__gm__ X_TYPE*)xGm_.GetPhyAddr();
        Phase2SimtCompute(xGmAddr, xIndexBase, coreDataLength);
    }

#ifndef __CCE_UT_TEST__
    SyncAll();
#endif

    // --------------- Phase 3: clearY cores cast GM int32 slice in place to GM float ---------------
    if (blockIdx_ < clearYCoreNum_) {
        __gm__ int32_t* yGmIntAddr = (__gm__ int32_t*)yGmInt_.GetPhyAddr();
        __gm__ float* yGmFloatAddr = (__gm__ float*)yGmFloat_.GetPhyAddr();
        asc_vf_call<GmCastNotFullDetFp32Out>(
            dim3{THREAD_NUM, 1, 1}, yGmIntAddr, yGmFloatAddr, clearYIndexBaseDet, clearYDataLengthDet);
    }
}

} // namespace HistogramV2SIMT

#endif // HISTOGRAM_V2_SIMT_NOT_FULL_LOAD_DET_FP32OUT_H
