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
 * \file bincount_simt_not_full_load_ub.h
 * \brief The implementation of bincount by not full load in ub.
 */

#ifndef BINCOUNT_SIMT_NOT_FULL_LOAD_UB_H
#define BINCOUNT_SIMT_NOT_FULL_LOAD_UB_H

#include "kernel_operator.h"
#include "bincount_simt_base.h"
#include "bincount_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace BincountSimt {
using namespace AscendC;

template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeLoop(
    __gm__ int32_t* arrayGmAddr, __ubuf__ WEIGHT_TYPE* yLocalAddr, const int64_t arrayAicoreOffset,
    const int64_t arrayDataLength, const int64_t ubLoop, const int64_t availableUbSize)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses Due to this aicore has 'arrayDataLength' number, loop calculate the count of array value.
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        // The value of array.
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayAicoreOffset + index]);
        // judge index in current Ub.
        int64_t currentLoopMinAddr = ubLoop * availableUbSize;
        int64_t currentLoopMaxAddr = currentLoopMinAddr + availableUbSize;
        if (arrayValue >= currentLoopMinAddr && arrayValue < currentLoopMaxAddr) {
            // calculate the count of array value by accumulation in ub.
            int64_t arrayOffset = arrayValue - currentLoopMinAddr;
            asc_atomic_add(yLocalAddr + arrayOffset, static_cast<WEIGHT_TYPE>(WEIGHT_ONE));
        }
    }
}

template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeWithWeightLoop(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* weightGmAddr, __ubuf__ WEIGHT_TYPE* yLocalAddr,
    const int64_t arrayAicoreOffset, const int64_t arrayDataLength, const int64_t ubLoop, const int64_t availableUbSize)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses Due to this aicore has 'arrayDataLength' number, loop calculate the weight of array value.
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        // The value of array.
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayAicoreOffset + index]);
        // judge index in current Ub.
        int64_t currentLoopMinAddr = ubLoop * availableUbSize;
        int64_t currentLoopMaxAddr = currentLoopMinAddr + availableUbSize;
        if (arrayValue >= currentLoopMinAddr && arrayValue < currentLoopMaxAddr) {
            // Calculate the weight of array value by accumulation in ub.
            int64_t arrayOffset = arrayValue - currentLoopMinAddr;
            asc_atomic_add(yLocalAddr + arrayOffset, weightGmAddr[arrayAicoreOffset + index]);
        }
    }
}

template <typename WEIGHT_TYPE>
class BincountSimtNotFullLoadUb : public BincountSimtBase<WEIGHT_TYPE> {
public:
    // constructor fuction
    __aicore__ inline BincountSimtNotFullLoadUb(){};
    // Init common global tensor and tiling data
    __aicore__ inline void Init(
        GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
        TPipe* tPipe, bool isWeightEmpty);
    // The calculate process of bincount
    __aicore__ inline void Process();

private:
    // Zero clearing of ouput in gm.
    __aicore__ inline void ResetBins();
    // Calculate the result of bincount.
    __aicore__ inline void Compute();
    // transfer output from ub to gm
    __aicore__ inline void CopyOut(int64_t offsetInGm, uint32_t stride);

private:
    // output in local memory.
    LocalTensor<WEIGHT_TYPE> binsLocal_;
    // queue of output.
    TPipe* tPipe_;
    TQue<TPosition::VECOUT, OUT_QUE_DEPTH> binsQue_;

    // available size of ub.
    int64_t availableUbSize_ = 0;
    // The count which can complete the input calculation in ub.
    int64_t ubLoopCount_ = 0;
};

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadUb<WEIGHT_TYPE>::Init(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
    TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(array, size, weights, bins, tilingData, tPipe, isWeightEmpty);

    availableUbSize_ = tilingData->ubNumCanUse;
    ubLoopCount_ = tilingData->ubLoopNum;

    tPipe_ = tPipe;
    tPipe_->InitBuffer(binsQue_, NUM_DOUBLE_BUFFER, availableUbSize_ * sizeof(WEIGHT_TYPE));
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadUb<WEIGHT_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    ResetBins();
    SyncAll();
    Compute();
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadUb<WEIGHT_TYPE>::ResetBins()
{
    int64_t resetBinsHeadIndex = this->blockIdx_ * this->resetBinsLength_;
    int64_t resetBinsDataLength =
        (this->blockIdx_ == this->resetBinsCoreNum_ - 1) ? this->resetBinsTailLength_ : this->resetBinsLength_;

    InitOutput<WEIGHT_TYPE>(this->binsGm_[resetBinsHeadIndex], resetBinsDataLength, static_cast<WEIGHT_TYPE>(0));
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadUb<WEIGHT_TYPE>::Compute()
{
    if (this->blockIdx_ >= this->needCoreNum_) {
        return;
    }

    int64_t arrayHeadIndex = this->blockIdx_ * this->formerLength_;
    int64_t arrayDataLength = (this->blockIdx_ == this->needCoreNum_ - 1) ? this->tailLength_ : this->formerLength_;
    __gm__ int32_t* arrayGmAddr = (__gm__ int32_t*)this->arrayGm_.GetPhyAddr();

    for (int64_t index = 0; index < this->ubLoopCount_; index++) {
        // The length of output each loop.
        int64_t binsLength = (index == this->ubLoopCount_ - 1) ?
                                 (this->size_ - (this->ubLoopCount_ - 1) * this->availableUbSize_) :
                                 this->availableUbSize_;
        // Init zero of binsLocal_
        binsLocal_ = binsQue_.template AllocTensor<WEIGHT_TYPE>();
        Duplicate<WEIGHT_TYPE>(binsLocal_, 0, binsLength);

        __ubuf__ WEIGHT_TYPE* binsLocalAddr = (__ubuf__ WEIGHT_TYPE*)binsLocal_.GetPhyAddr();
        // calculation of bincount by AtomicAdd in simt.
        if (this->isWeightEmpty_) {
            asc_vf_call<UbSimtComputeLoop<WEIGHT_TYPE>>(
                dim3{THREAD_NUM}, arrayGmAddr, binsLocalAddr, arrayHeadIndex, arrayDataLength, index,
                this->availableUbSize_);
        } else {
            __gm__ WEIGHT_TYPE* weightsGmAddr = (__gm__ WEIGHT_TYPE*)this->weightsGm_.GetPhyAddr();
            asc_vf_call<UbSimtComputeWithWeightLoop<WEIGHT_TYPE>>(
                dim3{THREAD_NUM}, arrayGmAddr, weightsGmAddr, binsLocalAddr, arrayHeadIndex, arrayDataLength,
                index, this->availableUbSize_);
        }

        // transfer ouput from ub to gm.
        CopyOut(index * this->availableUbSize_, static_cast<uint32_t>(binsLength * sizeof(WEIGHT_TYPE)));
        binsQue_.template FreeTensor<WEIGHT_TYPE>(binsLocal_);
    }
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadUb<WEIGHT_TYPE>::CopyOut(int64_t offsetInGm, uint32_t stride)
{
    binsQue_.EnQue(binsLocal_);
    binsLocal_ = binsQue_.template DeQue<WEIGHT_TYPE>();
    SetAtomicAdd<WEIGHT_TYPE>();
    DataCopyExtParams dataCopyExtParams{DATA_COPY_PAD_BLOCK_COUNT, stride, 0, 0, 0};
    DataCopyPad(this->binsGm_[offsetInGm], binsLocal_, dataCopyExtParams);
    SetAtomicNone();
}

} // namespace BincountSimt
#endif // BINCOUNT_SIMT_NOT_FULL_LOAD_UB_H