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
 * \file bincount_simt_not_full_load_gm.h
 * \brief The implementation of bincount by not full load in gm.
 */

#ifndef BINCOUNT_SIMT_NOT_FULL_LOAD_GM_H
#define BINCOUNT_SIMT_NOT_FULL_LOAD_GM_H

#include "bincount_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace BincountSimt {
using namespace AscendC;

// Calculatation of bincount in ub without weight.
template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmSimtCompute(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* binsGmAddr, const int64_t arrayHeadIndex,
    const int64_t arrayDataLength)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayHeadIndex + index]);
        asc_atomic_add(binsGmAddr + arrayValue, static_cast<WEIGHT_TYPE>(WEIGHT_ONE));
    }
}

// Calculatation of bincount in ub with weight.
template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GmSimtComputeWithWeight(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* weightsGmAddr, __gm__ WEIGHT_TYPE* binsGmAddr,
    const int64_t arrayHeadIndex, const int64_t arrayDataLength)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayHeadIndex + index]);
        asc_atomic_add(binsGmAddr + arrayValue, weightsGmAddr[arrayHeadIndex + index]);
    }
}

template <typename WEIGHT_TYPE>
class BincountSimtNotFullLoadGm : public BincountSimtBase<WEIGHT_TYPE> {
public:
    __aicore__ inline BincountSimtNotFullLoadGm(){};
    // Init common global tensor and tiling data
    __aicore__ inline void Init(
        GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
        TPipe* tPipe, bool isWeightEmpty);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ResetBins();
    __aicore__ inline void Compute();
};

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadGm<WEIGHT_TYPE>::Init(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
    TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(array, size, weights, bins, tilingData, tPipe, isWeightEmpty);
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadGm<WEIGHT_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    ResetBins();
    SyncAll();
    Compute();
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadGm<WEIGHT_TYPE>::ResetBins()
{
    if (this->blockIdx_ >= this->resetBinsCoreNum_) {
        return;
    }

    int64_t resetBinsHeadIndex = this->blockIdx_ * this->resetBinsLength_;
    int64_t resetBinsDataLength =
        (this->blockIdx_ == this->resetBinsCoreNum_ - 1) ? this->resetBinsTailLength_ : this->resetBinsLength_;

    InitOutput<WEIGHT_TYPE>(this->binsGm_[resetBinsHeadIndex], resetBinsDataLength, static_cast<WEIGHT_TYPE>(0));
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtNotFullLoadGm<WEIGHT_TYPE>::Compute()
{
    if (this->blockIdx_ >= this->needCoreNum_) {
        return;
    }

    int64_t arrayHeadIndex = this->blockIdx_ * this->formerLength_;
    int64_t arrayDataLength = (this->blockIdx_ == this->needCoreNum_ - 1) ? this->tailLength_ : this->formerLength_;
    __gm__ int32_t* arrayGmAddr = (__gm__ int32_t*)this->arrayGm_.GetPhyAddr();
    __gm__ WEIGHT_TYPE* binsGmAddr = (__gm__ WEIGHT_TYPE*)this->binsGm_.GetPhyAddr();
    // calculation of bincount by AtomicAdd in simt.
    if (this->isWeightEmpty_) {
        asc_vf_call<GmSimtCompute<WEIGHT_TYPE>>(
            dim3{THREAD_NUM}, arrayGmAddr, binsGmAddr, arrayHeadIndex, arrayDataLength);
    } else {
        __gm__ WEIGHT_TYPE* weightsGmAddr = (__gm__ WEIGHT_TYPE*)this->weightsGm_.GetPhyAddr();
        asc_vf_call<GmSimtComputeWithWeight<WEIGHT_TYPE>>(
            dim3{THREAD_NUM}, arrayGmAddr, weightsGmAddr, binsGmAddr, arrayHeadIndex, arrayDataLength);
    }
}

} // namespace BincountSimt
#endif