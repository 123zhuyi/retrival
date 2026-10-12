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
 * \file bincount_determine.h
 * \brief bincount determine realiztion
 */

#ifndef BINCOUNT_DETERMINE_H
#define BINCOUNT_DETERMINE_H

#include "bincount_tiling_data.h"

namespace BincountSimt {
using namespace AscendC;

// Calculatation of bincount determine.
template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void DetermineSimtCompute(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* binsGmAddr, const int64_t binsHeadIndex,
    const int64_t binsDataLength, const int64_t arraySize)
{
    // Simt::GetThreadIdx() is commonly used as an index for data, use different threads to process data with gm at
    // different addresses
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < binsDataLength;
         index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
        binsGmAddr[binsHeadIndex + index] = static_cast<WEIGHT_TYPE>(0);
        for (int64_t arrayIndex = 0; arrayIndex < arraySize; arrayIndex++) {
            int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayIndex]);
            if (arrayValue == binsHeadIndex + index) {
                binsGmAddr[arrayValue] += static_cast<WEIGHT_TYPE>(WEIGHT_ONE);
            }
        }
    }
}

// Calculatation of bincount determine.
template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void DetermineSimtComputeWithWeight(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* weightsGmAddr, __gm__ WEIGHT_TYPE* binsGmAddr,
    const int64_t binsHeadIndex, const int64_t binsDataLength, const int64_t arraySize)
{
    // Simt::GetThreadIdx() is commonly used as an index for data, use different threads to process data with gm at
    // different addresses
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < binsDataLength;
         index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
        binsGmAddr[binsHeadIndex + index] = static_cast<WEIGHT_TYPE>(0);
        for (int64_t arrayIndex = 0; arrayIndex < arraySize; arrayIndex++) {
            int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayIndex]);
            if (arrayValue == binsHeadIndex + index) {
                binsGmAddr[arrayValue] += weightsGmAddr[arrayIndex];
            }
        }
    }
}

template <typename WEIGHT_TYPE>
class BincountDetermine : public BincountSimtBase<WEIGHT_TYPE> {
public:
    __aicore__ inline BincountDetermine(){};
    __aicore__ inline void Init(
        GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
        TPipe* tPipe, bool isWeightEmpty);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Compute();

private:
    int64_t binsFormerLength_ = 0;
    int64_t needBinsCoreNum_ = 0;
    int64_t binsTailLength_ = 0;
    int64_t arraySize_ = 0;
};

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountDetermine<WEIGHT_TYPE>::Init(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
    TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(array, size, weights, bins, tilingData, tPipe, isWeightEmpty);
    binsFormerLength_ = tilingData->binsFormerLength;
    needBinsCoreNum_ = tilingData->needBinsCoreNum;
    binsTailLength_ = tilingData->binsTailLength;
    arraySize_ = tilingData->arraySize;
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountDetermine<WEIGHT_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    if (this->blockIdx_ < needBinsCoreNum_) {
        Compute();
    }
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountDetermine<WEIGHT_TYPE>::Compute()
{
    __gm__ int32_t* arrayGmAddr = (__gm__ int32_t*)this->arrayGm_.GetPhyAddr();
    __gm__ WEIGHT_TYPE* binsGmAddr = (__gm__ WEIGHT_TYPE*)this->binsGm_.GetPhyAddr();
    int64_t binsHeadIndex = this->blockIdx_ * binsFormerLength_;
    int64_t binsDataLength = (this->blockIdx_ == needBinsCoreNum_ - 1) ? binsTailLength_ : binsFormerLength_;

    if (this->isWeightEmpty_) {
        Simt::VF_CALL<DetermineSimtCompute<WEIGHT_TYPE>>(
            Simt::Dim3{THREAD_NUM}, arrayGmAddr, binsGmAddr, binsHeadIndex, binsDataLength, arraySize_);
    } else {
        __gm__ WEIGHT_TYPE* weightsGmAddr = (__gm__ WEIGHT_TYPE*)this->weightsGm_.GetPhyAddr();
        Simt::VF_CALL<DetermineSimtComputeWithWeight<WEIGHT_TYPE>>(
            Simt::Dim3{THREAD_NUM}, arrayGmAddr, weightsGmAddr, binsGmAddr, binsHeadIndex, binsDataLength, arraySize_);
    }
}

} // namespace BincountSimt
#endif // BINCOUNT_DETERMINE_H