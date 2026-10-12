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
 * \file bincount_simt_full_load.h
 * \brief The implementation of bincount by full load in ub.
 */

#ifndef BINCOUNT_SIMT_FULL_LOAD_H
#define BINCOUNT_SIMT_FULL_LOAD_H

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
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtCompute(
    __gm__ int32_t* arrayGmAddr, __ubuf__ WEIGHT_TYPE* yLocalAddr, const int64_t arrayAicoreOffset,
    const int64_t arrayDataLength)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses Due to this aicore has 'arrayDataLength' number, loop calculate the count of array value.
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        // The value of array.
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayAicoreOffset + index]);
        // calculate the count of array value by accumulation in ub.
        asc_atomic_add(yLocalAddr + arrayValue, static_cast<WEIGHT_TYPE>(WEIGHT_ONE));
    }
}

template <typename WEIGHT_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void UbSimtComputeWithWeight(
    __gm__ int32_t* arrayGmAddr, __gm__ WEIGHT_TYPE* weightGmAddr, __ubuf__ WEIGHT_TYPE* yLocalAddr,
    const int64_t arrayAicoreOffset, const int64_t arrayDataLength)
{
    // threadIdx.x is commonly used as an index for data, use different threads to process data with gm at
    // different addresses Due to this aicore has 'arrayDataLength' number, loop calculate the weight of array value.
    for (int64_t index = static_cast<int64_t>(threadIdx.x); index < arrayDataLength;
         index += static_cast<int64_t>(blockDim.x)) {
        // The value of array.
        int64_t arrayValue = static_cast<int64_t>(arrayGmAddr[arrayAicoreOffset + index]);
        // Calculate the weight of array value by accumulation in ub.
        asc_atomic_add(yLocalAddr + arrayValue, weightGmAddr[arrayAicoreOffset + index]);
    }
}

template <typename WEIGHT_TYPE>
class BincountSimtFullLoad : public BincountSimtBase<WEIGHT_TYPE> {
public:
    __aicore__ inline BincountSimtFullLoad(){};
    __aicore__ inline void Init(
        GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
        TPipe* tPipe, bool isWeightEmpty);
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
};

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtFullLoad<WEIGHT_TYPE>::Init(
    GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, const BincountTilingData* __restrict tilingData,
    TPipe* tPipe, bool isWeightEmpty)
{
    this->BaseInit(array, size, weights, bins, tilingData, tPipe, isWeightEmpty);
    tPipe_ = tPipe;
    tPipe_->InitBuffer(binsQue_, NUM_DOUBLE_BUFFER, this->size_ * sizeof(WEIGHT_TYPE));
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtFullLoad<WEIGHT_TYPE>::Process()
{
    if (this->blockIdx_ >= GetBlockNum()) {
        return;
    }

    ResetBins();
    SyncAll();
    Compute();
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtFullLoad<WEIGHT_TYPE>::ResetBins()
{
    if (this->blockIdx_ >= this->resetBinsCoreNum_) {
        return;
    }

    binsLocal_ = binsQue_.template AllocTensor<WEIGHT_TYPE>();
    Duplicate<WEIGHT_TYPE>(binsLocal_, 0, this->size_);
    binsQue_.EnQue(binsLocal_);
    binsLocal_ = binsQue_.template DeQue<WEIGHT_TYPE>();

    int64_t resetBinsDataLength =
        this->blockIdx_ == this->resetBinsCoreNum_ - 1 ? this->resetBinsTailLength_ : this->resetBinsLength_;
    uint32_t resetBinsDataByteSize = static_cast<uint32_t>(resetBinsDataLength * sizeof(WEIGHT_TYPE));
    DataCopyExtParams dataCopyExtParams{DATA_COPY_PAD_BLOCK_COUNT, resetBinsDataByteSize, 0, 0, 0};
    DataCopyPad(this->binsGm_[this->blockIdx_ * this->resetBinsLength_], binsLocal_, dataCopyExtParams);
    binsQue_.template FreeTensor<WEIGHT_TYPE>(binsLocal_);
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtFullLoad<WEIGHT_TYPE>::Compute()
{
    if (this->blockIdx_ >= this->needCoreNum_) {
        return;
    }

    // Params which compute bincount by simt
    binsLocal_ = binsQue_.template AllocTensor<WEIGHT_TYPE>();
    Duplicate<WEIGHT_TYPE>(binsLocal_, 0, this->size_);
    int64_t arrayAicoreOffset = this->blockIdx_ * this->formerLength_;
    int64_t arrayDataLength = this->blockIdx_ == this->needCoreNum_ - 1 ? this->tailLength_ : this->formerLength_;
    __gm__ int32_t* arrayGmAddr = (__gm__ int32_t*)this->arrayGm_.GetPhyAddr();
    __ubuf__ WEIGHT_TYPE* yLocalAddr = (__ubuf__ WEIGHT_TYPE*)binsLocal_.GetPhyAddr();

    if (this->isWeightEmpty_) {
        asc_vf_call<UbSimtCompute<WEIGHT_TYPE>>(
            dim3{THREAD_NUM}, arrayGmAddr, yLocalAddr, arrayAicoreOffset, arrayDataLength);
    } else {
        __gm__ WEIGHT_TYPE* weightGmAddr = (__gm__ WEIGHT_TYPE*)this->weightsGm_.GetPhyAddr();
        asc_vf_call<UbSimtComputeWithWeight<WEIGHT_TYPE>>(
            dim3{THREAD_NUM}, arrayGmAddr, weightGmAddr, yLocalAddr, arrayAicoreOffset, arrayDataLength);
    }

    // Datacopy from UB to GM.
    CopyOut(0, static_cast<uint32_t>(this->size_ * sizeof(WEIGHT_TYPE)));
    binsQue_.template FreeTensor<WEIGHT_TYPE>(binsLocal_);
}

template <typename WEIGHT_TYPE>
__aicore__ inline void BincountSimtFullLoad<WEIGHT_TYPE>::CopyOut(int64_t offsetInGm, uint32_t stride)
{
    binsQue_.EnQue(binsLocal_);
    binsLocal_ = binsQue_.template DeQue<WEIGHT_TYPE>();
    SetAtomicAdd<WEIGHT_TYPE>();
    DataCopyExtParams dataCopyExtParams{DATA_COPY_PAD_BLOCK_COUNT, stride, 0, 0, 0};
    DataCopyPad(this->binsGm_[offsetInGm], binsLocal_, dataCopyExtParams);
    SetAtomicNone();
}

} // namespace BincountSimt
#endif // BINCOUNT_SIMT_FULL_LOAD_H
