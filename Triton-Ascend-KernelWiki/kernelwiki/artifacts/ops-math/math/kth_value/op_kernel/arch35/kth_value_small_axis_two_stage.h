/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_SMALL_AXIS_TWO_STAGE_H
#define KTH_VALUE_SMALL_AXIS_TWO_STAGE_H

#include <type_traits>

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_tiling_data.h"
#include "../../sort/arch35/common/small_axis_two_stage_base.h"

namespace KthValue {
using namespace AscendC;

template <typename T>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void SimtStoreKthTwoStageBatch(uint32_t validSegs, uint32_t segmentLen, uint32_t kthIndex, uint64_t outputStart,
                                   __ubuf__ T* finalValues, __ubuf__ uint32_t* finalIdx, __gm__ volatile T* outputValue,
                                   __gm__ volatile int64_t* outputIndex)
{
    for (uint32_t seg = static_cast<uint32_t>(threadIdx.x); seg < validSegs;
         seg += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t srcOffset = seg * segmentLen + kthIndex;
        outputValue[outputStart + seg] = finalValues[srcOffset];
        outputIndex[outputStart + seg] = static_cast<int64_t>(finalIdx[srcOffset]);
    }
}

template <typename T>
class KthValueSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<KthValueSmallAxisTwoStage<T>, T, uint32_t, false> {
    using Base = SmallAxisCommon::SmallAxisTwoStageBase<KthValueSmallAxisTwoStage<T>, T, uint32_t, false>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, const KthValueTilingData* tiling,
                                TPipe* pipe);

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::maxFlatElems_;
    using Base::pipe_;
    using Base::segmentLen_;

    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline bool IsNonLastMode() const;
    __aicore__ inline int64_t GetOutputStart(uint32_t batchId) const;
    __aicore__ inline int64_t GetInputStart(uint32_t batchId) const;
    __aicore__ inline void LoadBatch(uint32_t batchId, uint32_t validSegs, uint32_t totalElems);
    __aicore__ inline void StoreKth(int64_t segStart, uint32_t validSegs);

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int64_t> indexGm_;

    const KthValueTilingData* tiling_ = nullptr;

    uint32_t kthIndex_ = 0;
    int64_t totalSegs_ = 0;
    int64_t innerSize_ = 1;
    uint32_t innerLoopNum_ = 0;
};

template <typename T>
__aicore__ inline void KthValueSmallAxisTwoStage<T>::Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
                                                          const KthValueTilingData* tiling, TPipe* pipe)
{
    if (tiling == nullptr || pipe == nullptr) {
        return;
    }
    pipe_ = pipe;
    tiling_ = tiling;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tiling_->keyParams0;
    batchNum_ = tiling_->keyParams1;
    segmentLen_ = tiling_->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    kthIndex_ = tiling_->kthIndex;
    totalSegs_ = tiling_->unsortedDimNum;
    innerSize_ = tiling_->innerSize <= 0 ? 1 : tiling_->innerSize;
    innerLoopNum_ = tiling_->innerLoopNum;

    inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (batchSize_ == 0U || segmentLen_ == 0U || maxFlatElems_ == 0U) {
        return;
    }

    Base::InitSortBuffers(pipe, maxFlatElems_, tiling_->tmpUbSize, tiling_->keyParams2 != 0U, sizeof(uint32_t));
}

template <typename T>
__aicore__ inline bool KthValueSmallAxisTwoStage<T>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || batchSize_ == 0U || segmentLen_ == 0U;
}

template <typename T>
__aicore__ inline uint32_t KthValueSmallAxisTwoStage<T>::ComputeValidSegs(uint32_t batchId) const
{
    if (IsNonLastMode()) {
        uint32_t innerTileId = batchId % innerLoopNum_;
        int64_t innerStart = static_cast<int64_t>(innerTileId) * static_cast<int64_t>(batchSize_);
        int64_t remain = innerSize_ - innerStart;
        if (remain <= 0) {
            return 0;
        }
        return remain >= static_cast<int64_t>(batchSize_) ? batchSize_ : static_cast<uint32_t>(remain);
    }
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T>
__aicore__ inline void KthValueSmallAxisTwoStage<T>::ProcessBatch(uint32_t batchId, uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    int64_t segStart = GetOutputStart(batchId);
    LoadBatch(batchId, validSegs, totalElems);
    Base::RunTwoStageSort(totalElems);
    StoreKth(segStart, validSegs);
}

template <typename T>
__aicore__ inline bool KthValueSmallAxisTwoStage<T>::IsNonLastMode() const
{
    return innerLoopNum_ != 0U;
}

template <typename T>
__aicore__ inline int64_t KthValueSmallAxisTwoStage<T>::GetOutputStart(uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    return outerId * innerSize_ + innerTileId * static_cast<int64_t>(batchSize_);
}

template <typename T>
__aicore__ inline int64_t KthValueSmallAxisTwoStage<T>::GetInputStart(uint32_t batchId) const
{
    if (!IsNonLastMode()) {
        return static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_) * static_cast<int64_t>(segmentLen_);
    }
    int64_t outerId = static_cast<int64_t>(batchId / innerLoopNum_);
    int64_t innerTileId = static_cast<int64_t>(batchId % innerLoopNum_);
    int64_t innerStart = innerTileId * static_cast<int64_t>(batchSize_);
    return outerId * static_cast<int64_t>(segmentLen_) * innerSize_ + innerStart;
}

template <typename T>
__aicore__ inline void KthValueSmallAxisTwoStage<T>::LoadBatch(uint32_t batchId, uint32_t validSegs,
                                                               uint32_t totalElems)
{
    if (IsNonLastMode()) {
        uint64_t outerId = static_cast<uint64_t>(batchId / innerLoopNum_);
        uint64_t innerTileId = static_cast<uint64_t>(batchId % innerLoopNum_);
        uint64_t innerStart = innerTileId * static_cast<uint64_t>(batchSize_);
        uint64_t outerBaseOffset = outerId * static_cast<uint64_t>(segmentLen_) * static_cast<uint64_t>(innerSize_);
        Base::LoadNonLastBatch(inputGm_, outerBaseOffset, innerStart, static_cast<uint64_t>(innerSize_), validSegs,
                               totalElems);
    } else {
        Base::LoadContiguousBatch(inputGm_, GetInputStart(batchId), totalElems);
    }
}

template <typename T>
__aicore__ inline void KthValueSmallAxisTwoStage<T>::StoreKth(int64_t segStart, uint32_t validSegs)
{
    event_t eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    asc_vf_call<SimtStoreKthTwoStageBatch<T>>(
        dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), validSegs, segmentLen_, kthIndex_, static_cast<uint64_t>(segStart),
        (__ubuf__ T*)finalValues_.GetPhyAddr(), (__ubuf__ uint32_t*)finalIdx_.GetPhyAddr(),
        (__gm__ volatile T*)valueGm_.GetPhyAddr(), (__gm__ volatile int64_t*)indexGm_.GetPhyAddr());
    eventIdVToS = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

} // namespace KthValue

#endif
