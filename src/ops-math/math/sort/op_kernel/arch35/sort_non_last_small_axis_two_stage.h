/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_NON_LAST_SMALL_AXIS_TWO_STAGE_H
#define SORT_NON_LAST_SMALL_AXIS_TWO_STAGE_H

#include "sort_small_axis_two_stage.h"

namespace Sort {
using namespace AscendC;

// outputValue/outputIdx are full-tensor GM bases. Compute the complete GM offset exactly once and scatter the
// dense UB [segment, axis] results back to GM [outer, axis, inner].
template <typename T, typename UbIdxT, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::TWO_STAGE_THREAD_NUM) __aicore__
    void StoreGroupedOuterBatchSimt(uint32_t totalElems, uint32_t segmentLen, uint32_t validSegs, uint64_t outerStart,
                                    uint32_t innerSize, __ubuf__ T* inputValue, __ubuf__ UbIdxT* inputIdx,
                                    __gm__ volatile T* outputValue, __gm__ volatile OutIdxT* outputIdx)
{
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < totalElems;
         idx += SmallAxisCommon::TWO_STAGE_THREAD_NUM) {
        uint32_t axis = idx / validSegs;
        uint32_t seg = idx - axis * validSegs;
        uint32_t outer = seg / innerSize;
        uint32_t inner = seg - outer * innerSize;
        uint32_t ubOffset = seg * segmentLen + axis;
        uint64_t gmOffset = ((outerStart + outer) * segmentLen + axis) * innerSize + inner;
        outputValue[gmOffset] = inputValue[ubOffset];
        outputIdx[gmOffset] = static_cast<OutIdxT>(inputIdx[ubOffset]);
    }
}

// Host tiling guarantees batchSize == outerPerBatch * innerSize, so every grouped batch owns complete outer
// slices. Grouping changes only GM segment mapping; UB layout and sorting remain in the established two-stage
// base, while non-grouped batches use the original class.
template <typename T, typename OutIdxT, bool IsDescend>
class SortGroupedOuterSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<SortGroupedOuterSmallAxisTwoStage<T, OutIdxT, IsDescend>, T,
                                                    uint32_t, IsDescend> {
    using Base = SmallAxisCommon::SmallAxisTwoStageBase<SortGroupedOuterSmallAxisTwoStage<T, OutIdxT, IsDescend>, T,
                                                        uint32_t, IsDescend>;

public:
    __aicore__ inline SortGroupedOuterSmallAxisTwoStage() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe)
    {
        (void)workspace;
        if (tilingData == nullptr || pipe == nullptr) {
            return;
        }
        this->blockIdx_ = GetBlockIdx();
        this->blockDim_ = GetBlockNum();
        batchSize_ = tilingData->keyParams0;
        batchNum_ = tilingData->keyParams1;
        segmentLen_ = tilingData->numTileDataSize;
        maxFlatElems_ = batchSize_ * segmentLen_;
        outerSize_ = tilingData->outerSize;
        innerSize_ = tilingData->innerSize;
        outerPerBatch_ = tilingData->keyParams4;

        inputXGm_.SetGlobalBuffer((__gm__ T*)x);
        outValueGm_.SetGlobalBuffer((__gm__ T*)y);
        outIdxGm_.SetGlobalBuffer((__gm__ OutIdxT*)idx);

        if (batchSize_ == 0U || segmentLen_ == 0U || maxFlatElems_ == 0U) {
            return;
        }
        // Axis-local indices are always non-negative and sch11 is limited to small axes. Keep a dedicated uint32_t
        // UB buffer and widen only during GM scatter; this does not alias Sort tmp or any phase-local scratch.
        constexpr uint32_t kFinalIdxElemBytes = static_cast<uint32_t>(sizeof(uint32_t));
        Base::InitSortBuffers(pipe, maxFlatElems_, tilingData->tmpUbSize, tilingData->keyParams2 != 0U,
                              kFinalIdxElemBytes);
    }

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::inputValues_;
    using Base::maxFlatElems_;
    using Base::segmentLen_;

    __aicore__ inline bool IsProcessInvalid() const
    {
        return this->blockIdx_ >= this->blockDim_ || batchSize_ == 0U || segmentLen_ == 0U || outerSize_ <= 0 ||
               innerSize_ <= 0 || outerPerBatch_ == 0U ||
               static_cast<uint64_t>(outerPerBatch_) * static_cast<uint64_t>(innerSize_) != batchSize_;
    }

    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const
    {
        uint64_t outerStart = static_cast<uint64_t>(batchId) * outerPerBatch_;
        if (outerStart >= static_cast<uint64_t>(outerSize_)) {
            return 0U;
        }
        uint64_t remainingOuter = static_cast<uint64_t>(outerSize_) - outerStart;
        uint32_t validOuter = remainingOuter >= outerPerBatch_ ? outerPerBatch_ : static_cast<uint32_t>(remainingOuter);
        return validOuter * static_cast<uint32_t>(innerSize_);
    }

    __aicore__ inline void LoadGroupedOuterBatchWithNddma(uint64_t outerStart, uint32_t validOuter)
    {
        constexpr uint32_t NDDMA_DIM_NUM = 3U;
        uint32_t innerSize = static_cast<uint32_t>(innerSize_);
        uint32_t outerElems = segmentLen_ * innerSize;
        // Map contiguous GM [outer, axis, inner] slices directly to dense UB [outer, inner, axis].
        NdDmaLoopInfo<NDDMA_DIM_NUM> loopInfo{{1, innerSize, static_cast<uint64_t>(outerElems)},
                                              {segmentLen_, 1, outerElems},
                                              {innerSize, segmentLen_, validOuter},
                                              {0, 0, 0},
                                              {0, 0, 0}};
        NdDmaParams<T, NDDMA_DIM_NUM> params{loopInfo, static_cast<T>(0)};
        NdDmaDci();
        static constexpr NdDmaConfig config;
        DataCopy<T, NDDMA_DIM_NUM, config>(inputValues_, inputXGm_[outerStart * static_cast<uint64_t>(outerElems)],
                                           params);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs)
    {
        uint32_t totalElems = validSegs * segmentLen_;
        uint64_t outerStart = static_cast<uint64_t>(batchId) * outerPerBatch_;
        uint32_t validOuter = validSegs / static_cast<uint32_t>(innerSize_);
        // NDDMA strides and loop sizes are element-based and do not require an inner row to be block-aligned.
        // Keep one producer for aligned and unaligned grouped batches so the following vector sort always has an
        // explicit MTE2_V handoff.
        LoadGroupedOuterBatchWithNddma(outerStart, validOuter);
        Base::RunTwoStageSort(totalElems);
        asc_vf_call<StoreGroupedOuterBatchSimt<T, uint32_t, OutIdxT>>(
            dim3(SmallAxisCommon::TWO_STAGE_THREAD_NUM), totalElems, segmentLen_, validSegs, outerStart,
            static_cast<uint32_t>(innerSize_), (__ubuf__ T*)finalValues_.GetPhyAddr(),
            (__ubuf__ uint32_t*)finalIdx_.GetPhyAddr(), (__gm__ volatile T*)outValueGm_.GetPhyAddr(),
            (__gm__ volatile OutIdxT*)outIdxGm_.GetPhyAddr());
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OutIdxT> outIdxGm_;
    int64_t outerSize_ = 1;
    int64_t innerSize_ = 1;
    uint32_t outerPerBatch_ = 0;
};

} // namespace Sort

#endif
