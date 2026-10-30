/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kth_value_merge_intra_core.h
 * \brief Intra-core block merge sort for kth_value (fp32)
 * \details Each core independently sorts blocks and merges them via 4-way MrgSort.
 *          No cross-core synchronization needed in merge phase.
 *          Phase 3 extracts only the kth element instead of the full sorted output.
 */

#ifndef KTH_VALUE_MERGE_INTRA_CORE_H
#define KTH_VALUE_MERGE_INTRA_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kth_value_tiling_data.h"
#include "../../sort/arch35/common/merge_sort_constants.h"
#include "../../sort/arch35/common/merge_intra_core_base.h"

namespace KthValue {
using namespace AscendC;

// Import shared constants from MergeSortConstants namespace
using MergeSortConstants::DEALING_CONCAT_NUM_ONCE;
using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_INTRA_BUFFER_NUM;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::UB_BLOCK_BYTES;

/**
 * @brief Intra-core block merge sort for kth_value: independent per-core sort+merge,
 *        then extracts only the kth element from the sorted result.
 * @tparam ValueType Input data type (float)
 * @tparam IndexType Index data type (int32_t or int64_t)
 * @tparam IsDescend Sort order: true for descending, false for ascending
 */
template <typename ValueType, typename IndexType, bool IsDescend>
class KthValueMergeIntraCore
    : public MergeIntraCoreCommon::MergeIntraCoreBase<KthValueMergeIntraCore<ValueType, IndexType, IsDescend>,
                                                      ValueType, IndexType, IsDescend> {
    using Base = MergeIntraCoreCommon::MergeIntraCoreBase<KthValueMergeIntraCore<ValueType, IndexType, IsDescend>,
                                                          ValueType, IndexType, IsDescend>;
    friend Base;

public:
    __aicore__ inline KthValueMergeIntraCore() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace,
                                const KthValueTilingData* tilingData, TPipe* pipe);

private:
    // KthValue-specific members
    uint32_t kthIndex_ = 0;
    TQue<QuePosition::VECOUT, 1> kthValueQueue_, kthIdxQueue_;

    __aicore__ inline void InitPhase1Buffers();
    __aicore__ inline void InitPhase2Buffers();
    __aicore__ inline void InitPhase3Buffers();
    __aicore__ inline void ExtractAndCopyOut(int64_t batchIdx, uint32_t resultRegion);
    __aicore__ inline void ExtractAndCopyChunk(int64_t cacheBatchOffset, uint32_t cacheOffset, int64_t outputOffset,
                                               uint32_t elemProcessed, uint32_t elemCount);
};

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR indices, GM_ADDR workspace, const KthValueTilingData* tilingData, TPipe* pipe)
{
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;

    // Parse tiling data
    this->batchNum_ = tilingData->unsortedDimNum;
    this->sortAxisNum_ = tilingData->lastAxisNum;
    this->batchPerCore_ = tilingData->keyParams0;
    this->blockSortSize_ = tilingData->numTileDataSize;
    kthIndex_ = tilingData->kthIndex;
    this->extractChunkSize_ = tilingData->keyParams4;
    this->blocksPerRow_ = tilingData->lastDimTileNum;
    this->maxCoreNum_ = tilingData->lastDimNeedCore;
    this->alignNum_ = tilingData->keyParams3;
    this->maxMergeIterations_ = tilingData->keyParams5;

    if (this->blockSortSize_ == 0 || this->extractChunkSize_ == 0 || this->blocksPerRow_ == 0 ||
        this->sortAxisNum_ <= 0) {
        return;
    }

    // Set GM buffers before deriving cache sizes. This keeps KthValue init local while avoiding
    // a long identical block with SortMergeIntraCore.
    this->inputXGm_.SetGlobalBuffer((__gm__ ValueType*)x);
    this->outValueGm_.SetGlobalBuffer((__gm__ ValueType*)value);
    this->outIdxGm_.SetGlobalBuffer((__gm__ IndexType*)indices);

    // Precompute constants
    this->blockSortLen_ = AscendC::GetSortLen<ValueType>(this->blockSortSize_);
    this->batchSortLen_ = AscendC::GetSortLen<ValueType>(this->alignNum_);
    this->sortBufferSize_ = this->blockSortLen_ * sizeof(ValueType);
    this->sortRepeatTimes_ = this->blockSortSize_ / DEALING_SORT_NUM_ONCE;
    this->concatRepeatTimes_ = this->blockSortSize_ / DEALING_CONCAT_NUM_ONCE;
    this->lastBlockSize_ = static_cast<uint32_t>(this->sortAxisNum_ -
                                                 static_cast<int64_t>(this->blocksPerRow_ - 1) * this->blockSortSize_);

    // Cache stores sort struct data (8 bytes per element: index + value)
    // Each core has its own cache region, reused across batches
    // perBatchCacheLen: sort struct length for one batch (with ping-pong, in ValueType units)
    int64_t perCoreCacheLen = static_cast<int64_t>(this->batchSortLen_) * 2; // ping-pong, reused per batch

    this->cacheGm_.SetGlobalBuffer((__gm__ ValueType*)workspace +
                                   static_cast<int64_t>(this->blockIdx_) * perCoreCacheLen);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase1Buffers()
{
    this->pipe_->InitBuffer(this->inQueueX_, MERGE_INTRA_BUFFER_NUM, this->blockSortSize_ * sizeof(ValueType));
    this->pipe_->InitBuffer(this->concatTmpBuf_, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->sortTmpBuf_, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->sortedOutQueue_, MERGE_INTRA_BUFFER_NUM, this->sortBufferSize_);
    this->pipe_->InitBuffer(this->indexTmpBuf_, this->blockSortSize_ * sizeof(uint32_t));
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase2Buffers()
{
    uint32_t mergeBufferSize = MERGE_LIST_MAX_NUM * this->sortBufferSize_;
    this->pipe_->InitBuffer(this->mergeInQueue_, 1, mergeBufferSize);
    this->pipe_->InitBuffer(this->mergeOutQueue_, 1, mergeBufferSize);
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::InitPhase3Buffers()
{
    uint32_t extractInSize = AscendC::GetSortLen<ValueType>(this->extractChunkSize_) * sizeof(ValueType);
    this->pipe_->InitBuffer(this->extractInQueue_, MERGE_INTRA_BUFFER_NUM, extractInSize);
    this->pipe_->InitBuffer(this->outValueQueue_, MERGE_INTRA_BUFFER_NUM, this->extractChunkSize_ * sizeof(ValueType));
    this->pipe_->InitBuffer(this->outIdxQueue_, MERGE_INTRA_BUFFER_NUM, this->extractChunkSize_ * sizeof(uint32_t));
    this->pipe_->InitBuffer(kthValueQueue_, 1, UB_BLOCK_BYTES);
    if constexpr (IsSameType<int64_t, IndexType>::value) {
        this->pipe_->InitBuffer(kthIdxQueue_, 1, UB_BLOCK_BYTES);
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::ExtractAndCopyOut(int64_t batchIdx,
                                                                                                  uint32_t resultRegion)
{
    int64_t outputOffset = batchIdx;

    // resultRegion: 0 = Ping (offset 0), 1 = Pong (offset batchSortLen_)
    int64_t cacheBatchOffset = (resultRegion == 1) ? this->batchSortLen_ : 0;

    uint32_t elemProcessed = 0;
    uint32_t cacheOffset = 0;

    // The merged cache is full sorted [N], but KthValue only needs one element. Skip extract chunks
    // until the chunk containing kthIndex_, then write exactly one value/index pair.
    while (elemProcessed <= kthIndex_) {
        uint32_t remainElem = static_cast<uint32_t>(this->sortAxisNum_) - elemProcessed;
        uint32_t elemCount = (this->extractChunkSize_ <= remainElem) ? this->extractChunkSize_ : remainElem;
        if (elemCount == 0)
            break;

        if (kthIndex_ < elemProcessed + elemCount) {
            ExtractAndCopyChunk(cacheBatchOffset, cacheOffset, outputOffset, elemProcessed, elemCount);
            break;
        }

        elemProcessed += elemCount;
        cacheOffset += AscendC::GetSortLen<ValueType>(elemCount);
    }
}

template <typename ValueType, typename IndexType, bool IsDescend>
__aicore__ inline void KthValueMergeIntraCore<ValueType, IndexType, IsDescend>::ExtractAndCopyChunk(
    int64_t cacheBatchOffset, uint32_t cacheOffset, int64_t outputOffset, uint32_t elemProcessed, uint32_t elemCount)
{
    LocalTensor<ValueType> kthCacheLocal = this->extractInQueue_.template AllocTensor<ValueType>();
    DataCopyExtParams loadParams;
    loadParams.blockCount = 1;
    loadParams.blockLen = static_cast<uint32_t>(AscendC::GetSortLen<ValueType>(elemCount) * sizeof(ValueType));
    loadParams.srcStride = 0;
    loadParams.dstStride = 0;
    DataCopyPad(kthCacheLocal, this->cacheGm_[cacheBatchOffset + cacheOffset], loadParams, {false, 0, 0, 0});
    this->extractInQueue_.EnQue(kthCacheLocal);

    kthCacheLocal = this->extractInQueue_.template DeQue<ValueType>();
    LocalTensor<ValueType> valueLocal = this->outValueQueue_.template AllocTensor<ValueType>();
    LocalTensor<uint32_t> indexLocal = this->outIdxQueue_.template AllocTensor<uint32_t>();
    Extract(valueLocal, indexLocal, kthCacheLocal, Ops::Base::CeilDiv(elemCount, DEALING_EXTRACT_NUM_ONCE));

    // Flip back sign bit for ascending order (was flipped in SortBlockToStruct)
    if constexpr (!IsDescend) {
        Adds(valueLocal.template ReinterpretCast<int32_t>(), valueLocal.template ReinterpretCast<int32_t>(), 0x80000000,
             elemCount);
    }

    this->outValueQueue_.EnQue(valueLocal);
    this->outIdxQueue_.EnQue(indexLocal);
    this->extractInQueue_.FreeTensor(kthCacheLocal);
    valueLocal = this->outValueQueue_.template DeQue<ValueType>();
    indexLocal = this->outIdxQueue_.template DeQue<uint32_t>();

    event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    uint32_t localK = kthIndex_ - elemProcessed;
    // localK is relative to the extracted chunk, not the whole row.
    LocalTensor<ValueType> outValue = kthValueQueue_.AllocTensor<ValueType>();
    outValue.SetValue(0, valueLocal.GetValue(localK));
    LocalTensor<int32_t> indexInt32 = indexLocal.template ReinterpretCast<int32_t>();
    LocalTensor<int64_t> indexInt64 = kthIdxQueue_.AllocTensor<int64_t>();
    indexInt64.SetValue(0, static_cast<int64_t>(indexInt32.GetValue(localK)));
    kthValueQueue_.EnQue(outValue);
    kthIdxQueue_.EnQue(indexInt64);
    outValue = kthValueQueue_.DeQue<ValueType>();
    indexInt64 = kthIdxQueue_.DeQue<int64_t>();

    event_t eventIdSToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(ValueType)), 0, 0, 0};
    DataCopyPad(this->outValueGm_[outputOffset], outValue, outParams);
    if constexpr (IsSameType<int64_t, IndexType>::value) {
        outParams.blockLen = static_cast<uint32_t>(sizeof(int64_t));
        DataCopyPad(this->outIdxGm_[outputOffset], indexInt64, outParams);
    }
    event_t eventIdMte3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    kthIdxQueue_.FreeTensor(indexInt64);
    kthValueQueue_.FreeTensor(outValue);
    this->outIdxQueue_.FreeTensor(indexLocal);
    this->outValueQueue_.FreeTensor(valueLocal);
}

} // namespace KthValue

#endif // KTH_VALUE_MERGE_INTRA_CORE_H
