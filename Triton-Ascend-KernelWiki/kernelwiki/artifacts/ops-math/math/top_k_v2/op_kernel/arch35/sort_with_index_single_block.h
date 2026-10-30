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
* \file sort_with_index_single_block.h
* \brief sort_with_index singleblock mode impl
*/

#ifndef SORT_WITH_INDEX_SINGLE_BLOCK_H
#define SORT_WITH_INDEX_SINGLE_BLOCK_H

#include "../../sort_with_index/arch35/radix_sort_with_index_single_block.h"

using namespace AscendC;

template <typename XType, bool IsDescend, typename IndexType>
struct SortWithIndexSingleBlock : public RadixSortWithIndexSingleBlock<XType, IsDescend, IndexType> {
public:
    __aicore__ inline SortWithIndexSingleBlock() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex,
                                GM_ADDR workspace, const TopKV2TilingDataSimd* tilingData, TPipe* pipe);
    __aicore__ inline void Process();
};

template <typename XType, bool IsDescend, typename IndexType>
__aicore__ inline void SortWithIndexSingleBlock<XType, IsDescend, IndexType>::Init(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex, GM_ADDR workspace, const TopKV2TilingDataSimd* tilingData,
    TPipe* pipe)
{
    this->xGm_.SetGlobalBuffer((__gm__ XType*)(x));
    this->indexGm_.SetGlobalBuffer((__gm__ IndexType*)(index));
    this->yGm_.SetGlobalBuffer((__gm__ XType*)(y));
    this->sortedIndexGm_.SetGlobalBuffer((__gm__ IndexType*)(sortedIndex));

    this->numTileData_ = tilingData->numTileDataSizeForSort;
    this->lastDimRealCore_ = tilingData->lastDimNeedCoreForSort;
    this->totalDataNum_ = tilingData->lastAxisNumForSort;
    this->unsortedDimNum_ = tilingData->unsortedDimNumForSort;
    this->sortLoopTimes_ = tilingData->sortLoopTimesForSort;
    this->lastDimTileNum_  = tilingData->lastDimTileNumForSort;
    this->unsortedDimParallel_ = tilingData->unsortedDimParallelForSort;
    this->oneCoreRowNum_ = tilingData->oneCoreRowNumForSort;
    this->sortAcApiNeedTmpBufferSize_ = tilingData->sortAcApiNeedBufferSizeForSort;

    this->blockIdx_ = GetBlockIdx();

    this->pipe_ = pipe;
    this->pipe_->InitBuffer(this->inQueueX_, 1, ROUND_UP_AGLIN(this->numTileData_ * sizeof(XType)));
    this->pipe_->InitBuffer(this->inQueueIndex_, 1, ROUND_UP_AGLIN(this->numTileData_ * sizeof(IndexType)));
    this->pipe_->InitBuffer(this->yQueue_, 1, ROUND_UP_AGLIN(this->numTileData_ * sizeof(XType)));
    this->pipe_->InitBuffer(this->sortedIndexQueue_, 1, ROUND_UP_AGLIN(this->numTileData_ * sizeof(IndexType)));
    this->pipe_->InitBuffer(this->sortedShareMemTbuf_, ROUND_UP_AGLIN(this->sortAcApiNeedTmpBufferSize_)); 
}

template <typename XType, bool IsDescend, typename IndexType>
__aicore__ inline void SortWithIndexSingleBlock<XType, IsDescend, IndexType>::Process()
{
    if (GetBlockIdx() >= this->unsortedDimParallel_) {
        return;
    }

    for (uint64_t i = 0; i < this->sortLoopTimes_; i++) {
        uint64_t loopOffset = i * this->unsortedDimParallel_ * this->totalDataNum_ * this->oneCoreRowNum_;
        this->ProcessSingleBlock(this->xGm_[loopOffset], this->indexGm_[loopOffset], i);
    } 
}
#endif // RADIX_SORT_WITH_INDEX_SINGLE_BLOCK_H