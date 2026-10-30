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
 * \file merge_sort_big_size.h
 * \brief merge_sort kernel entry
 */
#ifndef MERGE_SORT_BIG_SIZE_H
#define MERGE_SORT_BIG_SIZE_H
#include <cmath>
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/util_type_simd.h"
#include "common/merge_sort_constants.h"
#include "common/merge_more_core_base.h"

using namespace AscendC;

// Import shared constants from MergeSortConstants namespace
using MergeSortConstants::DEALING_CONCAT_NUM_ONCE;
using MergeSortConstants::DEALING_EXTRACT_NUM_ONCE;
using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;
using MergeSortConstants::MERGE_MORE_BUFFER_NUM;
using MergeSortConstants::UB_BLOCK_BYTES;
using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE>
struct MergeSortBigSize
    : public MergeMoreCoreCommon::MergeMoreCoreBase<MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>, T,
                                                    CONVERT_TYPE, IS_DESCEND, INDEX_TYPE> {
    using Base = MergeMoreCoreCommon::MergeMoreCoreBase<MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>, T,
                                                        CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>;
    friend Base;

    __aicore__ inline MergeSortBigSize() {}
    __aicore__ inline void Init(GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void InitMergeBuffers();
    __aicore__ inline void ExtractAndCopyOut();
};

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE>
__aicore__ inline void MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>::Init(
    GM_ADDR inputValue, GM_ADDR value, GM_ADDR indices, GM_ADDR workSpace, const SortRegBaseTilingData* tilingData,
    TPipe* pipe)
{
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = pipe;
    this->outputLastDimValue_ = tilingData->lastDimTileNum;
    this->numTileData_ = tilingData->numTileDataSize;
    this->frontCoreNum_ = tilingData->lastDimNeedCore;
    if (this->frontCoreNum_ == 0U) {
        return;
    }
    uint32_t sortBufferSize = 8;
    this->rowIdx_ = this->blockIdx_ / this->frontCoreNum_;
    this->rowCoreIdx_ = this->blockIdx_ % this->frontCoreNum_;
    this->rowDataOffset_ = static_cast<int64_t>(this->rowIdx_) * static_cast<int64_t>(this->outputLastDimValue_);
    // Per-row workspace stores Sort API sort-struct data. This capacity uses sortBufferSize bytes per
    // original element and UB-block byte alignment; it must cover later GetSortLen-based accesses.
    uint64_t rowWorkspaceBytes = ROUND_UP_AGLIN_UINT64(static_cast<uint64_t>(this->outputLastDimValue_) *
                                                       sortBufferSize);
    uint64_t rowWorkspaceElements = rowWorkspaceBytes / sizeof(CONVERT_TYPE);
    this->rowWorkspaceOffset_ = static_cast<int64_t>(this->rowIdx_) * static_cast<int64_t>(rowWorkspaceElements) * 2;
    this->onceMaxElements_ = tilingData->keyParams0 / DEALING_SORT_NUM_ONCE * DEALING_SORT_NUM_ONCE;

    this->inputValueGm_.SetGlobalBuffer((__gm__ T*)(inputValue));
    this->outValueGm_.SetGlobalBuffer((__gm__ T*)(value));
    this->outIndexGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    this->workspaceGm_[0].SetGlobalBuffer((__gm__ CONVERT_TYPE*)(workSpace) + this->rowWorkspaceOffset_,
                                          rowWorkspaceElements);
    this->workspaceGm_[1].SetGlobalBuffer(
        (__gm__ CONVERT_TYPE*)(workSpace) + this->rowWorkspaceOffset_ + rowWorkspaceElements, rowWorkspaceElements);

    uint32_t tailNum = this->outputLastDimValue_ - (this->frontCoreNum_ - 1) * this->numTileData_;
    uint32_t alignTile = ROUND_UP_AGLIN(tailNum);
    this->pipe_->InitBuffer(this->inputQueue_, MERGE_MORE_BUFFER_NUM, alignTile * sizeof(T));

    this->pipe_->InitBuffer(this->sortedValueUb_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortedValueIndexUb_, alignTile * sizeof(uint32_t));
    this->pipe_->InitBuffer(this->concatTempBuf_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortTempBuf_, alignTile * sortBufferSize);
    this->pipe_->InitBuffer(this->sortedValueLocalCastTbuf_, alignTile * sortBufferSize);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE>
__aicore__ inline void MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>::InitMergeBuffers()
{
    uint32_t sortBufferSize = 8;
    this->pipe_->InitBuffer(this->sortedQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    this->pipe_->InitBuffer(this->copyInQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sortBufferSize);
    this->pipe_->InitBuffer(this->castValueQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(CONVERT_TYPE));
    this->pipe_->InitBuffer(this->castIndexQueue_, MERGE_MORE_BUFFER_NUM,
                            MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(uint32_t));
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        this->pipe_->InitBuffer(this->outIndexQueue_, MERGE_MORE_BUFFER_NUM,
                                MERGE_LIST_MAX_NUM * this->onceMaxElements_ * sizeof(INDEX_TYPE));
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND, typename INDEX_TYPE>
__aicore__ inline void MergeSortBigSize<T, CONVERT_TYPE, IS_DESCEND, INDEX_TYPE>::ExtractAndCopyOut()
{
    LocalTensor<INDEX_TYPE> ubOutput2;
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        ubOutput2 = this->outIndexQueue_.template AllocTensor<INDEX_TYPE>();
    }
    LocalTensor<CONVERT_TYPE> sortTempBuffer = this->sortedQueue_.template DeQue<CONVERT_TYPE>();
    LocalTensor<CONVERT_TYPE> castValue = this->castValueQueue_.template AllocTensor<CONVERT_TYPE>();
    LocalTensor<uint32_t> castIndex = this->castIndexQueue_.template AllocTensor<uint32_t>();
    AscendC::Extract(castValue, castIndex, sortTempBuffer,
                     Ops::Base::CeilDiv(this->curLoopSortedNum_, static_cast<int64_t>(DEALING_EXTRACT_NUM_ONCE)));
    if constexpr (!IS_DESCEND) {
        this->FlipSignBit(castValue, ROUND_UP_AGLIN(this->curLoopSortedNum_));
    }
    DataCopyExtParams copyParamsValue;
    copyParamsValue.blockCount = 1;
    copyParamsValue.blockLen = this->curLoopSortedNum_ * sizeof(T);
    copyParamsValue.srcStride = 0;
    copyParamsValue.dstStride = 0;

    DataCopyExtParams copyParamsIndex;
    copyParamsIndex.blockCount = 1;
    copyParamsIndex.blockLen = this->curLoopSortedNum_ * sizeof(INDEX_TYPE);
    copyParamsIndex.srcStride = 0;
    copyParamsIndex.dstStride = 0;

    this->castValueQueue_.EnQue(castValue);
    castValue = this->castValueQueue_.template DeQue<T>();
    DataCopyPad(this->outValueGm_[this->rowDataOffset_ + this->outOffset_], castValue, copyParamsValue);
    this->castValueQueue_.FreeTensor(castValue);

    uint32_t sortedIndexAlign = ROUND_UP_AGLIN(this->curLoopSortedNum_ * sizeof(uint32_t)) / sizeof(uint32_t);
    LocalTensor<int32_t> castIndexTemp = castIndex.template ReinterpretCast<int32_t>();
    if constexpr (std::is_same<int64_t, INDEX_TYPE>::value) {
        AscendC::Cast(ubOutput2, castIndexTemp, AscendC::RoundMode::CAST_NONE, sortedIndexAlign);
        this->outIndexQueue_.EnQue(ubOutput2);
        ubOutput2 = this->outIndexQueue_.template DeQue<INDEX_TYPE>();
        DataCopyPad(this->outIndexGm_[this->rowDataOffset_ + this->outOffset_], ubOutput2, copyParamsIndex);
        this->outIndexQueue_.FreeTensor(ubOutput2);
    } else {
        this->castIndexQueue_.EnQue(castIndex);
        castIndex = this->castIndexQueue_.template DeQue<uint32_t>();
        castIndexTemp = castIndex.template ReinterpretCast<int32_t>();
        DataCopyPad(this->outIndexGm_[this->rowDataOffset_ + this->outOffset_], castIndexTemp, copyParamsIndex);
    }
    this->castIndexQueue_.FreeTensor(castIndex);
    this->sortedQueue_.FreeTensor(sortTempBuffer);
    this->outOffset_ += this->curLoopSortedNum_;
}
#endif // MERGE_SORT_BIG_SIZE_H
