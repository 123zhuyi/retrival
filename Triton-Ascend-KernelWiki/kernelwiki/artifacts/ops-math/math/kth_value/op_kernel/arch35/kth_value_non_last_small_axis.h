/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_NON_LAST_SMALL_AXIS_H
#define KTH_VALUE_NON_LAST_SMALL_AXIS_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "kth_value_tiling_data.h"
#include "../../sort/arch35/common/non_last_small_axis_base.h"

namespace KthValue {
using namespace AscendC;

template <typename T, bool IsDescend, bool UseMergeSort>
class KthValueNonLastSmallAxis
    : public SmallAxisCommon::NonLastSmallAxisBase<
          KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>, T,
          std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
          std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
          IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>> {
    using Base = SmallAxisCommon::NonLastSmallAxisBase<
        KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>, T,
        std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
        std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
        IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>>;

    using SortT = std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>;
    static constexpr bool IS_BF16_MERGE = UseMergeSort && std::is_same_v<T, bfloat16_t>;

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace,
                                const KthValueTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void StoreTile(int64_t inputOffset, int64_t outputOffset, uint32_t curInnerChunk);

private:
    __aicore__ inline void ParseTilingData();
    __aicore__ inline void CopyKthToOutput(uint32_t curInnerChunk, int64_t outputOffset);

    const KthValueTilingData* tilingData_ = nullptr;
    GlobalTensor<T> valueGm_;
    GlobalTensor<int64_t> indexGm_;

    TBuf<TPosition::VECCALC> compactValueBuf_;
    TBuf<TPosition::VECCALC> compactCastValueBuf_;
    TBuf<TPosition::VECCALC> compactIndexBuf_;

    LocalTensor<T> compactValue_;
    LocalTensor<SortT> compactCastValue_;
    LocalTensor<int64_t> compactIndex_;

    uint32_t kthIndex_ = 0;
};

template <typename T, bool IsDescend, bool UseMergeSort>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>::Init(GM_ADDR x, GM_ADDR values,
                                                                                  GM_ADDR indices, GM_ADDR workspace,
                                                                                  const KthValueTilingData* tilingData,
                                                                                  TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->pipe_ = pipe;
    tilingData_ = tilingData;
    this->blockIdx_ = GetBlockIdx();
    this->blockDim_ = GetBlockNum();
    ParseTilingData();

    this->inputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(values));
    indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(indices));

    if (this->axisLen_ == 0 || this->innerChunk_ == 0 || this->innerLoopNum_ == 0 || this->outerSize_ <= 0 ||
        this->innerSize_ <= 0) {
        return;
    }
    this->pipe_->InitBuffer(this->inputTileBuf_, this->axisLen_ * this->inputRowBytes_);
    if constexpr (IS_BF16_MERGE) {
        this->pipe_->InitBuffer(this->inputCastBuf_, this->innerChunk_ * this->inputValueAxisBytes_);
        this->inputCast_ = this->inputCastBuf_.template Get<T>();
    }
    this->pipe_->InitBuffer(this->sortInputBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedValueBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedIndexBuf_, this->innerChunk_ * this->indexAxisBytes_);
    this->pipe_->InitBuffer(compactValueBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(T)));
    if constexpr (IS_BF16_MERGE) {
        this->pipe_->InitBuffer(compactCastValueBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(SortT)));
        compactCastValue_ = compactCastValueBuf_.template Get<SortT>();
    }
    this->pipe_->InitBuffer(compactIndexBuf_, ROUND_UP_AGLIN(this->innerChunk_ * sizeof(int64_t)));
    this->pipe_->InitBuffer(this->tmpBuf_, this->tmpUbSize_);

    this->inputTile_ = this->inputTileBuf_.template Get<T>();
    this->sortInput_ = this->sortInputBuf_.template Get<SortT>();
    this->sortedValue_ = this->sortedValueBuf_.template Get<SortT>();
    this->sortedIndex_ = this->sortedIndexBuf_.template Get<uint32_t>();
    compactValue_ = compactValueBuf_.template Get<T>();
    compactIndex_ = compactIndexBuf_.template Get<int64_t>();
    this->tmp_ = this->tmpBuf_.template Get<uint8_t>();
}

template <typename T, bool IsDescend, bool UseMergeSort>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>::ParseTilingData()
{
    this->axisLen_ = static_cast<uint32_t>(tilingData_->lastAxisNum);
    kthIndex_ = static_cast<uint32_t>(tilingData_->kthIndex);
    this->outerSize_ = tilingData_->outerSize;
    this->innerSize_ = tilingData_->innerSize;
    this->innerLoopNum_ = tilingData_->innerLoopNum;
    this->innerChunk_ = tilingData_->innerChunk;
    this->inputRowBytes_ = tilingData_->inputRowBytes;
    this->valueAxisBytes_ = tilingData_->valueAxisBytes;
    this->indexAxisBytes_ = tilingData_->indexAxisBytes;
    this->inputRowElems_ = this->inputRowBytes_ / sizeof(T);
    this->valueAxisElems_ = this->valueAxisBytes_ / sizeof(SortT);
    this->indexAxisElems_ = this->indexAxisBytes_ / sizeof(uint32_t);
    this->sortCount_ = tilingData_->keyParams0 == 0U ? this->axisLen_ : tilingData_->keyParams0;
    if constexpr (IS_BF16_MERGE) {
        this->inputValueAxisBytes_ = tilingData_->keyParams1;
        this->inputValueAxisElems_ = this->inputValueAxisBytes_ / sizeof(T);
    }
    this->tmpUbSize_ = tilingData_->tmpUbSize;
}

template <typename T, bool IsDescend, bool UseMergeSort>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>::CopyKthToOutput(uint32_t curInnerChunk,
                                                                                             int64_t outputOffset)
{
    event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    uint32_t compactCastElems = 0;
    if constexpr (IS_BF16_MERGE) {
        compactCastElems = ROUND_UP_AGLIN(curInnerChunk * sizeof(SortT)) / sizeof(SortT);
        Duplicate(compactCastValue_, static_cast<SortT>(0), compactCastElems);
        event_t eventIdVToSForInit = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToSForInit);
        WaitFlag<HardEvent::V_S>(eventIdVToSForInit);
    }
    // Compact one kth element per inner lane. The GM output is [outer, 1, inner], so outputOffset
    // points to a contiguous inner chunk.
    for (uint32_t inner = 0; inner < curInnerChunk; ++inner) {
        uint32_t srcOffset = inner * this->valueAxisElems_ + kthIndex_;
        uint32_t idxOffset = inner * this->indexAxisElems_ + kthIndex_;
        if constexpr (IS_BF16_MERGE) {
            compactCastValue_.SetValue(inner, this->sortedValue_.GetValue(srcOffset));
        } else {
            compactValue_.SetValue(inner, this->sortedValue_.GetValue(srcOffset));
        }
        compactIndex_.SetValue(inner, static_cast<int64_t>(this->sortedIndex_.GetValue(idxOffset)));
    }
    if constexpr (IS_BF16_MERGE) {
        event_t eventIdSToV = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        Cast(compactValue_, compactCastValue_, RoundMode::CAST_RINT, compactCastElems);
        event_t eventIdVToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    }
    event_t eventIdSToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(curInnerChunk * sizeof(T)), 0, 0, 0};
    DataCopyPad(valueGm_[outputOffset], compactValue_, valueCopyParam);
    DataCopyExtParams indexCopyParam{1, static_cast<uint32_t>(curInnerChunk * sizeof(int64_t)), 0, 0, 0};
    DataCopyPad(indexGm_[outputOffset], compactIndex_, indexCopyParam);
    event_t eventIdMte3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename T, bool IsDescend, bool UseMergeSort>
__aicore__ inline void KthValueNonLastSmallAxis<T, IsDescend, UseMergeSort>::StoreTile(int64_t inputOffset,
                                                                                       int64_t outputOffset,
                                                                                       uint32_t curInnerChunk)
{
    (void)inputOffset;
    CopyKthToOutput(curInnerChunk, outputOffset);
}

} // namespace KthValue

#endif
