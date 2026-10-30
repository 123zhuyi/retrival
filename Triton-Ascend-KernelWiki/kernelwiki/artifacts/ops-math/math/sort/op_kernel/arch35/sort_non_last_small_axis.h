/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_NON_LAST_SMALL_AXIS_H
#define SORT_NON_LAST_SMALL_AXIS_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "sort_tiling_data.h"
#include "common/util_type_simd.h"
#include "common/non_last_small_axis_base.h"

namespace Sort {
using namespace AscendC;

template <typename T, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) __aicore__
    void StoreSingleInnerOutput(uint32_t axisLen, uint32_t threadNum, uint64_t baseOffset, uint64_t innerSize,
                                __ubuf__ T* sortedValue, __ubuf__ uint32_t* sortedIndex, __gm__ volatile T* outputValue,
                                __gm__ volatile OutIdxT* outputIndex)
{
    for (uint32_t axis = static_cast<uint32_t>(threadIdx.x); axis < axisLen; axis += threadNum) {
        uint64_t gmOffset = baseOffset + static_cast<uint64_t>(axis) * innerSize;
        outputValue[gmOffset] = sortedValue[axis];
        outputIndex[gmOffset] = static_cast<OutIdxT>(sortedIndex[axis]);
    }
}

template <typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) __aicore__
    void BuildOutputIndexTile(uint32_t axisLen, uint32_t innerChunk, uint32_t threadNum, uint32_t indexAxisElems,
                              uint32_t outputIndexRowElems, __ubuf__ uint32_t* sortedIndex,
                              __ubuf__ OutIdxT* outputIndex)
{
    uint32_t total = axisLen * innerChunk;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += threadNum) {
        uint32_t inner = idx / axisLen;
        uint32_t axis = idx - inner * axisLen;
        outputIndex[axis * outputIndexRowElems + inner] = static_cast<OutIdxT>(
            sortedIndex[inner * indexAxisElems + axis]);
    }
}

template <typename T, typename SortT, typename OutIdxT>
__simt_vf__ LAUNCH_BOUND(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM) __aicore__
    void StoreConvertedOutputTile(uint32_t axisLen, uint32_t innerChunk, uint32_t threadNum, uint32_t valueAxisElems,
                                  uint32_t indexAxisElems, uint64_t baseOffset, uint64_t innerSize,
                                  __ubuf__ SortT* sortedValue, __ubuf__ uint32_t* sortedIndex,
                                  __gm__ volatile T* outputValue, __gm__ volatile OutIdxT* outputIndex)
{
    uint32_t total = axisLen * innerChunk;
    for (uint32_t idx = static_cast<uint32_t>(threadIdx.x); idx < total; idx += threadNum) {
        uint32_t axis = idx / innerChunk;
        uint32_t inner = idx - axis * innerChunk;
        uint64_t gmOffset = baseOffset + static_cast<uint64_t>(axis) * innerSize + inner;
        outputValue[gmOffset] = static_cast<T>(sortedValue[inner * valueAxisElems + axis]);
        outputIndex[gmOffset] = static_cast<OutIdxT>(sortedIndex[inner * indexAxisElems + axis]);
    }
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
class SortNonLastSmallAxis
    : public SmallAxisCommon::NonLastSmallAxisBase<
          SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>, T,
          std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
          std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
          std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
          IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>> {
    using Base = SmallAxisCommon::NonLastSmallAxisBase<
        SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>, T,
        std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>,
        std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>,
        std::conditional_t<sizeof(T) == 1, std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>,
        IsDescend, UseMergeSort, UseMergeSort && std::is_same_v<T, bfloat16_t>>;

public:
    using SortT_ = typename std::conditional_t<UseMergeSort && std::is_same_v<T, bfloat16_t>, float, T>;
    using RangeType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), int16_t, int32_t>;
    using IdxType_ = std::conditional_t<sizeof(T) <= sizeof(int16_t), uint16_t, uint32_t>;
    using CastType_ = std::conditional_t<sizeof(T) == 1,
                                         std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, int16_t>, T>;
    static constexpr bool IsBf16Merge_ = UseMergeSort && std::is_same_v<T, bfloat16_t>;

    __aicore__ inline SortNonLastSmallAxis() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void ParseTilingData();
    __aicore__ inline void BuildOutputs(uint32_t curInnerChunk);
    __aicore__ inline void GatherOutputValues(uint32_t curInnerChunk);
    __aicore__ inline void StoreTileOutput(int64_t baseOffset, uint32_t curInnerChunk);
    __aicore__ inline void StoreSingleInnerTile(int64_t baseOffset);
    __aicore__ inline void StoreConvertedTile(int64_t baseOffset, uint32_t curInnerChunk);
    __aicore__ inline void StoreTile(int64_t inputOffset, int64_t outputOffset, uint32_t curInnerChunk);

private:
    const SortRegBaseTilingData* tilingData_ = nullptr;

    GlobalTensor<T> outValueGm_;
    GlobalTensor<OutIdxT> outIndexGm_;

    TBuf<TPosition::VECCALC> outputIndexBuf_;

    LocalTensor<OutIdxT> outputIndex_;

    uint32_t outputIndexRowBytes_ = 0;
    uint32_t outputIndexRowElems_ = 0;
};

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace, const SortRegBaseTilingData* tilingData, TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    this->tilingData_ = tilingData;
    this->pipe_ = pipe;
    this->blockIdx_ = GetBlockIdx();
    this->blockDim_ = GetBlockNum();
    ParseTilingData();

    this->inputGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIndexGm_.SetGlobalBuffer((__gm__ OutIdxT*)idx);

    if (this->axisLen_ == 0 || this->innerChunk_ == 0 || this->innerLoopNum_ == 0 || this->outerSize_ <= 0 ||
        this->innerSize_ <= 0) {
        return;
    }

    this->pipe_->InitBuffer(this->inputTileBuf_, this->axisLen_ * this->inputRowBytes_);
    if constexpr (IsBf16Merge_) {
        this->pipe_->InitBuffer(this->inputCastBuf_, this->innerChunk_ * this->inputValueAxisBytes_);
        this->inputCast_ = this->inputCastBuf_.template Get<T>();
    }
    this->pipe_->InitBuffer(this->sortInputBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedValueBuf_, this->innerChunk_ * this->valueAxisBytes_);
    this->pipe_->InitBuffer(this->sortedIndexBuf_, this->innerChunk_ * this->indexAxisBytes_);
    // When innerChunk_ == 1: StoreSingleInnerTile writes sorted indices directly
    // to GM via SIMT lanes, no UB transpose buffer needed.
    // When innerChunk_ > 1: BuildOutputs + StoreTileOutput need a UB buffer
    // to hold the transposed index tile before strided DataCopy to GM.
    if (this->innerChunk_ > 1) {
        this->pipe_->InitBuffer(this->outputIndexBuf_, this->axisLen_ * outputIndexRowBytes_);
        outputIndex_ = this->outputIndexBuf_.template Get<OutIdxT>();
    }
    this->pipe_->InitBuffer(this->tmpBuf_, this->tmpUbSize_);

    this->inputTile_ = this->inputTileBuf_.template Get<T>();
    this->sortInput_ = this->sortInputBuf_.template Get<SortT_>();
    this->sortedValue_ = this->sortedValueBuf_.template Get<SortT_>();
    this->sortedIndex_ = this->sortedIndexBuf_.template Get<uint32_t>();
    this->tmp_ = this->tmpBuf_.template Get<uint8_t>();
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::ParseTilingData()
{
    this->axisLen_ = static_cast<uint32_t>(this->tilingData_->lastAxisNum);
    this->outerSize_ = this->tilingData_->outerSize;
    this->innerSize_ = this->tilingData_->innerSize;
    this->innerLoopNum_ = this->tilingData_->innerLoopNum;
    this->innerChunk_ = this->tilingData_->innerChunk;
    this->inputRowBytes_ = this->tilingData_->inputRowBytes;
    this->valueAxisBytes_ = this->tilingData_->valueAxisBytes;
    this->indexAxisBytes_ = this->tilingData_->indexAxisBytes;
    outputIndexRowBytes_ = this->tilingData_->outputIndexRowBytes;
    this->inputRowElems_ = this->inputRowBytes_ / sizeof(T);
    this->valueAxisElems_ = this->valueAxisBytes_ / sizeof(SortT_);
    this->indexAxisElems_ = this->indexAxisBytes_ / sizeof(uint32_t);
    outputIndexRowElems_ = outputIndexRowBytes_ / sizeof(OutIdxT);
    this->sortCount_ = this->axisLen_;
    if constexpr (UseMergeSort) {
        this->sortCount_ = Ops::Base::CeilAlign(this->axisLen_, SmallAxisCommon::NON_LAST_MERGE_SORT_ALIGN);
    }
    if constexpr (IsBf16Merge_) {
        // Host allocates a bf16 staging row only for merge BF16, where SortT_ is fp32.
        this->inputValueAxisBytes_ = ROUND_UP_AGLIN(this->sortCount_ * sizeof(T));
        this->inputValueAxisElems_ = this->inputValueAxisBytes_ / sizeof(T);
    }
    this->tmpUbSize_ = this->tilingData_->tmpUbSize;
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::BuildOutputs(uint32_t curInnerChunk)
{
    GatherOutputValues(curInnerChunk);
    asc_vf_call<BuildOutputIndexTile<OutIdxT>>(
        dim3(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM), this->axisLen_, curInnerChunk,
        SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM, this->indexAxisElems_, outputIndexRowElems_,
        (__ubuf__ uint32_t*)this->sortedIndex_.GetPhyAddr(), (__ubuf__ OutIdxT*)outputIndex_.GetPhyAddr());
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::GatherOutputValues(
    uint32_t curInnerChunk)
{
    __ubuf__ T* sortedValueAddr = (__ubuf__ T*)this->sortedValue_.GetPhyAddr();
    __ubuf__ T* outputValueAddr = (__ubuf__ T*)this->inputTile_.GetPhyAddr();

    // Reuse inputTile_ as the output value tile after sorting. This transpose
    // restores [inner, axis] sorted rows back to [axis, inner] copy-out rows.
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<CastType_> valueReg;
        AscendC::Reg::RegTensor<RangeType_> valueIdxReg;
        AscendC::Reg::MaskReg valueMask = AscendC::Reg::UpdateMask<CastType_>(curInnerChunk);
        AscendC::Reg::MaskReg valueIdxMask = AscendC::Reg::CreateMask<RangeType_, AscendC::Reg::MaskPattern::ALL>();

        AscendC::Reg::Arange(valueIdxReg, 0);
        AscendC::Reg::Muls(valueIdxReg, valueIdxReg, static_cast<RangeType_>(this->valueAxisElems_), valueIdxMask);
        for (uint16_t axis = 0; axis < this->axisLen_; ++axis) {
            AscendC::Reg::DataCopyGather(valueReg, sortedValueAddr + axis,
                                         (AscendC::Reg::RegTensor<IdxType_>&)valueIdxReg, valueMask);
            if constexpr (sizeof(T) != 1) {
                AscendC::Reg::DataCopy(outputValueAddr + axis * this->inputRowElems_, valueReg, valueMask);
            } else {
                __local_mem__ CastType_* outputValueAddrB16 = reinterpret_cast<__local_mem__ CastType_*>(
                    outputValueAddr + axis * this->inputRowElems_);
                AscendC::Reg::DataCopy<CastType_, AscendC::Reg::StoreDist::DIST_PACK_B16>(outputValueAddrB16, valueReg,
                                                                                          valueMask);
            }
        }
    }
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::StoreTileOutput(
    int64_t baseOffset, uint32_t curInnerChunk)
{
    // Sync: ensure all prior VEC writes (gather transpose) are visible to MTE3
    event_t eventIdVToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    uint32_t valueBytes = curInnerChunk * sizeof(T);
    uint32_t valueAlignedBytes = ROUND_UP_AGLIN(valueBytes);
    uint32_t valueSrcStride = (this->inputRowBytes_ > valueAlignedBytes) ?
                                  (this->inputRowBytes_ - valueAlignedBytes) / UB_BLOCK_SIZE :
                                  0;
    int64_t valueDstStride = (this->innerSize_ - static_cast<int64_t>(curInnerChunk)) * static_cast<int64_t>(sizeof(T));
    // Store the restored [axisLen, curInnerChunk] tile back into the original
    // non-last-axis GM layout without materializing a full transposed tensor.
    DataCopyExtParams valueCopyParam{static_cast<uint16_t>(this->axisLen_), valueBytes, valueSrcStride, valueDstStride,
                                     0};
    DataCopyPad(outValueGm_[baseOffset], this->inputTile_, valueCopyParam);

    uint32_t indexBytes = curInnerChunk * sizeof(OutIdxT);
    uint32_t indexAlignedBytes = ROUND_UP_AGLIN(indexBytes);
    uint32_t indexSrcStride = (outputIndexRowBytes_ > indexAlignedBytes) ?
                                  (outputIndexRowBytes_ - indexAlignedBytes) / UB_BLOCK_SIZE :
                                  0;
    int64_t indexDstStride = (this->innerSize_ - static_cast<int64_t>(curInnerChunk)) *
                             static_cast<int64_t>(sizeof(OutIdxT));
    DataCopyExtParams indexCopyParam{static_cast<uint16_t>(this->axisLen_), indexBytes, indexSrcStride, indexDstStride,
                                     0};
    DataCopyPad(outIndexGm_[baseOffset], outputIndex_, indexCopyParam);

    // Sync: ensure MTE3 GM writes complete before the next tile starts
    event_t eventIdMte3ToMte2 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    // Sync: ensure MTE3 → Scalar visibility for pipe teardown
    event_t eventIdMte3ToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::StoreSingleInnerTile(
    int64_t baseOffset)
{
    // Sync: ensure all prior VEC writes (sort results) are visible to MTE3
    event_t eventIdVToMte3 = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    // For a single inner element, sorted rows are already contiguous by axis and can
    // be written directly by SIMT lanes, avoiding the output tile transpose.
    asc_vf_call<StoreSingleInnerOutput<T, OutIdxT>>(
        dim3(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM), this->axisLen_,
        SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM, static_cast<uint64_t>(baseOffset),
        static_cast<uint64_t>(this->innerSize_), (__ubuf__ T*)this->sortedValue_.GetPhyAddr(),
        (__ubuf__ uint32_t*)this->sortedIndex_.GetPhyAddr(), (__gm__ volatile T*)outValueGm_.GetPhyAddr(),
        (__gm__ volatile OutIdxT*)outIndexGm_.GetPhyAddr());
    // Sync: ensure SIMT (VEC) GM writes complete before the next tile starts
    event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::StoreConvertedTile(
    int64_t baseOffset, uint32_t curInnerChunk)
{
    // BF16 merge keeps sorted values in fp32. Convert each selected value to BF16
    // during SIMT writeback so no extra BF16 output tile is needed.
    asc_vf_call<StoreConvertedOutputTile<T, SortT_, OutIdxT>>(
        dim3(SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM), this->axisLen_, curInnerChunk,
        SmallAxisCommon::NON_LAST_TRANSPOSE_THREAD_NUM, this->valueAxisElems_, this->indexAxisElems_,
        static_cast<uint64_t>(baseOffset), static_cast<uint64_t>(this->innerSize_),
        (__ubuf__ SortT_*)this->sortedValue_.GetPhyAddr(), (__ubuf__ uint32_t*)this->sortedIndex_.GetPhyAddr(),
        (__gm__ volatile T*)outValueGm_.GetPhyAddr(), (__gm__ volatile OutIdxT*)outIndexGm_.GetPhyAddr());
    event_t eventIdVToS = static_cast<event_t>(this->pipe_->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

template <typename T, typename OutIdxT, bool IsDescend, bool UseMergeSort>
__aicore__ inline void SortNonLastSmallAxis<T, OutIdxT, IsDescend, UseMergeSort>::StoreTile(int64_t inputOffset,
                                                                                            int64_t outputOffset,
                                                                                            uint32_t curInnerChunk)
{
    (void)outputOffset;
    if constexpr (IsBf16Merge_) {
        StoreConvertedTile(inputOffset, curInnerChunk);
        return;
    }
    if (curInnerChunk == 1) {
        StoreSingleInnerTile(inputOffset);
        return;
    }
    BuildOutputs(curInnerChunk);
    StoreTileOutput(inputOffset, curInnerChunk);
}

} // namespace Sort

#endif
