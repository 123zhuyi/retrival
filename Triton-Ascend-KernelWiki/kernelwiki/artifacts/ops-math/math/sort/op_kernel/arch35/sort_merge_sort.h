/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sort_merge_sort.h
 * \brief
 */

#ifndef MERGE_SORT_H
#define MERGE_SORT_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/merge_sort_constants.h"

namespace Sort {
using namespace AscendC;

using MergeSortConstants::XOR_OP_VALUE_FP;
using MergeSortConstants::XOR_OP_VALUE_HALF;

const uint32_t UB_AGLIN_VALUE = Ops::Base::GetUbBlockSize();
const uint32_t CONCAT_AGLIN_VALUE = 16;
// T1输入x dtype T2输出Idx dtype UT无符号的数据类型
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis = 0>
class MergeSort {
public:
    __aicore__ inline MergeSort(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortRegBaseTilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSingleBlockSort(GlobalTensor<T1> inputX, int32_t sortLoopRound);
    __aicore__ inline void ParserTilingData();
    __aicore__ inline void CopyDataIn(GlobalTensor<T1> inputX, uint64_t tileOffset, uint32_t currTileSize,
                                      uint32_t oneCoreRowNum);
    __aicore__ inline void VbsMergeSortBf16(LocalTensor<bfloat16_t> xLocal, LocalTensor<T1> sortedValueLocal,
                                            LocalTensor<uint32_t> sortedValueIndexLocal, uint32_t currTileSize,
                                            uint32_t nowCoreRealRowNum);
    __aicore__ inline void VbsMergeSort(LocalTensor<T1> xLocal, LocalTensor<T1> sortedValueLocal,
                                        LocalTensor<uint32_t> sortedValueIndexLocal, uint32_t currTileSize,
                                        uint32_t nowCoreRealRowNum);
    __aicore__ inline void CopyValue2Gm(uint64_t gmOffset, uint64_t tileOffset, uint32_t outputLastDimValue,
                                        uint32_t oneCoreRowNum);
    __aicore__ inline void flipSignBit(LocalTensor<CONVERT_TYPE> xLocal, uint32_t offsetOneRow, uint32_t aglinTileSize);

    GlobalTensor<T1> inputXGm_;
    GlobalTensor<T1> outValueGm_;
    GlobalTensor<T2> outIdxGm_;

    TPipe* pipe_;
    // 599040 5120tile
    const SortRegBaseTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TBuf<TPosition::VECCALC> tmpUb_;
    TQue<QuePosition::VECOUT, 1> outIdxQueue_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    // merg sort
    TBuf<TPosition::VECCALC> contCatTmpTbuf_;
    TBuf<TPosition::VECCALC> indeXLocalTbuf_;
    TBuf<TPosition::VECCALC> sortedTmpLocalTbuf_;
    TBuf<TPosition::VECCALC> sortedLocalResTbuf_;
    TBuf<TPosition::VECCALC> xLocalCastTbuf_;
    TBuf<TPosition::VECCALC> sortedValueLocalCastTbuf_;
    LocalTensor<uint32_t> indexLocal_;
    uint32_t blockIdx_ = 0;
    uint32_t oneCoreRowNum_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t numTileData_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t sortLoopTimes_ = 0;
    int64_t outputLastDimValue_ = 0;
    uint32_t alignSize_ = 0;
};

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::Init(
    GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace, const SortRegBaseTilingData* __restrict tilingData,
    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    ParserTilingData();
    inputXGm_.SetGlobalBuffer((__gm__ T1*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    outIdxGm_.SetGlobalBuffer((__gm__ T2*)sortIndex);

    uint32_t bufferNum = tilingData_->keyParams4;
    pipe_->InitBuffer(inQueueX_, bufferNum, tilingData_->keyParams1);
    pipe_->InitBuffer(outValueQueue_, bufferNum, tilingData_->keyParams1);
    pipe_->InitBuffer(outIdxQueue_, bufferNum, tilingData_->keyParams2);
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);

    uint32_t byteSize = 8;
    uint32_t tmpBufferSize = alignSize_ * byteSize;
    pipe_->InitBuffer(indeXLocalTbuf_, alignSize_ * sizeof(uint32_t));
    pipe_->InitBuffer(contCatTmpTbuf_, tmpUbSize_);
    pipe_->InitBuffer(sortedTmpLocalTbuf_, tmpBufferSize * sizeof(CONVERT_TYPE));
    pipe_->InitBuffer(sortedLocalResTbuf_, tmpBufferSize * sizeof(CONVERT_TYPE));
    pipe_->InitBuffer(xLocalCastTbuf_, alignSize_ * sizeof(CONVERT_TYPE) * oneCoreRowNum_);
    pipe_->InitBuffer(sortedValueLocalCastTbuf_, alignSize_ * sizeof(CONVERT_TYPE) * oneCoreRowNum_);
    indexLocal_ = indeXLocalTbuf_.AllocTensor<uint32_t>();
    // init indexLocal value
    __local_mem__ int32_t* indexValuePtr = (__ubuf__ int32_t*)indexLocal_.GetPhyAddr();
    uint32_t vfLenB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint16_t repeatTime = CeilDivision(alignSize_, vfLenB32);
    uint32_t aglinTileSizeCopy = alignSize_;
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vciTensor;
        Reg::RegTensor<int32_t> indexTensor;
        Reg::Arange(vciTensor, 0);
        for (uint16_t i = 0; i < repeatTime; i++) {
            Reg::MaskReg vciMask = Reg::UpdateMask<uint32_t>(aglinTileSizeCopy);
            Reg::Adds(indexTensor, vciTensor, i * vfLenB32, vciMask);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(indexValuePtr, indexTensor, vfLenB32, vciMask);
        }
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::ParserTilingData()
{
    oneCoreRowNum_ = tilingData_->keyParams0;
    tmpUbSize_ = tilingData_->tmpUbSize;
    numTileData_ = tilingData_->numTileDataSize;
    unsortedDimNum_ = tilingData_->unsortedDimNum;
    unsortedDimParallel_ = tilingData_->unsortedDimParallel;
    sortLoopTimes_ = tilingData_->sortLoopTimes;
    outputLastDimValue_ = tilingData_->lastAxisNum;
    alignSize_ = tilingData_->keyParams3;
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::CopyDataIn(
    GlobalTensor<T1> inputX, uint64_t tileOffset, uint32_t currTileSize, uint32_t oneCoreRowNum)
{
    LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();
    uint32_t localTensorLen = alignSize_ * oneCoreRowNum;
    T1 defaultValue = static_cast<T1>(NAN);
    if constexpr (isDescend == 1) {
        defaultValue = static_cast<T1>(-INFINITY);
    }
    Duplicate(xLocal, defaultValue, localTensorLen);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(currTileSize * sizeof(T1)) / sizeof(T1);
    uint32_t dstStride = ((alignSize_ - currTileSizeAlign) * sizeof(T1)) / UB_AGLIN_VALUE;
    DataCopyPadExtParams<T1> padParams;
    padParams.isPad = true;
    padParams.rightPadding = currTileSizeAlign - currTileSize;
    padParams.paddingValue = static_cast<T1>(defaultValue);
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = oneCoreRowNum;
    dataCopyParam.blockLen = currTileSize * sizeof(T1);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = dstStride;
    DataCopyPad(xLocal, inputX[tileOffset], dataCopyParam, padParams);
    inQueueX_.EnQue(xLocal);
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::flipSignBit(
    LocalTensor<CONVERT_TYPE> xLocal, uint32_t offsetOneRow, uint32_t aglinTileSize)
{
    if constexpr (IsSameType<float, CONVERT_TYPE>::value) {
        AscendC::LocalTensor<int32_t> castTensor = xLocal[offsetOneRow].template ReinterpretCast<int32_t>();
        AscendC::Adds(castTensor, castTensor, XOR_OP_VALUE_FP, aglinTileSize);
    } else if constexpr (IsSameType<half, CONVERT_TYPE>::value) {
        AscendC::LocalTensor<int16_t> castTensor = xLocal[offsetOneRow].template ReinterpretCast<int16_t>();
        AscendC::Adds(castTensor, castTensor, XOR_OP_VALUE_HALF, aglinTileSize);
    }
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::VbsMergeSortBf16(
    LocalTensor<bfloat16_t> xLocal, LocalTensor<T1> sortedValueLocal, LocalTensor<uint32_t> sortedValueIndexLocal,
    uint32_t currTileSize, uint32_t nowCoreRealRowNum)
{
    uint32_t aglinTileSize = alignSize_;
    uint32_t sortRepeatTimes = alignSize_ / UB_AGLIN_VALUE;
    uint32_t concatRepeatTimes = alignSize_ / CONCAT_AGLIN_VALUE;
    uint32_t extractRepeatTimes = sortRepeatTimes;

    AscendC::LocalTensor<CONVERT_TYPE> concatTmpLocal = contCatTmpTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedLocal = sortedLocalResTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortTmpLocal = sortedTmpLocalTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> xLocalCast = xLocalCastTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedValueLocalCast = sortedValueLocalCastTbuf_.Get<CONVERT_TYPE>();
    AscendC::Cast(xLocalCast, xLocal, AscendC::RoundMode::CAST_NONE, aglinTileSize * nowCoreRealRowNum);
    for (int32_t round = 0; round < nowCoreRealRowNum; round++) {
        uint32_t offsetOneRow = round * aglinTileSize;
        if constexpr (isDescend == 0) {
            flipSignBit(xLocalCast, offsetOneRow, aglinTileSize);
        }
        if constexpr (isSort32SmallAxis == 1) {
            AscendC::Sort32<CONVERT_TYPE>(sortedLocal, xLocalCast[offsetOneRow], indexLocal_, 1);
        } else {
            AscendC::LocalTensor<CONVERT_TYPE> concatLocal;
            AscendC::Concat(concatLocal, xLocalCast[offsetOneRow], concatTmpLocal, concatRepeatTimes);
            AscendC::Sort<CONVERT_TYPE, true>(sortedLocal, concatLocal, indexLocal_, sortTmpLocal, sortRepeatTimes);
        }
        // isSort32SmallAxis enables compile-time repeatTimes=1 for hardware Extract fast path
        AscendC::Extract(sortedValueLocalCast[offsetOneRow], sortedValueIndexLocal[offsetOneRow], sortedLocal,
                         isSort32SmallAxis == 1 ? 1 : extractRepeatTimes);
        if constexpr (isDescend == 0) {
            flipSignBit(sortedValueLocalCast, offsetOneRow, aglinTileSize);
        }
    }
    AscendC::Cast(sortedValueLocal, sortedValueLocalCast, AscendC::RoundMode::CAST_RINT,
                  aglinTileSize * nowCoreRealRowNum);
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::VbsMergeSort(
    LocalTensor<T1> xLocal, LocalTensor<T1> sortedValueLocal, LocalTensor<uint32_t> sortedValueIndexLocal,
    uint32_t currTileSize, uint32_t nowCoreRealRowNum)
{
    uint32_t sortRepeatTimes = alignSize_ / UB_AGLIN_VALUE;
    uint32_t concatRepeatTimes = alignSize_ / CONCAT_AGLIN_VALUE;
    uint32_t extractRepeatTimes = sortRepeatTimes;

    AscendC::LocalTensor<CONVERT_TYPE> concatTmpLocal = contCatTmpTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedLocal = sortedLocalResTbuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortTmpLocal = sortedTmpLocalTbuf_.Get<CONVERT_TYPE>();
    for (int32_t round = 0; round < nowCoreRealRowNum; round++) {
        uint32_t offsetOneRow = round * alignSize_;
        if constexpr (isDescend == 0) {
            flipSignBit(xLocal, offsetOneRow, alignSize_);
        }
        if constexpr (isSort32SmallAxis == 1) {
            AscendC::Sort32<T1>(sortedLocal, xLocal[offsetOneRow], indexLocal_, 1);
        } else {
            AscendC::LocalTensor<CONVERT_TYPE> concatLocal;
            AscendC::Concat(concatLocal, xLocal[offsetOneRow], concatTmpLocal, concatRepeatTimes);
            // sort API中，index必须是int32_t
            AscendC::Sort<CONVERT_TYPE, true>(sortedLocal, concatLocal, indexLocal_, sortTmpLocal, sortRepeatTimes);
        }
        // 处理sort后的结果数据，输出排序后的value和index
        // isSort32SmallAxis enables compile-time repeatTimes=1 for hardware Extract fast path
        AscendC::Extract(sortedValueLocal[offsetOneRow], sortedValueIndexLocal[offsetOneRow], sortedLocal,
                         isSort32SmallAxis == 1 ? 1 : extractRepeatTimes);
        if constexpr (isDescend == 0) {
            flipSignBit(sortedValueLocal, offsetOneRow, alignSize_);
        }
    }
}

template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::CopyValue2Gm(
    uint64_t gmOffset, uint64_t tileOffset, uint32_t outputLastDimValue, uint32_t oneCoreRowNum)
{
    // value stride
    uint32_t currTileSizeAlign = ROUND_UP_AGLIN(outputLastDimValue * sizeof(T1)) / sizeof(T1);
    uint32_t ubStrideValue = ((alignSize_ - currTileSizeAlign) * sizeof(T1)) / UB_AGLIN_VALUE;
    // index stride
    uint32_t currSrcTileIndexSizeAlign = ROUND_UP_AGLIN(outputLastDimValue * sizeof(int32_t)) / sizeof(int32_t);
    uint32_t ubSrcStrideIndex = ((alignSize_ - currSrcTileIndexSizeAlign) * sizeof(int32_t)) / UB_AGLIN_VALUE;

    uint32_t currTileIndexSizeAlign = ROUND_UP_AGLIN(outputLastDimValue * sizeof(T2)) / sizeof(T2);
    uint32_t ubStrideIndex = ((alignSize_ - currTileIndexSizeAlign) * sizeof(T2)) / UB_AGLIN_VALUE;
    // copy result out
    AscendC::LocalTensor<T1> outValueLocal = outValueQueue_.DeQue<T1>();
    AscendC::LocalTensor<T2> outIndexLocal = outIdxQueue_.DeQue<T2>();
    AscendC::DataCopyExtParams dataCopyParamValue{static_cast<uint16_t>(oneCoreRowNum),
                                                  static_cast<uint32_t>(outputLastDimValue * sizeof(T1)), ubStrideValue,
                                                  0, 0};
    AscendC::DataCopyPad(outValueGm_[gmOffset + tileOffset], outValueLocal, dataCopyParamValue);

    AscendC::DataCopyExtParams dataCopyParamIndex{
        static_cast<uint16_t>(oneCoreRowNum),                   // 连续数据块的个数
        static_cast<uint32_t>(outputLastDimValue * sizeof(T2)), // 每个连续传输数据块的长度，长度为Byte
        ubStrideIndex,                                          // 源操作数，相邻连续数据块的间隔
        0,                                                      // 目的操作数，相邻连续数据块的间隔
        0};
    AscendC::DataCopyPad(outIdxGm_[gmOffset + tileOffset], outIndexLocal, dataCopyParamIndex);
    outIdxQueue_.FreeTensor(outIndexLocal);
    outValueQueue_.FreeTensor(outValueLocal);
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::ProcessSingleBlockSort(
    GlobalTensor<T1> inputX, int32_t sortLoopRound)
{
    int64_t unsortedDimIndex = (blockIdx_ + sortLoopRound * unsortedDimParallel_) * oneCoreRowNum_;
    if (unsortedDimIndex >= unsortedDimNum_) {
        return;
    }
    uint32_t nowCoreRealRowNum = oneCoreRowNum_;
    int64_t remainNum = unsortedDimNum_ - unsortedDimIndex;
    if (remainNum < static_cast<int64_t>(oneCoreRowNum_)) {
        nowCoreRealRowNum = static_cast<uint32_t>(remainNum);
    }
    // offset
    uint64_t tileOffset = blockIdx_ * numTileData_ * oneCoreRowNum_;
    // copy data
    CopyDataIn(inputX, tileOffset, numTileData_, nowCoreRealRowNum);
    AscendC::LocalTensor<T1> xLocal = inQueueX_.DeQue<T1>();

    AscendC::LocalTensor<T1> sortedValueLocal = outValueQueue_.AllocTensor<T1>();
    // get buffer
    AscendC::LocalTensor<int64_t> sortedValueIndexInt64Local;
    AscendC::LocalTensor<uint32_t> sortedValueIndexLocal;
    if constexpr (IsSameType<int64_t, T2>::value) {
        sortedValueIndexInt64Local = outIdxQueue_.AllocTensor<int64_t>();
        uint32_t realNum = alignSize_ * oneCoreRowNum_;
        sortedValueIndexLocal = sortedValueIndexInt64Local.template ReinterpretCast<uint32_t>()[realNum];
    } else {
        sortedValueIndexLocal = outIdxQueue_.AllocTensor<uint32_t>();
    }
    if constexpr (IsSameType<bfloat16_t, T1>::value) {
        VbsMergeSortBf16(xLocal, sortedValueLocal, sortedValueIndexLocal, numTileData_, nowCoreRealRowNum);
    } else {
        VbsMergeSort(xLocal, sortedValueLocal, sortedValueIndexLocal, numTileData_, nowCoreRealRowNum);
    }
    if constexpr (IsSameType<int64_t, T2>::value) {
        AscendC::LocalTensor<int32_t> sortedValueIndexInt32Local = sortedValueIndexLocal
                                                                       .template ReinterpretCast<int32_t>();
        AscendC::Cast(sortedValueIndexInt64Local, sortedValueIndexInt32Local, AscendC::RoundMode::CAST_NONE,
                      nowCoreRealRowNum * alignSize_);
        outIdxQueue_.EnQue<int64_t>(sortedValueIndexInt64Local);
    } else {
        outIdxQueue_.EnQue<uint32_t>(sortedValueIndexLocal);
    }
    outValueQueue_.EnQue<T1>(sortedValueLocal);
    inQueueX_.FreeTensor(xLocal);
    // copy result out
    uint64_t gmOffset = sortLoopRound * unsortedDimParallel_ * outputLastDimValue_ * oneCoreRowNum_;
    uint64_t answerTileOffset = blockIdx_ * outputLastDimValue_ * oneCoreRowNum_;

    CopyValue2Gm(gmOffset, answerTileOffset, outputLastDimValue_, nowCoreRealRowNum);
}
template <typename T1, typename T2, typename CONVERT_TYPE, uint64_t isDescend, uint64_t isSort32SmallAxis>
__aicore__ inline void MergeSort<T1, T2, CONVERT_TYPE, isDescend, isSort32SmallAxis>::Process()
{
    if (blockIdx_ > GetBlockNum()) {
        return;
    }
    for (int32_t i = 0; i < sortLoopTimes_; i++) {
        uint64_t loopOffset = i * unsortedDimParallel_ * oneCoreRowNum_ * numTileData_;
        ProcessSingleBlockSort(inputXGm_[loopOffset], i);
    }
}
} // namespace Sort
#endif // namespace MergeSort
