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
 * \file merge_sort_simd.h
 * \brief merge_sort kernel entry
 */
#ifndef SORT_WITH_INDEX_MERGE_SORT_SIMD_H
#define SORT_WITH_INDEX_MERGE_SORT_SIMD_H
#include "kernel_operator.h"
#include "../../sort/arch35/common/util_type_simd.h" // 引入使用 ROUND_UP_AGLIN
#include "constant_var_simd.h"
#include "sort_with_index_common.h"

namespace SortWithIndex {

using namespace AscendC;

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND>
struct MergeSortKernel {
public:
    __aicore__ inline MergeSortKernel() {}
    __aicore__ inline void InitMergeSortBuffer(uint32_t tileDataSize, uint32_t coreRowNum, uint32_t mergeSortBufSize);
    __aicore__ inline void DoMergeSort(LocalTensor<T> inputLocal, LocalTensor<T> sortedValLocal,
                                       LocalTensor<uint32_t> sortedIdxLocal, uint32_t tileDataSize,
                                       uint32_t realRowNum);
    __aicore__ inline void DoMergeSortBf16(LocalTensor<bfloat16_t> inputLocal, LocalTensor<T> sortedValLocal,
                                           LocalTensor<uint32_t> sortedIdxLocal, uint32_t tileDataSize,
                                           uint32_t realRowNum);
    __aicore__ inline void FlipSignBit(LocalTensor<CONVERT_TYPE> inputLocal, uint32_t rowOff, uint32_t alignTileSz);
    __aicore__ inline void SetPipePtr(TPipe* pipe) { pipePtr_ = pipe; }

public:
    TPipe* pipePtr_ = nullptr;
    // merg sort
    TBuf<TPosition::VECCALC> concatTmpBuf_;
    TBuf<TPosition::VECCALC> idxLocalBuf_;
    TBuf<TPosition::VECCALC> sortedTmpBuf_;
    TBuf<TPosition::VECCALC> sortedResBuf_;
    TBuf<TPosition::VECCALC> xCastBuf_;
    TBuf<TPosition::VECCALC> sortedValCastBuf_;
    LocalTensor<uint32_t> idxLocal_;
};

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND>
__aicore__ inline void MergeSortKernel<T, CONVERT_TYPE, IS_DESCEND>::InitMergeSortBuffer(uint32_t tileDataSize,
                                                                                         uint32_t coreRowNum,
                                                                                         uint32_t mergeSortBufSize)
{
    uint32_t alignTileSize = ((tileDataSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE) * UB_AGLIN_VALUE;
    uint32_t elemBytes = 8;
    uint32_t tmpBufSize = alignTileSize * elemBytes;
    pipePtr_->InitBuffer(idxLocalBuf_, ROUND_UP_AGLIN(alignTileSize * sizeof(uint32_t)));
    pipePtr_->InitBuffer(concatTmpBuf_, ROUND_UP_AGLIN(mergeSortBufSize));
    pipePtr_->InitBuffer(sortedTmpBuf_, ROUND_UP_AGLIN(tmpBufSize * sizeof(CONVERT_TYPE)));
    pipePtr_->InitBuffer(sortedResBuf_, ROUND_UP_AGLIN(tmpBufSize * sizeof(CONVERT_TYPE)));
    pipePtr_->InitBuffer(xCastBuf_, ROUND_UP_AGLIN(alignTileSize * sizeof(CONVERT_TYPE)) * coreRowNum);
    pipePtr_->InitBuffer(sortedValCastBuf_, ROUND_UP_AGLIN(alignTileSize * sizeof(CONVERT_TYPE)) * coreRowNum);
    idxLocal_ = idxLocalBuf_.AllocTensor<uint32_t>();
    // init idxLocal_ value
    __ubuf__ int32_t* idxValPtr = (__ubuf__ int32_t*)idxLocal_.GetPhyAddr();
    uint16_t repeatCnt = (alignTileSize + ONE_TIMES_B32_NUM - 1) / ONE_TIMES_B32_NUM;
    uint32_t alignTileSizeCp = alignTileSize;
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vciReg;
        Reg::RegTensor<int32_t> idxReg;
        Reg::Arange(vciReg, 0);
        for (uint16_t i = 0; i < repeatCnt; i++) {
            Reg::MaskReg vciMsk = Reg::UpdateMask<uint32_t>(alignTileSizeCp);
            Reg::Adds(idxReg, vciReg, i * ONE_TIMES_B32_NUM, vciMsk);
            Reg::StoreAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(idxValPtr, idxReg, ONE_TIMES_B32_NUM, vciMsk);
        }
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND>
__aicore__ inline void MergeSortKernel<T, CONVERT_TYPE, IS_DESCEND>::DoMergeSort(LocalTensor<T> inputLocal,
                                                                                 LocalTensor<T> sortedValLocal,
                                                                                 LocalTensor<uint32_t> sortedIdxLocal,
                                                                                 uint32_t tileDataSize,
                                                                                 uint32_t realRowNum)
{
    uint32_t alignTileSize = ((tileDataSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE) * UB_AGLIN_VALUE;
    uint32_t sortRepeatCnt = (alignTileSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE;
    uint32_t concatRepeatCnt = (alignTileSize + CONCAT_AGLIN_VALUE - 1) / CONCAT_AGLIN_VALUE;
    uint32_t extractRepeatCnt = (alignTileSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE;

    AscendC::LocalTensor<CONVERT_TYPE> concatTmp = concatTmpBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedRes = sortedResBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortTmp = sortedTmpBuf_.Get<CONVERT_TYPE>();
    for (int32_t rowRound = 0; rowRound < realRowNum; rowRound++) {
        uint32_t rowOffset = rowRound * alignTileSize;
        if constexpr (!IS_DESCEND) {
            FlipSignBit(inputLocal, rowOffset, alignTileSize);
        }
        AscendC::LocalTensor<CONVERT_TYPE> concatOut;
        AscendC::Concat(concatOut, inputLocal[rowOffset], concatTmp, concatRepeatCnt);
        // sort API中，index必须是int32_t
        AscendC::Sort<CONVERT_TYPE, true>(sortedRes, concatOut, idxLocal_, sortTmp, sortRepeatCnt);
        // 处理sort后的结果数据，输出排序后的value和index
        AscendC::Extract(sortedValLocal[rowOffset], sortedIdxLocal[rowOffset], sortedRes, extractRepeatCnt);
        if constexpr (!IS_DESCEND) {
            FlipSignBit(sortedValLocal, rowOffset, alignTileSize);
        }
    }
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND>
__aicore__ inline void MergeSortKernel<T, CONVERT_TYPE, IS_DESCEND>::DoMergeSortBf16(
    LocalTensor<bfloat16_t> inputLocal, LocalTensor<T> sortedValLocal, LocalTensor<uint32_t> sortedIdxLocal,
    uint32_t tileDataSize, uint32_t realRowNum)
{
    uint32_t alignTileSize = ((tileDataSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE) * UB_AGLIN_VALUE;
    uint32_t sortRepeatCnt = (alignTileSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE;
    uint32_t concatRepeatCnt = (alignTileSize + CONCAT_AGLIN_VALUE - 1) / CONCAT_AGLIN_VALUE;
    uint32_t extractRepeatCnt = (alignTileSize + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE;

    AscendC::LocalTensor<CONVERT_TYPE> concatTmp = concatTmpBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedRes = sortedResBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortTmp = sortedTmpBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> xCastLocal = xCastBuf_.Get<CONVERT_TYPE>();
    AscendC::LocalTensor<CONVERT_TYPE> sortedValCast = sortedValCastBuf_.Get<CONVERT_TYPE>();
    AscendC::Cast(xCastLocal, inputLocal, AscendC::RoundMode::CAST_NONE, alignTileSize * realRowNum);
    for (int32_t rowRound = 0; rowRound < realRowNum; rowRound++) {
        uint32_t rowOffset = rowRound * alignTileSize;
        if constexpr (!IS_DESCEND) {
            FlipSignBit(xCastLocal, rowOffset, alignTileSize);
        }
        AscendC::LocalTensor<CONVERT_TYPE> concatOut;
        AscendC::Concat(concatOut, xCastLocal[rowOffset], concatTmp, concatRepeatCnt);
        AscendC::Sort<CONVERT_TYPE, true>(sortedRes, concatOut, idxLocal_, sortTmp, sortRepeatCnt);
        AscendC::Extract(sortedValCast[rowOffset], sortedIdxLocal[rowOffset], sortedRes, extractRepeatCnt);
        if constexpr (!IS_DESCEND) {
            FlipSignBit(sortedValCast, rowOffset, alignTileSize);
        }
    }
    AscendC::Cast(sortedValLocal, sortedValCast, AscendC::RoundMode::CAST_RINT, alignTileSize * realRowNum);
}

template <typename T, typename CONVERT_TYPE, bool IS_DESCEND>
__aicore__ inline void MergeSortKernel<T, CONVERT_TYPE, IS_DESCEND>::FlipSignBit(LocalTensor<CONVERT_TYPE> inputLocal,
                                                                                 uint32_t rowOff, uint32_t alignTileSz)
{
    if constexpr (SortIsSame<float, CONVERT_TYPE>::value) {
        AscendC::LocalTensor<int32_t> castTns = inputLocal[rowOff].template ReinterpretCast<int32_t>();
        AscendC::Adds(castTns, castTns, XOR_OP_VALUE_FP, alignTileSz);
    } else if constexpr (SortIsSame<half, CONVERT_TYPE>::value) {
        AscendC::LocalTensor<int16_t> castTns = inputLocal[rowOff].template ReinterpretCast<int16_t>();
        AscendC::Adds(castTns, castTns, XOR_OP_VALUE_HALF, alignTileSz);
    }
}
} // namespace SortWithIndex
#endif
