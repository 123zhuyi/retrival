/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file split_row_and_col.h
 * \brief
 */
#ifndef SPLIT_ROW_AND_COL
#define SPLIT_ROW_AND_COL

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace Triangulator {
using namespace AscendC;
using namespace Ops::Base;

constexpr uint32_t NORMAL_BUFFER_NUM = 2;

constexpr uint8_t UPPER_RIGHT = 0;
constexpr uint8_t DIGIT_ONE = 1;
constexpr uint8_t ON_LINE = 1;
constexpr uint8_t LOWER_LEFT = 2;
constexpr uint8_t COPY_ZERO = 0;
constexpr uint8_t COPY_INPUT = 1;
constexpr uint8_t COPY_NORMAL = 2;

struct NormalBlockInfo {
    int64_t rowActual;
    int64_t colActual;
    int64_t gmOffset;
    uint8_t position;
    int64_t headRows;
    int64_t tailRows;
    uint32_t stride;
};

template <typename T, const bool IS_LOWER>
class SplitRowAndCol {
public:
    __aicore__ inline SplitRowAndCol(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorNormalTilingData* tilingDataPtr,
                                TPipe* pipeIn);
    __aicore__ inline void ParseTilingData(__tiling_data_ptr__ TriangulatorNormalTilingData* tilingDataPtr);
    __aicore__ inline void Process();
    __aicore__ inline void DoCopyVector(NormalBlockInfo& info);
    __aicore__ inline void CopyIn(NormalBlockInfo& info);
    __aicore__ inline void CopyOut(NormalBlockInfo& info);

    __aicore__ inline void ComputeBlockInfo(int64_t blockIndex, NormalBlockInfo& info);
    __aicore__ inline void DupByRowsTril();
    __aicore__ inline void DupByRowsTriu();
    __aicore__ inline void InQueToOutQue();

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, NORMAL_BUFFER_NUM> inQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, NORMAL_BUFFER_NUM> inOutQueue_;
    TQue<QuePosition::VECOUT, NORMAL_BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> zeroBuf_;

    GlobalTensor<T> dstGm_;
    GlobalTensor<T> srcGm_;

    int64_t row_;
    int64_t col_;
    int64_t diagOffset_;
    int64_t operatorType_;

    int64_t rowInner_;
    int64_t colInner_;
    int64_t rowOuter_;
    int64_t colOuter_;
    int64_t rowTail_;
    int64_t colTail_;
    int64_t baseBlockNum_;
    int64_t planeArea_;

    int64_t bufferSize_;
    int64_t usedCoreNum_;

    int64_t normalCoreProcessNum_;
    int64_t tailCoreProcessNum_;

    uint32_t blockIdx_;
    uint32_t regCount_;
};

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::ParseTilingData(
    __tiling_data_ptr__ TriangulatorNormalTilingData* tilingDataPtr)
{
    usedCoreNum_ = tilingDataPtr->usedCoreNum;
    normalCoreProcessNum_ = tilingDataPtr->normalCoreProcessNum;
    tailCoreProcessNum_ = tilingDataPtr->tailCoreProcessNum;
    bufferSize_ = tilingDataPtr->bufferSize;
    diagOffset_ = tilingDataPtr->diagOffset;
    row_ = tilingDataPtr->row;
    col_ = tilingDataPtr->col;
    rowInner_ = tilingDataPtr->rowInner;
    colInner_ = tilingDataPtr->colInner;
    rowOuter_ = CeilDiv(row_, rowInner_);
    colOuter_ = CeilDiv(col_, colInner_);
    rowTail_ = row_ - (rowOuter_ - 1) * rowInner_;
    colTail_ = col_ - (colOuter_ - 1) * colInner_;
    baseBlockNum_ = rowOuter_ * colOuter_;
    planeArea_ = row_ * col_;
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::Init(
    GM_ADDR x, GM_ADDR y, __tiling_data_ptr__ TriangulatorNormalTilingData* tilingDataPtr, TPipe* pipeIn)
{
    // tiling_data
    ParseTilingData(tilingDataPtr);
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    constexpr uint32_t dataCount = GetVRegSize() / sizeof(T);
    constexpr uint32_t blockCount = GetUbBlockSize() / sizeof(T);
    regCount_ = colInner_ <= dataCount ? CeilAlign(static_cast<uint32_t>(colInner_), blockCount) : dataCount;

    srcGm_.SetGlobalBuffer((__gm__ T*)x);
    dstGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_ = pipeIn;
    pipe_->InitBuffer(inQueue_, NORMAL_BUFFER_NUM, bufferSize_);
    pipe_->InitBuffer(outQueue_, NORMAL_BUFFER_NUM, bufferSize_);
    pipe_->InitBuffer(inOutQueue_, NORMAL_BUFFER_NUM, bufferSize_);
    pipe_->InitBuffer(zeroBuf_, bufferSize_);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    LocalTensor<T> zero = zeroBuf_.Get<T>();
    Duplicate(zero, T(0), bufferSize_ / sizeof(T));
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    int64_t startBlockOffset = blockIdx_ < tailCoreProcessNum_ ?
                                   blockIdx_ * (normalCoreProcessNum_ + 1) :
                                   tailCoreProcessNum_ + blockIdx_ * normalCoreProcessNum_;
    int64_t curBlocks = blockIdx_ < tailCoreProcessNum_ ? normalCoreProcessNum_ + 1 : normalCoreProcessNum_;
    int64_t endBlockOffset = startBlockOffset + curBlocks;
    NormalBlockInfo curBlockInfo;
    NormalBlockInfo nextBlockInfo;
    ComputeBlockInfo(startBlockOffset, curBlockInfo);
    CopyIn(curBlockInfo);
    for (int64_t baseBlockIndex = startBlockOffset + 1; baseBlockIndex < endBlockOffset; baseBlockIndex++) {
        ComputeBlockInfo(baseBlockIndex, nextBlockInfo);
        CopyIn(nextBlockInfo);
        DoCopyVector(curBlockInfo);
        CopyOut(curBlockInfo);
        curBlockInfo = nextBlockInfo;
    }
    DoCopyVector(curBlockInfo);
    CopyOut(curBlockInfo);
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::ComputeBlockInfo(int64_t blockIndex, NormalBlockInfo& info)
{
    int64_t highAxisIndex = blockIndex / baseBlockNum_;
    int64_t highAxisTail = blockIndex % baseBlockNum_;
    int64_t rowIndex = highAxisTail / colOuter_;
    int64_t colIndex = highAxisTail % colOuter_;

    info.rowActual = rowIndex == rowOuter_ - 1 ? rowTail_ : rowInner_;
    info.colActual = colIndex == colOuter_ - 1 ? colTail_ : colInner_;

    info.headRows = colIndex * colInner_ - (rowIndex * rowInner_ + diagOffset_);
    info.tailRows = info.rowActual - info.colActual - info.headRows;
    info.position = ON_LINE;
    if (info.headRows >= info.rowActual) {
        info.position = UPPER_RIGHT;
    } else if (info.tailRows >= info.rowActual) {
        info.position = LOWER_LEFT;
    }

    info.gmOffset = highAxisIndex * planeArea_ + rowIndex * rowInner_ * col_ + colIndex * colInner_;
    info.stride = colInner_ * sizeof(T) / GetUbBlockSize() -
                  CeilDiv(static_cast<uint32_t>(info.colActual * sizeof(T)), static_cast<uint32_t>(GetUbBlockSize()));
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::CopyIn(NormalBlockInfo& info)
{
    if (info.position == UPPER_RIGHT && IS_LOWER || info.position == LOWER_LEFT && !IS_LOWER) {
        return;
    }

    DataCopyExtParams copyInParam = {static_cast<uint16_t>(info.rowActual),
                                     static_cast<uint32_t>(info.colActual * sizeof(T)),
                                     static_cast<uint32_t>((col_ - info.colActual) * sizeof(T)),
                                     static_cast<uint32_t>(info.stride), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> copyInPadParam = {false, 0, 0, 0};
    if (info.position == ON_LINE) {
        LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
        inQueue_.EnQue(xLocal);
    } else {
        LocalTensor<T> xLocal = inOutQueue_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, srcGm_[info.gmOffset], copyInParam, copyInPadParam);
        inOutQueue_.EnQue(xLocal);
    }
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::DoCopyVector(NormalBlockInfo& info)
{
    if (info.position == ON_LINE) {
        LocalTensor<T> xLocal = inQueue_.DeQue<T>();
        LocalTensor<T> yLocal = outQueue_.AllocTensor<T>();
        int64_t copyCols = (info.headRows > 0 ? 1 : -info.headRows + 1) - (IS_LOWER ? 0 : 1);
        int64_t headRows = info.headRows > 0 ? info.headRows : 0;
        int64_t tailRows = info.tailRows > 0 ? info.tailRows : 0;
        int64_t midRows = info.rowActual - tailRows - headRows;
        midRows = midRows > 0 ? midRows : 0;
        uint16_t vfInnerLoops = CeilDiv(static_cast<uint32_t>(colInner_), regCount_);
        uint16_t headLoops = headRows * vfInnerLoops;
        uint16_t tailLoops = tailRows * vfInnerLoops;
        uint16_t midRowsU16 = static_cast<uint16_t>(midRows);

        __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
        __local_mem__ T* yAddr = (__local_mem__ T*)yLocal.GetPhyAddr();

        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<T> vregZero;
            AscendC::Reg::RegTensor<T> vregSrc;
            AscendC::Reg::RegTensor<T> vregDst;
            AscendC::Reg::Duplicate(vregZero, 0);
            uint32_t count = regCount_;
            AscendC::Reg::MaskReg rowMask = AscendC::Reg::UpdateMask<T>(count);
            AscendC::Reg::MaskReg selectMask;
            for (uint16_t i = 0; i < headLoops; ++i) {
                if constexpr (IS_LOWER) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregZero, regCount_, rowMask);
                } else {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregSrc, regCount_, rowMask);
                }
            }
            if constexpr (IS_LOWER) {
                xAddr += headLoops * regCount_;
            }
            for (uint16_t i = 0; i < midRowsU16; ++i) {
                uint32_t copyEleNum = copyCols + i;
                for (uint16_t j = 0; j < vfInnerLoops; ++j) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    selectMask = AscendC::Reg::UpdateMask<T>(copyEleNum);
                    if constexpr (IS_LOWER) {
                        AscendC::Reg::Select(vregDst, vregSrc, vregZero, selectMask);
                        AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregDst, regCount_,
                                                                                      rowMask);
                    } else {
                        AscendC::Reg::Select(vregDst, vregZero, vregSrc, selectMask);
                        AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregDst, regCount_,
                                                                                      rowMask);
                    }
                }
            }
            for (uint16_t i = 0; i < tailLoops; ++i) {
                if constexpr (IS_LOWER) {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, xAddr, regCount_);
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregSrc, regCount_, rowMask);
                } else {
                    AscendC::Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(yAddr, vregZero, regCount_, rowMask);
                }
            }
            if constexpr (!IS_LOWER) {
                xAddr += tailLoops * regCount_;
            }
        }
        inQueue_.FreeTensor(xLocal);
        outQueue_.EnQue(yLocal);
    }
}

template <typename T, const bool IS_LOWER>
__aicore__ inline void SplitRowAndCol<T, IS_LOWER>::CopyOut(NormalBlockInfo& info)
{
    DataCopyExtParams copyOutParam = {
        static_cast<uint16_t>(info.rowActual), static_cast<uint32_t>(info.colActual * sizeof(T)),
        static_cast<uint32_t>(info.stride), static_cast<uint32_t>((col_ - info.colActual) * sizeof(T)),
        static_cast<uint32_t>(0)};
    if (info.position == ON_LINE) {
        LocalTensor<T> yLocal = outQueue_.DeQue<T>();
        AscendC::DataCopyPad(dstGm_[info.gmOffset], yLocal, copyOutParam);
        outQueue_.FreeTensor(yLocal);
    } else if (info.position == UPPER_RIGHT && IS_LOWER || info.position == LOWER_LEFT && !IS_LOWER) {
        LocalTensor<T> yLocal = zeroBuf_.Get<T>();
        AscendC::DataCopyPad(dstGm_[info.gmOffset], yLocal, copyOutParam);
    } else {
        LocalTensor<T> yLocal = inOutQueue_.DeQue<T>();
        AscendC::DataCopyPad(dstGm_[info.gmOffset], yLocal, copyOutParam);
        inOutQueue_.FreeTensor(yLocal);
    }
}
} // namespace Triangulator

#endif // SPLIT_ROW_AND_COL
