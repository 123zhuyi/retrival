/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matrix_diag_big_n.h
 * \brief impl of MatrixDiagAsc  n > 128B
 */

#ifndef OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_YUB
#define OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_YUB

#include "op_kernel/platform_util.h"
#include "matrix_diag_tiling_data_struct.h"

namespace MatrixDiagAsc {
using namespace AscendC;

constexpr uint16_t VECTOR_LENGTH = Ops::Base::GetVRegSize();
constexpr uint16_t NUM_256 = 256;
constexpr uint16_t NUM_32 = 32;

template <typename T, typename U>
class MatrixDiagSliceYUb {
public:
    __aicore__ inline MatrixDiagSliceYUb(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MatrixDiagScatterLargeTilingData* tilingDataPtr,
                                TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void MatrixDiagCopyGM2Ub(int64_t xGMOffset, int64_t UbUnitElementLocal);
    __aicore__ inline void MatrixDiagComputeIdx(uint32_t UbUnitElementLocal, LocalTensor<U>& curIndexLocal);

    template <typename V>
    __aicore__ inline void GenScatterIndex(uint32_t UbUnitElementLocal, LocalTensor<U>& curIndexLocal);
    __aicore__ inline void MatrixDiagScatter(uint32_t UbUnitElementLocal, LocalTensor<U>& curIndexLocal);

    __aicore__ inline void MatrixDiagCopyOut(int64_t yGMOffset, uint32_t UbUnitElementLocal, int64_t dstStrideLocal);
    __aicore__ inline void MatrixDiagCopyOutZero(int64_t yGMOffset, int64_t ubUnitHigh, int64_t ubUnitWidth,
                                                 int64_t dstStrideLocal);

private:
    TPipe* pipe_ = nullptr;
    const MatrixDiagScatterLargeTilingData* tdPtr_ = nullptr;

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> indexTailBuf_;

    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;

    int64_t blockIdx_ = 0;
    int64_t bufferCnt_ = 2; // enable db

    int64_t actualCoreNum_ = 0;
    int64_t nSize_ = 0;
    int64_t curUbFactor_ = 0;
    int64_t curUbTailFactor_ = 0;
    int64_t curUbCount_ = 0;
    int64_t curUbTotalCount_ = 0;
    int64_t curBlockFactor_ = 0;
    int64_t curBlockTailFactor_ = 0;
    int64_t curBlockMainCount_ = 0;

    int64_t wholeBlockFactor_ = 0;
    int64_t blockStartIdx_ = 0;
    int64_t ubStartIdx_ = 0;

    LocalTensor<U> indexLocal_;
    LocalTensor<U> indexTailLocal_;
};

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::Init(GM_ADDR x, GM_ADDR y,
                                                      const MatrixDiagScatterLargeTilingData* tilingDataPtr,
                                                      TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tdPtr_ = tilingDataPtr;
    pipe_ = pipe;
    actualCoreNum_ = tdPtr_->realCoreNum;
    nSize_ = tdPtr_->nSize;
    curUbFactor_ = tdPtr_->nUbFactor;
    curUbTailFactor_ = tdPtr_->nUbTailFactor;
    curUbCount_ = tdPtr_->nUbCount;
    curBlockFactor_ = tdPtr_->blockFactor;
    curBlockTailFactor_ = tdPtr_->blockTailFactor;
    curBlockMainCount_ = tdPtr_->blockMainCount;
    wholeBlockFactor_ = blockIdx_ * curBlockFactor_;
    curUbTotalCount_ = curUbCount_ * curUbCount_;
    if (curUbTailFactor_ == 0) {
        curUbTailFactor_ = curUbFactor_;
    }
    if (blockIdx_ > (curBlockMainCount_ - 1)) {
        wholeBlockFactor_ = curBlockFactor_ * curBlockMainCount_ +
                            (blockIdx_ - curBlockMainCount_) * curBlockTailFactor_;
    }
    blockStartIdx_ = wholeBlockFactor_;
    ubStartIdx_ = wholeBlockFactor_ -
                  curUbTotalCount_ * (wholeBlockFactor_ / curUbTotalCount_); // 在n*n中的idx，从0开始
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_->InitBuffer(inQue_, bufferCnt_, curUbFactor_ * sizeof(T));
    pipe_->InitBuffer(outQue_, bufferCnt_, curUbFactor_ * curUbFactor_ * sizeof(T));
    pipe_->InitBuffer(indexBuf_, VECTOR_LENGTH);
    pipe_->InitBuffer(indexTailBuf_, VECTOR_LENGTH);
    indexLocal_ = indexBuf_.Get<U>();
    indexTailLocal_ = indexTailBuf_.Get<U>();
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::Process()
{
    if (blockIdx_ > (actualCoreNum_ - 1)) {
        return;
    }
    int64_t UbLoops = curBlockFactor_;

    if (blockIdx_ > (curBlockMainCount_ - 1)) { // 是尾核
        UbLoops = curBlockTailFactor_;
    }
    MatrixDiagComputeIdx(curUbFactor_, indexLocal_);
    uint32_t ubUnitWidthAlign = (curUbTailFactor_ + (NUM_32 / sizeof(T)) - 1) / (NUM_32 / sizeof(T)) *
                                (NUM_32 / sizeof(T));
    MatrixDiagComputeIdx(ubUnitWidthAlign, indexTailLocal_);
    for (int64_t idx_blockFactor = 0; idx_blockFactor < UbLoops; ++idx_blockFactor) {
        int64_t ubIdx = (ubStartIdx_ + idx_blockFactor) -
                        curUbTotalCount_ * ((ubStartIdx_ + idx_blockFactor) /
                                            curUbTotalCount_); // 对 curUbTotalCount_
                                                               // 取模，是因为一个核可能处理的UB块数大于curUbTotalCount_
        int64_t ubRowIdx = ubIdx / curUbCount_;
        int64_t ubColIdx = ubIdx - curUbCount_ * (ubIdx / curUbCount_);
        int64_t batchIdx = blockStartIdx_ / curUbTotalCount_;
        blockStartIdx_ = blockStartIdx_ + 1;
        int64_t xGmOffset = batchIdx * nSize_ + ubColIdx * curUbFactor_;
        int64_t yGmOffset = batchIdx * nSize_ * nSize_ + ubRowIdx * nSize_ * curUbFactor_ + ubColIdx * curUbFactor_;
        if (ubColIdx == ubRowIdx) {
            MatrixDiagCopyGM2Ub(xGmOffset, curUbFactor_);
            if (ubColIdx == curUbCount_ - 1) {
                MatrixDiagScatter(ubUnitWidthAlign, indexTailLocal_);
                MatrixDiagCopyOut(yGmOffset, curUbTailFactor_, nSize_ - curUbTailFactor_);
            } else {
                MatrixDiagScatter(curUbFactor_, indexLocal_);
                MatrixDiagCopyOut(yGmOffset, curUbFactor_, nSize_ - curUbFactor_);
            }
        } else if ((ubColIdx < curUbCount_ - 1) && (ubRowIdx < curUbCount_ - 1)) {
            MatrixDiagCopyOutZero(yGmOffset, curUbFactor_, curUbFactor_, nSize_ - curUbFactor_);
        } else if ((ubColIdx == curUbCount_ - 1) && (ubRowIdx < curUbCount_ - 1)) {
            MatrixDiagCopyOutZero(yGmOffset, curUbFactor_, curUbTailFactor_, nSize_ - curUbTailFactor_);
        } else if ((ubRowIdx == curUbCount_ - 1) && (ubColIdx < curUbCount_ - 1)) {
            MatrixDiagCopyOutZero(yGmOffset, curUbTailFactor_, curUbFactor_, nSize_ - curUbFactor_);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::MatrixDiagCopyGM2Ub(int64_t xGMOffset, int64_t UbUnitElementLocal)
{
    LocalTensor<T> UbFactorLocal = inQue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = UbUnitElementLocal * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(UbFactorLocal, inputGM_[xGMOffset], copyInParams, copyInPadParams);
    inQue_.EnQue<T>(UbFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::MatrixDiagScatter(uint32_t UbUnitElementLocal,
                                                                   LocalTensor<U>& curIndexLocal)
{
    // yUb dup 0
    LocalTensor<T> yUbFactorLocal = outQue_.AllocTensor<T>();
    Duplicate(yUbFactorLocal, static_cast<T>(0), UbUnitElementLocal * UbUnitElementLocal);
    auto yUbAddr = (__ubuf__ T*)yUbFactorLocal.GetPhyAddr();
    // index
    auto indexAddr = (__ubuf__ U*)curIndexLocal.GetPhyAddr();
    // xUb
    LocalTensor<T> UbFactorLocal = inQue_.DeQue<T>();
    auto xUbAddr = (__ubuf__ T*)UbFactorLocal.GetPhyAddr();
    uint16_t vfLen = VECTOR_LENGTH / sizeof(U);
    if constexpr (sizeof(T) == 1) {
        uint32_t xUbMain = UbUnitElementLocal > vfLen ? vfLen : UbUnitElementLocal;
        uint32_t xUbTail = UbUnitElementLocal > vfLen ? (UbUnitElementLocal - vfLen) : 0;
        uint16_t mStride = xUbMain * (UbUnitElementLocal + 1);
        __VEC_SCOPE__
        {
            Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(xUbMain);
            Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(xUbTail);
            Reg::RegTensor<U> vd1;
            Reg::RegTensor<U> indexReg;
            Reg::RegTensor<T> xReg;
            Reg::RegTensor<T> xReg0;
            Reg::RegTensor<T> xReg1;
            Reg::DataCopy(indexReg, indexAddr);
            Reg::DataCopy(xReg, xUbAddr);
            Reg::Interleave(xReg0, xReg1, xReg, xReg);
            Reg::DataCopyScatter(yUbAddr, xReg0, indexReg, p0);
            Reg::Adds(vd1, indexReg, mStride, p1);
            Reg::DataCopyScatter(yUbAddr, xReg1, vd1, p1);
        }
    } else {
        __VEC_SCOPE__
        {
            Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(UbUnitElementLocal);
            Reg::RegTensor<U> indexReg;
            Reg::RegTensor<T> xReg;
            Reg::DataCopy(indexReg, indexAddr);
            Reg::DataCopy(xReg, xUbAddr);
            Reg::DataCopyScatter(yUbAddr, xReg, indexReg, p0);
        }
    }
    outQue_.EnQue<T>(yUbFactorLocal);
    inQue_.FreeTensor(UbFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::MatrixDiagCopyOut(int64_t yGmOffset, uint32_t UbUnitElementLocal,
                                                                   int64_t dstStrideLocal)
{
    LocalTensor<T> yUbFactorLocal = outQue_.DeQue<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = UbUnitElementLocal;
    copyInParams.blockLen = UbUnitElementLocal * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = dstStrideLocal * sizeof(T);
    DataCopyPad(outputGM_[yGmOffset], yUbFactorLocal, copyInParams);
    outQue_.FreeTensor(yUbFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::MatrixDiagCopyOutZero(int64_t yGmOffset, int64_t ubUnitHigh,
                                                                       int64_t ubUnitWidth, int64_t dstStrideLocal)
{
    LocalTensor<T> yUbFactorLocal = outQue_.AllocTensor<T>();
    int32_t ubUnitWidthAlign = (ubUnitWidth + (NUM_32 / sizeof(T)) - 1) / (NUM_32 / sizeof(T)) * (NUM_32 / sizeof(T));
    Duplicate(yUbFactorLocal, static_cast<T>(0), ubUnitHigh * ubUnitWidthAlign);
    outQue_.EnQue<T>(yUbFactorLocal);
    LocalTensor<T> yUbFactorLocal1 = outQue_.DeQue<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = ubUnitHigh;
    copyInParams.blockLen = ubUnitWidth * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = dstStrideLocal * sizeof(T);
    DataCopyPad(outputGM_[yGmOffset], yUbFactorLocal1, copyInParams);
    outQue_.FreeTensor(yUbFactorLocal1);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::MatrixDiagComputeIdx(uint32_t UbUnitElementLocal,
                                                                      LocalTensor<U>& curIndexLocal)
{
    // 计算index
    if constexpr (IsSameType<U, uint32_t>::value) {
        GenScatterIndex<int32_t>(UbUnitElementLocal, curIndexLocal);
    } else if constexpr (IsSameType<U, uint16_t>::value) {
        GenScatterIndex<int16_t>(UbUnitElementLocal, curIndexLocal);
    } else if constexpr (IsSameType<U, uint64_t>::value) {
        GenScatterIndex<int64_t>(UbUnitElementLocal, curIndexLocal);
    }
}

template <typename T, typename U>
template <typename V>
__aicore__ inline void MatrixDiagSliceYUb<T, U>::GenScatterIndex(uint32_t UbUnitElementLocal,
                                                                 LocalTensor<U>& curIndexLocal)
{
    auto indexAddr = (__ubuf__ U*)curIndexLocal.GetPhyAddr();
    uint16_t vfEle = VECTOR_LENGTH / sizeof(U);

    __VEC_SCOPE__
    {
        Reg::RegTensor<U> index;
        Reg::RegTensor<V> arrageReg;
        Reg::RegTensor<U> zeroReg;
        Reg::RegTensor<U> divReg;
        Reg::RegTensor<U> mulsReg;
        Reg::RegTensor<U> mulReg;
        Reg::RegTensor<U> subReg;
        Reg::RegTensor<U> muls2Reg;
        Reg::RegTensor<U> addReg;
        uint32_t tmpNum = vfEle;
        Reg::MaskReg maskP = AscendC::Reg::UpdateMask<U>(tmpNum);
        Reg::Arange(arrageReg, 0);
        index = (Reg::RegTensor<U>&)arrageReg;
        Reg::Duplicate(zeroReg, static_cast<U>(UbUnitElementLocal), maskP);
        Reg::Div(divReg, index, zeroReg, maskP);
        Reg::Muls(mulsReg, divReg, UbUnitElementLocal * UbUnitElementLocal, maskP);
        Reg::Mul(mulReg, divReg, zeroReg, maskP);
        Reg::Sub(subReg, index, mulReg, maskP);
        Reg::Muls(muls2Reg, subReg, UbUnitElementLocal + 1, maskP);
        Reg::Add(addReg, mulsReg, muls2Reg, maskP);
        Reg::DataCopy(indexAddr, addReg, maskP);
    }
}
} // namespace MatrixDiagAsc
#endif // OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_YUB
