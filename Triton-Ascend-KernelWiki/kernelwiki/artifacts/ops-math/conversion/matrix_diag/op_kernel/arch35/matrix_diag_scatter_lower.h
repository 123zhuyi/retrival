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
 * \file matrix_diag_scatter_lower.h
 * \brief matrix_diag_scatter
 */

#ifndef MATRIX_DIAG_SCATTER_LOWER_H
#define MATRIX_DIAG_SCATTER_LOWER_H

#include "kernel_operator.h"
#include "matrix_diag_tiling_data_struct.h"
#include "op_kernel/platform_util.h"

namespace MatrixDiagAsc {
using namespace AscendC;

template <typename T, typename U, typename Y>
class MatrixDiagScatterLower {
public:
    __aicore__ inline MatrixDiagScatterLower(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MatrixDiagScatterTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPerCore(uint64_t ubLoopNum);
    __aicore__ inline void ProcessPerCoreInt8(uint64_t ubLoopNum);
    __aicore__ inline void CopyIn(uint64_t i, int64_t totalNum);
    __aicore__ inline void CopyOut(uint64_t i, int64_t totalNum);
    __aicore__ inline void genIndex();
    __aicore__ inline void VFProcess(uint32_t maskNum0, uint32_t processNum, uint32_t totalNum, U addrOffset,
                                     uint16_t loopNum, __local_mem__ U* idxPtr, __local_mem__ T* srcPtr,
                                     __local_mem__ T* dstPtr);
    __aicore__ inline void VFProcessInt8(uint32_t maskNumInt8, uint32_t processNum, uint32_t totalNum, U addrOffset,
                                         uint16_t loopNum, __local_mem__ U* idxPtr, __local_mem__ T* srcPtr,
                                         __local_mem__ T* dstPtr);
    __aicore__ inline void MainProcess(uint64_t loop, uint32_t maskNum, uint32_t processNum, uint32_t totalNum,
                                       U addrOffset, uint16_t loopNum, __local_mem__ U* idxPtr);

private:
    constexpr static int32_t BUFFER_NUM = 2;
    int64_t blockIdx_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECCALC, 1> indexBuf_;
    GlobalTensor<T> xGM_;
    GlobalTensor<T> yGM_;
    // tiling params
    const MatrixDiagScatterTilingData* tiling_;
    int32_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(T);
    int64_t xGmOffset_ = 0;
    int64_t yGmOffset_ = 0;
    int64_t blockNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t nSize_ = 0;
    int64_t batchUbFactor_ = 0;
    int64_t batchNum_ = 0;
    int64_t blockMainCount_ = 0;
    int64_t blockTailFactor_ = 0;
    int64_t realCoreNum_ = 0;
    int64_t batchUbTailFactor_ = 0;
    // Datacopy params
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
};

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::Init(GM_ADDR x, GM_ADDR y,
                                                             const MatrixDiagScatterTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tilingData;
    nSize_ = tiling_->nSize;
    batchUbFactor_ = tiling_->batchUbFactor;
    blockMainCount_ = tiling_->blockMainCount;
    blockTailFactor_ = tiling_->blockTailFactor;
    realCoreNum_ = tiling_->realCoreNum;
    batchUbTailFactor_ = tiling_->batchUbTailFactor;
    blockFactor_ = tiling_->blockFactor;
    xGM_.SetGlobalBuffer((__gm__ T*)x);
    yGM_.SetGlobalBuffer((__gm__ T*)y);
    pipe->InitBuffer(xQueue_, BUFFER_NUM, batchUbFactor_ * nSize_ * sizeof(T));
    pipe->InitBuffer(yQueue_, BUFFER_NUM, batchUbFactor_ * nSize_ * nSize_ * sizeof(T));
    pipe->InitBuffer(indexBuf_, 1, Ops::Base::GetVRegSize());
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::genIndex()
{
    uint32_t num = vlLen_;
    LocalTensor<Y> indexLocal = indexBuf_.AllocTensor<Y>();
    __local_mem__ U* dst = (__local_mem__ U*)indexLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        Reg::RegTensor<Y> indexReg;
        Reg::RegTensor<U> niReg;
        Reg::RegTensor<U> tmp;
        Reg::RegTensor<U> tmp1;
        Reg::RegTensor<U> tmp2;
        Reg::RegTensor<U> subReg;
        Reg::RegTensor<U> addReg;
        Reg::RegTensor<U> tmp3;
        Reg::MaskReg mask;
        mask = Reg::UpdateMask<U>(num);
        Reg::Arange(indexReg, 0);
        Reg::Duplicate(niReg, U(nSize_));
        Reg::Div(tmp, (Reg::RegTensor<U>&)indexReg, niReg, mask);
        Reg::Muls(tmp1, tmp, (U)(nSize_ * nSize_), mask);
        Reg::Mul(subReg, tmp, niReg, mask);
        Reg::Sub(tmp2, (Reg::RegTensor<U>&)indexReg, subReg, mask);
        Reg::Add(addReg, tmp1, tmp2, mask);
        Reg::Muls(tmp3, tmp2, U(nSize_), mask);
        Reg::Add(addReg, addReg, tmp3, mask);
        Reg::DataCopy(dst, addReg, mask);
    }
    indexBuf_.EnQue(indexLocal);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::CopyIn(uint64_t i, int64_t totalNum)
{
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = static_cast<uint32_t>(totalNum * sizeof(T)); // unit Byte
    copyInParams_.srcStride = 0;                                          // unit Byte
    copyInParams_.dstStride = 0;                                          // unit block(32byte)
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    int64_t curUbInFactor = batchUbFactor_ * nSize_;
    int64_t blockOffsetX_ = blockIdx_ < blockMainCount_ ?
                                blockIdx_ * blockFactor_ * curUbInFactor :
                                blockMainCount_ * blockFactor_ * curUbInFactor +
                                    (blockIdx_ - blockMainCount_) * blockTailFactor_ * curUbInFactor;
    xGmOffset_ = blockOffsetX_ + i * curUbInFactor;
    DataCopyPad(xLocal, xGM_[xGmOffset_], copyInParams_, padParams_);
    xQueue_.EnQue(xLocal);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::CopyOut(uint64_t i, int64_t totalNum)
{
    copyOutParams_.blockCount = 1;
    copyOutParams_.blockLen = static_cast<uint32_t>(totalNum * nSize_ * sizeof(T)); // unit Byte
    copyOutParams_.srcStride = 0;                                                   // unit Byte
    copyOutParams_.dstStride = 0;                                                   // unit block(32byte)
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    int64_t curUbOutFactor = batchUbFactor_ * nSize_ * nSize_;
    int64_t blockOffsetY_ = blockIdx_ < blockMainCount_ ?
                                blockIdx_ * blockFactor_ * curUbOutFactor :
                                blockMainCount_ * blockFactor_ * curUbOutFactor +
                                    (blockIdx_ - blockMainCount_) * blockTailFactor_ * curUbOutFactor;
    yGmOffset_ = blockOffsetY_ + i * curUbOutFactor;
    DataCopyPad(yGM_[yGmOffset_], yLocal, copyOutParams_);
    yQueue_.FreeTensor(yLocal);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::Process()
{
    uint64_t ubLoopNum;
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    if (blockIdx_ < blockMainCount_) {
        ubLoopNum = blockFactor_;
    } else {
        ubLoopNum = blockTailFactor_;
    }
    genIndex();
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        ProcessPerCoreInt8(ubLoopNum);
    } else {
        ProcessPerCore(ubLoopNum);
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::MainProcess(uint64_t loop, uint32_t maskNum,
                                                                    uint32_t processNum, uint32_t totalNum,
                                                                    U addrOffset, uint16_t loopNum,
                                                                    __local_mem__ U* idxPtr)
{
    CopyIn(loop, totalNum);
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    Duplicate<T>(yLocal, (T)0, totalNum * nSize_);
    auto* srcPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    auto* dstPtr = (__local_mem__ T*)yLocal.GetPhyAddr();
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        VFProcessInt8(maskNum, processNum, totalNum, addrOffset, loopNum, idxPtr, srcPtr, dstPtr);
    } else {
        VFProcess(maskNum, processNum, totalNum, addrOffset, loopNum, idxPtr, srcPtr, dstPtr);
    }
    xQueue_.FreeTensor(xLocal);
    yQueue_.EnQue(yLocal);
    CopyOut(loop, totalNum);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::ProcessPerCore(uint64_t ubLoopNum)
{
    batchNum_ = vlLen_ / nSize_;
    uint32_t totalNum = batchUbFactor_ * nSize_;
    uint32_t processNum = totalNum <= vlLen_ ? totalNum : batchNum_ * nSize_;
    uint16_t loopNum = totalNum / processNum;
    LocalTensor<Y> indexLocal = indexBuf_.DeQue<Y>();
    auto* idxPtr = (__local_mem__ U*)indexLocal.GetPhyAddr();
    U addrOffset = processNum * nSize_;
    uint32_t maskNum0 = processNum;
    for (uint64_t i = 0; i < ubLoopNum; i++) {
        // 判断尾块
        if (blockIdx_ == realCoreNum_ - 1 && i == ubLoopNum - 1) {
            totalNum = batchUbTailFactor_ * nSize_;
            loopNum = totalNum / processNum;
        }
        MainProcess(i, maskNum0, processNum, totalNum, addrOffset, loopNum, idxPtr);
    }
    indexBuf_.FreeTensor(indexLocal);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::VFProcess(uint32_t maskNum0, uint32_t processNum,
                                                                  uint32_t totalNum, U addrOffset, uint16_t loopNum,
                                                                  __local_mem__ U* idxPtr, __local_mem__ T* srcPtr,
                                                                  __local_mem__ T* dstPtr)
{
    uint32_t tailNum = totalNum - processNum * loopNum;
    uint32_t maskTailNum = tailNum;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcReg;
        Reg::RegTensor<T> zeroReg;
        Reg::RegTensor<U> indexReg;
        Reg::RegTensor<T> lowerReg;
        Reg::RegTensor<T> higherReg;
        Reg::UnalignReg u0;
        Reg::MaskReg mask0;
        Reg::MaskReg mask1;
        mask0 = Reg::UpdateMask<T>(maskNum0);
        mask1 = Reg::UpdateMask<T>(maskTailNum);
        Reg::DataCopy(indexReg, idxPtr);
        for (uint16_t j = 0; j < loopNum; j++) {
            Reg::DataCopyUnAlignPre(u0, srcPtr);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(srcReg, u0, srcPtr, processNum);
            Reg::DataCopyScatter(dstPtr, srcReg, indexReg, mask0);
            Reg::Adds(indexReg, indexReg, addrOffset, mask0);
        }
        if (tailNum != 0) {
            Reg::DataCopyUnAlignPre(u0, srcPtr);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(srcReg, u0, srcPtr, processNum);
            Reg::DataCopyScatter(dstPtr, srcReg, indexReg, mask1);
        }
    }
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::ProcessPerCoreInt8(uint64_t ubLoopNum)
{
    uint32_t totalNum = batchUbFactor_ * nSize_;
    int32_t halfVlLen = vlLen_ / 2;
    batchNum_ = halfVlLen / nSize_;
    uint32_t processNum = totalNum <= halfVlLen ? totalNum : batchNum_ * nSize_;
    uint16_t loopNum = totalNum / processNum;
    LocalTensor<Y> indexLocal = indexBuf_.DeQue<Y>();
    auto* idxPtr = (__local_mem__ U*)indexLocal.GetPhyAddr();
    uint32_t maskNumInt8 = processNum * 2;
    U addrOffset = processNum * nSize_;
    uint32_t maskNum0 = processNum;
    for (uint64_t i = 0; i < ubLoopNum; i++) {
        // 判断尾块
        if (blockIdx_ == realCoreNum_ - 1 && i == ubLoopNum - 1) {
            totalNum = batchUbTailFactor_ * nSize_;
            loopNum = totalNum / processNum;
        }
        MainProcess(i, maskNumInt8, processNum, totalNum, addrOffset, loopNum, idxPtr);
    }
    indexBuf_.FreeTensor(indexLocal);
}

template <typename T, typename U, typename Y>
__aicore__ inline void MatrixDiagScatterLower<T, U, Y>::VFProcessInt8(uint32_t maskNumInt8, uint32_t processNum,
                                                                      uint32_t totalNum, U addrOffset, uint16_t loopNum,
                                                                      __local_mem__ U* idxPtr, __local_mem__ T* srcPtr,
                                                                      __local_mem__ T* dstPtr)
{
    uint32_t tailNum = totalNum - processNum * loopNum;
    uint32_t maskTailNum = tailNum * 2;
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> srcRegInt8;
        Reg::RegTensor<T> zeroRegInt8;
        Reg::RegTensor<U> indexRegInt8;
        Reg::RegTensor<T> lowerRegInt8;
        Reg::RegTensor<T> higherRegInt8;
        Reg::UnalignReg u0Int8;
        Reg::MaskReg mask0Int8;
        Reg::MaskReg mask1Int8;
        mask0Int8 = Reg::UpdateMask<T>(maskNumInt8);
        mask1Int8 = Reg::UpdateMask<T>(maskTailNum);
        Reg::DataCopy(indexRegInt8, idxPtr);
        for (uint16_t j = 0; j < loopNum; j++) {
            Reg::DataCopyUnAlignPre(u0Int8, srcPtr);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(srcRegInt8, u0Int8, srcPtr, processNum);
            Reg::Duplicate(zeroRegInt8, (T)0);
            Reg::Interleave(lowerRegInt8, higherRegInt8, srcRegInt8, zeroRegInt8);
            Reg::DataCopyScatter(dstPtr, lowerRegInt8, indexRegInt8, mask0Int8);
            Reg::Adds(indexRegInt8, indexRegInt8, addrOffset, mask0Int8);
        }
        if (tailNum != 0) {
            Reg::DataCopyUnAlignPre(u0Int8, srcPtr);
            Reg::DataCopyUnAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(srcRegInt8, u0Int8, srcPtr, processNum);
            Reg::Duplicate(zeroRegInt8, (U)0);
            Reg::Interleave(lowerRegInt8, higherRegInt8, srcRegInt8, zeroRegInt8);
            Reg::DataCopyScatter(dstPtr, lowerRegInt8, indexRegInt8, mask1Int8);
        }
    }
}
} // namespace MatrixDiagAsc
#endif // MATRIX_DIAG_SCATTER_LOWER_H
