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

#ifndef OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_BATCH
#define OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_BATCH

#include "op_kernel/platform_util.h"
#include "matrix_diag_tiling_data_struct.h"

namespace MatrixDiagAsc {
using namespace AscendC;

constexpr uint16_t VF_LENGTH = Ops::Base::GetVRegSize();

template <typename T, typename U>
class MatrixDiagSliceBatch {
public:
    __aicore__ inline MatrixDiagSliceBatch(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MatrixDiagScatterTilingData* tilingDataPtr, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void MatrixDiagDup();
    __aicore__ inline void MatrixDiagComputeIdx();
    template <typename V>
    __aicore__ inline void GenScatterIndex();
    __aicore__ inline void MatrixDiagCopyGM2Ub(int64_t xGMOffset);
    __aicore__ inline void MatrixDiagScatter();
    __aicore__ inline void MatrixDiagCopyOut(int64_t yGMOffset);
    __aicore__ inline void MatrixDiagB8Scatter(__ubuf__ T* yUbAddr, __ubuf__ T* xUbAddr, __ubuf__ U* indexAddr,
                                               uint16_t batchUbFactorLocal);

private:
    TPipe* pipe_ = nullptr;
    const MatrixDiagScatterTilingData* tdPtr_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> inputGM_;
    GlobalTensor<T> outputGM_;
    int64_t blockIdx_ = 0;
    int64_t bufferCnt_ = 2; // enable db
    int64_t actualCoreNum_ = 0;
    uint32_t nSize_ = 0;
    int64_t curBatchUbFactor_ = 0;
    int64_t curBatchUbTailFactor_ = 0;
    int64_t curBlockFactor_ = 0;
    int64_t curBlockTailFactor_ = 0;
    int64_t curBlockMainCount_ = 0;
    LocalTensor<U> indexLocal_;
};

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::Init(GM_ADDR x, GM_ADDR y,
                                                        const MatrixDiagScatterTilingData* tilingDataPtr, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tdPtr_ = tilingDataPtr;
    pipe_ = pipe;
    actualCoreNum_ = tdPtr_->realCoreNum;
    nSize_ = tdPtr_->nSize;
    curBatchUbFactor_ = tdPtr_->batchUbFactor;
    curBatchUbTailFactor_ = tdPtr_->batchUbTailFactor;
    curBlockFactor_ = tdPtr_->blockFactor;
    curBlockTailFactor_ = tdPtr_->blockTailFactor;
    curBlockMainCount_ = tdPtr_->blockMainCount;
    if (curBlockMainCount_ == actualCoreNum_) {
        curBlockMainCount_ = curBlockMainCount_ - 1;
        curBlockTailFactor_ = curBlockFactor_;
    }
    inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
    pipe_->InitBuffer(inQue_, bufferCnt_, curBatchUbFactor_ * nSize_ * sizeof(T));
    pipe_->InitBuffer(outQue_, bufferCnt_, curBatchUbFactor_ * nSize_ * nSize_ * sizeof(T));
    pipe_->InitBuffer(indexBuf_, VF_LENGTH);
    indexLocal_ = indexBuf_.Get<U>();
}
template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::Process()
{
    if (blockIdx_ > (actualCoreNum_ - 1)) {
        return;
    }
    int64_t ubLoops = curBlockFactor_;
    if (blockIdx_ > (curBlockMainCount_ - 1)) { // 是尾核
        ubLoops = curBlockTailFactor_;
    }
    MatrixDiagComputeIdx();
    for (int64_t idx_blockFactor = 0; idx_blockFactor < ubLoops; ++idx_blockFactor) {
        // GM -> UB
        int64_t xGMOffset = blockIdx_ * ubLoops * curBatchUbFactor_ * nSize_ +
                            idx_blockFactor * curBatchUbFactor_ * nSize_;
        if (blockIdx_ > (curBlockMainCount_ - 1)) { // 是尾核
            xGMOffset = curBlockMainCount_ * curBlockFactor_ * curBatchUbFactor_ * nSize_ +
                        (blockIdx_ - curBlockMainCount_) * ubLoops * curBatchUbFactor_ * nSize_ +
                        idx_blockFactor * curBatchUbFactor_ * nSize_;
            if ((blockIdx_ == (actualCoreNum_ - 1)) && (idx_blockFactor == (ubLoops - 1))) {
                curBatchUbFactor_ = curBatchUbTailFactor_;
            }
        }
        int64_t yGMOffset = xGMOffset * nSize_;
        MatrixDiagCopyGM2Ub(xGMOffset);
        MatrixDiagDup();
        MatrixDiagScatter();
        MatrixDiagCopyOut(yGMOffset);
    }
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagDup()
{
    LocalTensor<T> yUbFactorLocal = outQue_.AllocTensor<T>();
    Duplicate(yUbFactorLocal, static_cast<T>(0), curBatchUbFactor_ * nSize_ * nSize_);
    outQue_.EnQue<T>(yUbFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagCopyGM2Ub(int64_t xGMOffset)
{
    LocalTensor<T> ubFactorLocal = inQue_.AllocTensor<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = curBatchUbFactor_ * nSize_ * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPadExtParams<T> copyInPadParams{false, 0, 0, 0};
    DataCopyPad(ubFactorLocal, inputGM_[xGMOffset], copyInParams, copyInPadParams);
    inQue_.EnQue<T>(ubFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagB8Scatter(__ubuf__ T* yUbAddr, __ubuf__ T* xUbAddr,
                                                                       __ubuf__ U* indexAddr,
                                                                       uint16_t batchUbFactorLocal)
{
    uint16_t vfLen = VF_LENGTH / sizeof(U);
    uint32_t indexStride = nSize_ * nSize_;
    uint32_t xUbMain = nSize_ > vfLen ? vfLen : nSize_;
    uint32_t xUbTail = nSize_ > vfLen ? (nSize_ - vfLen) : 0;
    uint32_t mStride = xUbMain * (nSize_ + 1);
    __VEC_SCOPE__
    {
        Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(xUbMain);
        Reg::MaskReg p1 = AscendC::Reg::UpdateMask<U>(xUbTail);
        Reg::RegTensor<U> vd0;
        Reg::RegTensor<U> indexReg;
        Reg::RegTensor<T> xReg;
        Reg::RegTensor<T> xReg0;
        Reg::RegTensor<T> xReg1;
        Reg::UnalignReg ureg;
        Reg::AddrReg aReg0;
        Reg::DataCopy(indexReg, indexAddr);
        Reg::DataCopyUnAlignPre(ureg, xUbAddr);
        for (uint16_t idx_batchUbFactor = 0; idx_batchUbFactor < batchUbFactorLocal; ++idx_batchUbFactor) {
            aReg0 = Reg::CreateAddrReg<T>(idx_batchUbFactor, nSize_);
            Reg::DataCopyUnAlignPre(ureg, xUbAddr, aReg0);
            Reg::DataCopyUnAlign(xReg, ureg, xUbAddr, aReg0, 0);
            Reg::Interleave(xReg0, xReg1, xReg, xReg);
            Reg::DataCopyScatter(yUbAddr, xReg0, indexReg, p0);
            Reg::Adds(vd0, indexReg, mStride, p0);
            Reg::DataCopyScatter(yUbAddr, xReg1, vd0, p1);
            Reg::Adds(indexReg, indexReg, indexStride, p0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagScatter()
{
    LocalTensor<T> yUbFactorLocal = outQue_.DeQue<T>();
    auto yUbAddr = (__ubuf__ T*)yUbFactorLocal.GetPhyAddr();
    // index
    auto indexAddr = (__ubuf__ U*)indexLocal_.GetPhyAddr();
    // xUb
    LocalTensor<T> ubFactorLocal = inQue_.DeQue<T>();
    auto xUbAddr = (__ubuf__ T*)ubFactorLocal.GetPhyAddr();

    uint16_t batchUbFactorLocal = curBatchUbFactor_;
    if constexpr (sizeof(T) == 1) {
        MatrixDiagB8Scatter(yUbAddr, xUbAddr, indexAddr, batchUbFactorLocal);
    } else {
        uint32_t indexStride = nSize_ * nSize_;
        uint32_t num = nSize_;
        __VEC_SCOPE__
        {
            Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
            Reg::RegTensor<U> vd0;
            Reg::RegTensor<U> indexReg;
            Reg::RegTensor<T> xReg;
            Reg::UnalignReg ureg;
            Reg::AddrReg aReg0;
            Reg::DataCopy(indexReg, indexAddr);
            for (uint16_t idx_batchUbFactor = 0; idx_batchUbFactor < batchUbFactorLocal; ++idx_batchUbFactor) {
                aReg0 = Reg::CreateAddrReg<T>(idx_batchUbFactor, nSize_);
                Reg::DataCopyUnAlignPre(ureg, xUbAddr, aReg0);
                Reg::DataCopyUnAlign(xReg, ureg, xUbAddr, aReg0, 0);
                Reg::DataCopyScatter(yUbAddr, xReg, indexReg, p0);
                Reg::Adds(indexReg, indexReg, indexStride, p0);
            }
        }
    }
    outQue_.EnQue<T>(yUbFactorLocal);
    inQue_.FreeTensor(ubFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagCopyOut(int64_t yGMOffset)
{
    LocalTensor<T> yUbFactorLocal = outQue_.DeQue<T>();
    DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = curBatchUbFactor_ * nSize_ * nSize_ * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopyPad(outputGM_[yGMOffset], yUbFactorLocal, copyInParams);
    outQue_.FreeTensor(yUbFactorLocal);
}

template <typename T, typename U>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::MatrixDiagComputeIdx()
{
    // 计算index
    if constexpr (IsSameType<U, uint32_t>::value) {
        GenScatterIndex<int32_t>();
    } else if constexpr (IsSameType<U, uint16_t>::value) {
        GenScatterIndex<int16_t>();
    } else if constexpr (IsSameType<U, uint64_t>::value) {
        GenScatterIndex<int64_t>();
    }
}

template <typename T, typename U>
template <typename V>
__aicore__ inline void MatrixDiagSliceBatch<T, U>::GenScatterIndex()
{
    auto dstAddr = (__ubuf__ U*)indexLocal_.GetPhyAddr();
    uint16_t vfLen = VF_LENGTH / sizeof(U);
    __VEC_SCOPE__
    {
        Reg::RegTensor<U> indexReg;
        Reg::RegTensor<V> tmp;
        Reg::RegTensor<U> v0;
        Reg::RegTensor<U> vd1;
        Reg::RegTensor<U> vd2;
        Reg::RegTensor<U> vd3;
        Reg::RegTensor<U> vd4;
        Reg::RegTensor<U> vd5;
        Reg::RegTensor<U> vd6;
        uint32_t num = vfLen;
        Reg::MaskReg p0 = AscendC::Reg::UpdateMask<U>(num);
        Reg::Arange(tmp, 0);
        indexReg = (Reg::RegTensor<U>&)tmp;
        Reg::Duplicate(v0, static_cast<U>(nSize_), p0);
        Reg::Div(vd1, indexReg, v0, p0);
        Reg::Muls(vd2, vd1, nSize_ * nSize_, p0);
        Reg::Mul(vd3, vd1, v0, p0);
        Reg::Sub(vd4, indexReg, vd3, p0);
        Reg::Muls(vd5, vd4, nSize_ + 1, p0);
        Reg::Add(vd6, vd2, vd5, p0);
        Reg::DataCopy(dstAddr, vd6, p0);
    }
}
} // namespace MatrixDiagAsc
#endif // OP_KERNEL_MATRIX_DIAG_SCATTER_SLICE_BATCH
