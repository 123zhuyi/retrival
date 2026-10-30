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
 * \file cdist.h
 * \brief
 */
#ifndef __CDIST_H__
#define __CDIST_H__

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../cdist_tiling_data.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace NsCdist {

using namespace AscendC;

template <typename T>
class Cdist {
public:
    __aicore__ inline Cdist(){};
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const CdistTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const CdistTilingData* tilingData);
    __aicore__ inline void CopyInX1(uint32_t Offset);
    __aicore__ inline void CopyInX2(uint32_t Offset);
    __aicore__ inline void CopyOut(uint32_t Offset);
    __aicore__ inline void CastY();
    __aicore__ inline void CastXToB32();
    __aicore__ inline void Compute();
    __aicore__ inline void ComputeSplitM();
    __aicore__ inline void ProcessSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                         uint32_t blockFactorR);
    __aicore__ inline void ProcessNoSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                           uint32_t blockFactorR);
    __aicore__ inline void CalSplitMResult(int32_t processNum);
    __aicore__ inline void ComputeOneSize(__local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                          __local_mem__ float* dstPtr);
    __aicore__ inline void ComputePNorm2(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                         __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                         __local_mem__ float* dstPtr);
    __aicore__ inline void ComputePNorm1(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                         __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                         __local_mem__ float* dstPtr);
    __aicore__ inline void ComputePNorm0(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                         __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                         __local_mem__ float* dstPtr);
    __aicore__ inline void ComputePNormInf(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                           __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                           __local_mem__ float* dstPtr);
    __aicore__ inline void ComputePNormOther(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                             __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                             __local_mem__ float* dstPtr);

private:
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t BLOCK_SIZE = 32;
    constexpr static uint32_t BASE_ONE = 1;
    constexpr static uint32_t LOOP_ZERO = 0;
    constexpr static ExpConfig expConfig = {ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
    constexpr static LnConfig lnConfig = {LnAlgo::PRECISION_1ULP_FTZ_FALSE};
    constexpr static SqrtConfig sqrtConfig = {SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
    int64_t blockIdx_;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECCALC, 1> tmpQueue_;
    TQue<QuePosition::VECCALC, 1> x1CastQueue_;
    TQue<QuePosition::VECCALC, 1> x2CastQueue_;
    TQue<QuePosition::VECCALC, 1> yCastQueue_;
    GlobalTensor<T> x1GM_;
    GlobalTensor<T> x2GM_;
    GlobalTensor<T> yGM_;
    LocalTensor<float> yFp32_;
    LocalTensor<float> tmpLocal_;
    const CdistTilingData* tiling_;
    int32_t vlLen_ = Ops::Base::GetVRegSize() / sizeof(float);
    uint32_t realCoreNum_ = 0;
    uint32_t B_ = 0;
    uint32_t P_ = 0;
    uint32_t R_ = 0;
    uint32_t RAlign_ = 0;
    uint32_t M_ = 0;
    uint32_t MAlign_ = 0;
    uint32_t blockMainNumB_ = 0;
    uint32_t blockTailNumB_ = 0;
    uint32_t blockMainFactorB_ = 0;
    uint32_t blockTailFactorB_ = 0;
    uint32_t blockMainNumP_ = 0;
    uint32_t blockTailNumP_ = 0;
    uint32_t blockMainFactorP_ = 0;
    uint32_t blockTailFactorP_ = 0;
    uint32_t blockMainNumR_ = 0;
    uint32_t blockTailNumR_ = 0;
    uint32_t blockMainFactorR_ = 0;
    uint32_t blockTailFactorR_ = 0;
    uint32_t ubLoopNumB_ = 0;
    uint32_t ubFactorB_ = 0;
    uint32_t ubTailFactorB_ = 0;
    uint32_t ubLoopNumP_ = 0;
    uint32_t ubFactorP_ = 0;
    uint32_t ubTailFactorP_ = 0;
    uint32_t ubLoopNumR_ = 0;
    uint32_t ubFactorR_ = 0;
    uint32_t ubTailFactorR_ = 0;
    uint32_t ubLoopNumM_ = 0;
    uint32_t ubFactorM_ = 0;
    uint32_t ubTailFactorM_ = 0;
    float p_ = 0;
    uint32_t bSize_ = 0;
    uint32_t pSize_ = 0;
    uint32_t rSize_ = 0;
    uint32_t mSize_ = 0;
    uint32_t ubFactorMAlign_ = 0;
    uint32_t ubFactorRAlign_ = 0;
    // Datacopy params
    DataCopyExtParams copyInParamsX1_{1, 0, 0, 0, 0};
    DataCopyExtParams copyInParamsX2_{1, 0, 0, 0, 0};
    LoopModeParams loopParamX1_{1, 0, 0, 0, 0, 0};
    LoopModeParams loopParamX2_{1, 0, 0, 0, 0, 0};
    DataCopyExtParams copyOutParams_{1, 0, 0, 0, 0};
    LoopModeParams loopParamOut_{1, 0, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams_{false, 0, 0, 0};
};

template <typename T>
__aicore__ inline void Cdist<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const CdistTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tiling_ = tilingData;
    ParseTilingData(tiling_);
    x1GM_.SetGlobalBuffer((__gm__ T*)x1);
    x2GM_.SetGlobalBuffer((__gm__ T*)x2);
    yGM_.SetGlobalBuffer((__gm__ T*)y);
    if (ubLoopNumM_ == 1) {
        ubFactorMAlign_ = ((ubFactorM_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    } else {
        ubFactorMAlign_ = ubFactorM_;
    }
    ubFactorRAlign_ = ((ubFactorR_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    pipe_->InitBuffer(x1Queue_, BUFFER_NUM, ubFactorB_ * ubFactorP_ * ubFactorMAlign_ * sizeof(T));
    pipe_->InitBuffer(x2Queue_, BUFFER_NUM, ubFactorB_ * ubFactorR_ * ubFactorMAlign_ * sizeof(T));
    pipe_->InitBuffer(yQueue_, BUFFER_NUM, ubFactorB_ * ubFactorP_ * ubFactorRAlign_ * sizeof(T));
    pipe_->InitBuffer(tmpQueue_, 1, BLOCK_SIZE);
    if (sizeof(T) != sizeof(float)) {
        pipe_->InitBuffer(x1CastQueue_, 1, ubFactorB_ * ubFactorP_ * ubFactorMAlign_ * sizeof(float));
        pipe_->InitBuffer(x2CastQueue_, 1, ubFactorB_ * ubFactorR_ * ubFactorMAlign_ * sizeof(float));
        pipe_->InitBuffer(yCastQueue_, 1, ubFactorB_ * ubFactorP_ * ubFactorRAlign_ * sizeof(float));
        yFp32_ = yCastQueue_.AllocTensor<float>();
        Duplicate<float>(yFp32_, (float)0, ubFactorB_ * ubFactorP_ * ubFactorRAlign_);
        yCastQueue_.EnQue(yFp32_);
    } else {
        pipe_->InitBuffer(x1CastQueue_, 1, 0);
        pipe_->InitBuffer(x2CastQueue_, 1, 0);
        pipe_->InitBuffer(yCastQueue_, 1, 0);
        yFp32_ = yQueue_.AllocTensor<float>();
        Duplicate<float>(yFp32_, (float)0, ubFactorB_ * ubFactorP_ * ubFactorRAlign_);
        yQueue_.EnQue(yFp32_);
    }
    tmpLocal_ = tmpQueue_.AllocTensor<float>();
    Duplicate<float>(tmpLocal_, (float)0, 1);
    tmpQueue_.EnQue(tmpLocal_);
}

template <typename T>
__aicore__ inline void Cdist<T>::ParseTilingData(const CdistTilingData* tdPtr)
{
    B_ = tdPtr->B;
    P_ = tdPtr->P;
    R_ = tdPtr->R;
    M_ = tdPtr->M;
    blockMainNumB_ = tdPtr->blockMainNumB;
    blockTailNumB_ = tdPtr->blockTailNumB;
    blockMainFactorB_ = tdPtr->blockMainFactorB;
    blockTailFactorB_ = tdPtr->blockTailFactorB;
    blockMainNumP_ = tdPtr->blockMainNumP;
    blockTailNumP_ = tdPtr->blockTailNumP;
    blockMainFactorP_ = tdPtr->blockMainFactorP;
    blockTailFactorP_ = tdPtr->blockTailFactorP;
    blockMainNumR_ = tdPtr->blockMainNumR;
    blockTailNumR_ = tdPtr->blockTailNumR;
    blockMainFactorR_ = tdPtr->blockMainFactorR;
    blockTailFactorR_ = tdPtr->blockTailFactorR;
    ubLoopNumB_ = tdPtr->ubLoopNumB;
    ubFactorB_ = tdPtr->ubFactorB;
    ubTailFactorB_ = tdPtr->ubTailFactorB;
    ubLoopNumP_ = tdPtr->ubLoopNumP;
    ubFactorP_ = tdPtr->ubFactorP;
    ubTailFactorP_ = tdPtr->ubTailFactorP;
    ubLoopNumR_ = tdPtr->ubLoopNumR;
    ubFactorR_ = tdPtr->ubFactorR;
    ubTailFactorR_ = tdPtr->ubTailFactorR;
    ubLoopNumM_ = tdPtr->ubLoopNumM;
    ubFactorM_ = tdPtr->ubFactorM;
    ubTailFactorM_ = tdPtr->ubTailFactorM;
    p_ = tdPtr->p;
}

template <typename T>
__aicore__ inline void Cdist<T>::CopyInX1(uint32_t Offset)
{
    LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
    copyInParamsX1_.blockCount = static_cast<uint16_t>(pSize_);
    copyInParamsX1_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                    static_cast<uint32_t>(mSize_ * sizeof(T));             // unit Byte
    copyInParamsX1_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T)); // unit Byte
    copyInParamsX1_.dstStride = 0; // unit block(32byte)
    loopParamX1_.loop1Size = static_cast<uint32_t>(bSize_);
    loopParamX1_.loop2Size = 1;
    loopParamX1_.loop1SrcStride = static_cast<uint64_t>((M_ * P_) * sizeof(T));
    loopParamX1_.loop2SrcStride = 0;
    loopParamX1_.loop1DstStride = static_cast<uint64_t>((pSize_ * MAlign_) * sizeof(T));
    loopParamX1_.loop2DstStride = 0;
    SetLoopModePara(loopParamX1_, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(x1Local, x1GM_[Offset], copyInParamsX1_, padParams_);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    x1Queue_.EnQue(x1Local);
}

template <typename T>
__aicore__ inline void Cdist<T>::CopyInX2(uint32_t Offset)
{
    LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
    copyInParamsX2_.blockCount = static_cast<uint16_t>(rSize_);
    copyInParamsX2_.blockLen = (ubLoopNumM_ == 1) ? static_cast<uint32_t>(M_ * sizeof(T)) :
                                                    static_cast<uint32_t>(mSize_ * sizeof(T));             // unit Byte
    copyInParamsX2_.srcStride = (ubLoopNumM_ == 1) ? 0 : static_cast<uint32_t>((M_ - mSize_) * sizeof(T)); // unit Byte
    copyInParamsX2_.dstStride = 0; // unit block(32byte)
    loopParamX2_.loop1Size = static_cast<uint32_t>(bSize_);
    loopParamX2_.loop2Size = 1;
    loopParamX2_.loop1SrcStride = static_cast<uint64_t>((M_ * R_) * sizeof(T));
    loopParamX2_.loop2SrcStride = 0;
    loopParamX2_.loop1DstStride = static_cast<uint64_t>((rSize_ * MAlign_) * sizeof(T));
    loopParamX2_.loop2DstStride = 0;
    SetLoopModePara(loopParamX2_, DataCopyMVType::OUT_TO_UB);
    DataCopyPad(x2Local, x2GM_[Offset], copyInParamsX2_, padParams_);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    x2Queue_.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void Cdist<T>::CopyOut(uint32_t Offset)
{
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    copyOutParams_.blockCount = static_cast<uint16_t>(pSize_);
    copyOutParams_.blockLen = static_cast<uint32_t>(rSize_ * sizeof(T)); // unit Byte
    copyOutParams_.srcStride = 0;
    copyOutParams_.dstStride = static_cast<uint32_t>((R_ - rSize_) * sizeof(T));
    loopParamOut_.loop1Size = static_cast<uint32_t>(bSize_);
    loopParamOut_.loop2Size = 1;
    loopParamOut_.loop1SrcStride = static_cast<uint64_t>((pSize_ * RAlign_) * sizeof(T));
    loopParamOut_.loop2SrcStride = 0;
    loopParamOut_.loop1DstStride = static_cast<uint64_t>((P_ * R_) * sizeof(T));
    loopParamOut_.loop2DstStride = 0;
    SetLoopModePara(loopParamOut_, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(yGM_[Offset], yLocal, copyOutParams_);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    yQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Cdist<T>::CastY()
{
    LocalTensor<T> yLocal;
    if constexpr (sizeof(T) != sizeof(float)) {
        yLocal = yQueue_.AllocTensor<T>();
        yFp32_ = yCastQueue_.DeQue<float>();
        Cast(yLocal, yFp32_, RoundMode::CAST_RINT, (uint32_t)(ubFactorB_ * ubFactorP_ * ubFactorRAlign_));
    } else {
        yLocal = yQueue_.DeQue<T>();
    }
    yQueue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void Cdist<T>::CastXToB32()
{
    LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
    LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
    LocalTensor<float> x1Cast;
    LocalTensor<float> x2Cast;
    if constexpr (sizeof(T) != sizeof(float)) {
        LocalTensor<float> x1Cast = x1CastQueue_.AllocTensor<float>();
        LocalTensor<float> x2Cast = x2CastQueue_.AllocTensor<float>();
        Cast(x1Cast, x1Local, RoundMode::CAST_NONE, (uint32_t)(ubFactorB_ * ubFactorP_ * ubFactorMAlign_));
        Cast(x2Cast, x2Local, RoundMode::CAST_NONE, (uint32_t)(ubFactorB_ * ubFactorR_ * ubFactorMAlign_));
        x1CastQueue_.EnQue(x1Cast);
        x2CastQueue_.EnQue(x2Cast);
    } else {
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }
    x1Queue_.FreeTensor(x1Local);
    x2Queue_.FreeTensor(x2Local);
}

template <typename T>
__aicore__ inline void Cdist<T>::Compute()
{
    LocalTensor<float> x1Local;
    LocalTensor<float> x2Local;
    if constexpr (sizeof(T) != sizeof(float)) {
        x1Local = x1CastQueue_.DeQue<float>();
        x2Local = x2CastQueue_.DeQue<float>();
        yFp32_ = yCastQueue_.DeQue<float>();
    } else {
        x1Local = x1Queue_.DeQue<T>();
        x2Local = x2Queue_.DeQue<T>();
        yFp32_ = yQueue_.DeQue<float>();
    }
    auto* srcPtrX1 = (__local_mem__ float*)x1Local.GetPhyAddr();
    auto* srcPtrX2 = (__local_mem__ float*)x2Local.GetPhyAddr();
    auto* dstPtr = (__local_mem__ float*)yFp32_.GetPhyAddr();
    ComputeOneSize(srcPtrX1, srcPtrX2, dstPtr);
    if constexpr (sizeof(T) != sizeof(float)) {
        yCastQueue_.EnQue(yFp32_);
        x1CastQueue_.FreeTensor(x1Local);
        x2CastQueue_.FreeTensor(x2Local);
    } else {
        yQueue_.EnQue(yFp32_);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputeSplitM()
{
    int32_t processNum = bSize_ * pSize_ * rSize_;
    LocalTensor<float> x1LocalSplitM;
    LocalTensor<float> x2LocalSplitM;
    if constexpr (sizeof(T) != sizeof(float)) {
        x1LocalSplitM = x1CastQueue_.DeQue<float>();
        x2LocalSplitM = x2CastQueue_.DeQue<float>();
        yFp32_ = yCastQueue_.DeQue<float>();
    } else {
        x1LocalSplitM = x1Queue_.DeQue<T>();
        x2LocalSplitM = x2Queue_.DeQue<T>();
        yFp32_ = yQueue_.DeQue<float>();
    }
    tmpLocal_ = tmpQueue_.DeQue<float>();
    auto* dstPtr = (__local_mem__ float*)tmpLocal_.GetPhyAddr();
    auto* srcPtrX1 = (__local_mem__ float*)x1LocalSplitM.GetPhyAddr();
    auto* srcPtrX2 = (__local_mem__ float*)x2LocalSplitM.GetPhyAddr();
    ComputeOneSize(srcPtrX1, srcPtrX2, dstPtr);
    if (p_ == static_cast<float>(INFINITY)) {
        Max(yFp32_, tmpLocal_, yFp32_, processNum);
    } else {
        Add(yFp32_, tmpLocal_, yFp32_, processNum);
    }
    if constexpr (sizeof(T) != sizeof(float)) {
        yCastQueue_.EnQue(yFp32_);
        x1CastQueue_.FreeTensor(x1LocalSplitM);
        x2CastQueue_.FreeTensor(x2LocalSplitM);
    } else {
        yQueue_.EnQue(yFp32_);
        x1Queue_.FreeTensor(x1LocalSplitM);
        x2Queue_.FreeTensor(x2LocalSplitM);
    }
    tmpQueue_.EnQue(tmpLocal_);
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputeOneSize(__local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                                __local_mem__ float* dstPtr)
{
    uint16_t M = (ubLoopNumM_ == 1) ? M_ : mSize_;
    uint16_t loopNumM = M / vlLen_;
    uint32_t tailNumM = M - vlLen_ * loopNumM;
    for (uint32_t b = 0; b < bSize_; b++) {
        for (uint32_t p = 0; p < pSize_; p++) {
            for (uint32_t r = 0; r < rSize_; r++) {
                if (p_ == 2.0f) {
                    ComputePNorm2(b, p, r, loopNumM, tailNumM, srcPtrX1, srcPtrX2, dstPtr);
                } else if (p_ == 1.0f) {
                    ComputePNorm1(b, p, r, loopNumM, tailNumM, srcPtrX1, srcPtrX2, dstPtr);
                } else if (p_ == 0.0f) {
                    ComputePNorm0(b, p, r, loopNumM, tailNumM, srcPtrX1, srcPtrX2, dstPtr);
                } else if (p_ == static_cast<float>(INFINITY)) {
                    ComputePNormInf(b, p, r, loopNumM, tailNumM, srcPtrX1, srcPtrX2, dstPtr);
                } else {
                    ComputePNormOther(b, p, r, loopNumM, tailNumM, srcPtrX1, srcPtrX2, dstPtr);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputePNorm2(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                               __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                               __local_mem__ float* dstPtr)
{
    uint32_t maksTailNumNorm2 = tailNumM;
    uint32_t maskOneNumNorm2 = BASE_ONE;
    __local_mem__ float* yOffsetNorm2 = dstPtr + b * pSize_ * RAlign_ + p * RAlign_ + r;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x1RegNorm2;
        Reg::RegTensor<float> x2RegNorm2;
        Reg::RegTensor<float> subRegNorm2;
        Reg::RegTensor<float> mulRegNorm2;
        Reg::RegTensor<float> sumRegNorm2;
        Reg::RegTensor<float> resultRegNorm2;
        Reg::RegTensor<float> dstRegNorm2;
        Reg::MaskReg maskAllNorm2 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskTailNorm2;
        Reg::MaskReg maskOneNorm2;
        Reg::UnalignRegForStore uRegNorm2;
        maskTailNorm2 = Reg::UpdateMask<float>(maksTailNumNorm2);
        maskOneNorm2 = Reg::UpdateMask<float>(maskOneNumNorm2);
        static constexpr Reg::SqrtSpecificMode modesqrt = {Reg::MaskMergeMode::ZEROING, true,
                                                           SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};
        Reg::Duplicate(dstRegNorm2, (float)0);
        for (uint16_t m = 0; m < loopNumM; m++) {
            __local_mem__ float* x1OffsetNorm2 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * m;
            __local_mem__ float* x2OffsetNorm2 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * m;
            Reg::LoadAlign(x1RegNorm2, x1OffsetNorm2);
            Reg::LoadAlign(x2RegNorm2, x2OffsetNorm2);
            Reg::Sub(subRegNorm2, x1RegNorm2, x2RegNorm2, maskAllNorm2);
            Reg::Mul(mulRegNorm2, subRegNorm2, subRegNorm2, maskAllNorm2);
            Reg::ReduceSum(sumRegNorm2, mulRegNorm2, maskAllNorm2);
            Reg::Add(dstRegNorm2, dstRegNorm2, sumRegNorm2, maskOneNorm2);
        }
        __local_mem__ float* x1OffsetNorm2 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * loopNumM;
        __local_mem__ float* x2OffsetNorm2 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * loopNumM;
        Reg::LoadAlign(x1RegNorm2, x1OffsetNorm2);
        Reg::LoadAlign(x2RegNorm2, x2OffsetNorm2);
        Reg::Sub(subRegNorm2, x1RegNorm2, x2RegNorm2, maskTailNorm2);
        Reg::Mul(mulRegNorm2, subRegNorm2, subRegNorm2, maskTailNorm2);
        Reg::ReduceSum(sumRegNorm2, mulRegNorm2, maskTailNorm2);
        Reg::Add(dstRegNorm2, dstRegNorm2, sumRegNorm2, maskOneNorm2);
        if (ubLoopNumM_ == 1) {
            Reg::Sqrt<float, &modesqrt>(resultRegNorm2, dstRegNorm2, maskOneNorm2);
            Reg::StoreUnAlign(yOffsetNorm2, resultRegNorm2, uRegNorm2, BASE_ONE);
        } else {
            Reg::StoreUnAlign(yOffsetNorm2, dstRegNorm2, uRegNorm2, BASE_ONE);
        }
        Reg::StoreUnAlignPost(yOffsetNorm2, uRegNorm2, 0);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputePNorm1(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                               __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                               __local_mem__ float* dstPtr)
{
    uint32_t maksTailNumNorm1 = tailNumM;
    uint32_t maskOneNumNorm1 = BASE_ONE;
    __local_mem__ float* yOffsetNorm1 = dstPtr + b * pSize_ * RAlign_ + p * RAlign_ + r;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x1RegNorm1;
        Reg::RegTensor<float> x2RegNorm1;
        Reg::RegTensor<float> subRegNorm1;
        Reg::RegTensor<float> absRegNorm1;
        Reg::RegTensor<float> sumRegNorm1;
        Reg::RegTensor<float> dstRegNorm1;
        Reg::MaskReg maskAllNorm1 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskTailNorm1;
        Reg::MaskReg maskOneNorm1;
        Reg::UnalignRegForStore uRegNorm1;
        maskTailNorm1 = Reg::UpdateMask<float>(maksTailNumNorm1);
        maskOneNorm1 = Reg::UpdateMask<float>(maskOneNumNorm1);
        Reg::Duplicate(dstRegNorm1, (float)0);
        for (uint16_t m = 0; m < loopNumM; m++) {
            __local_mem__ float* x1OffsetNorm1 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * m;
            __local_mem__ float* x2OffsetNorm1 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * m;
            Reg::LoadAlign(x1RegNorm1, x1OffsetNorm1);
            Reg::LoadAlign(x2RegNorm1, x2OffsetNorm1);
            Reg::Sub(subRegNorm1, x1RegNorm1, x2RegNorm1, maskAllNorm1);
            Reg::Abs(absRegNorm1, subRegNorm1, maskAllNorm1);
            Reg::ReduceSum(sumRegNorm1, absRegNorm1, maskAllNorm1);
            Reg::Add(dstRegNorm1, dstRegNorm1, sumRegNorm1, maskOneNorm1);
        }
        __local_mem__ float* x1OffsetNorm1 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * loopNumM;
        __local_mem__ float* x2OffsetNorm1 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * loopNumM;
        Reg::LoadAlign(x1RegNorm1, x1OffsetNorm1);
        Reg::LoadAlign(x2RegNorm1, x2OffsetNorm1);
        Reg::Sub(subRegNorm1, x1RegNorm1, x2RegNorm1, maskTailNorm1);
        Reg::Abs(absRegNorm1, subRegNorm1, maskTailNorm1);
        Reg::ReduceSum(sumRegNorm1, absRegNorm1, maskTailNorm1);
        Reg::Add(dstRegNorm1, dstRegNorm1, sumRegNorm1, maskOneNorm1);
        Reg::StoreUnAlign(yOffsetNorm1, dstRegNorm1, uRegNorm1, BASE_ONE);
        Reg::StoreUnAlignPost(yOffsetNorm1, uRegNorm1, 0);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputePNorm0(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM, uint32_t tailNumM,
                                               __local_mem__ float* srcPtrX1, __local_mem__ float* srcPtrX2,
                                               __local_mem__ float* dstPtr)
{
    uint32_t maksTailNumNorm0 = tailNumM;
    uint32_t maskOneNumNorm0 = BASE_ONE;
    __local_mem__ float* yOffsetNorm0 = dstPtr + b * pSize_ * RAlign_ + p * RAlign_ + r;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x1RegNorm0;
        Reg::RegTensor<float> x2RegNorm0;
        Reg::RegTensor<float> subRegNorm0;
        Reg::RegTensor<float> absRegNorm0;
        Reg::RegTensor<float> castRegNorm0;
        Reg::RegTensor<float> minRegNorm0;
        Reg::RegTensor<float> sumRegNorm0;
        Reg::RegTensor<float> dstRegNorm0;
        Reg::MaskReg maskAllNorm0 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskTailNorm0;
        Reg::MaskReg maskOneNorm0;
        Reg::UnalignRegForStore uRegNorm0;
        maskTailNorm0 = Reg::UpdateMask<float>(maksTailNumNorm0);
        maskOneNorm0 = Reg::UpdateMask<float>(maskOneNumNorm0);
        Reg::Duplicate(dstRegNorm0, (float)0);
        for (uint16_t m = 0; m < loopNumM; m++) {
            __local_mem__ float* x1OffsetNorm0 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * m;
            __local_mem__ float* x2OffsetNorm0 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * m;
            Reg::LoadAlign(x1RegNorm0, x1OffsetNorm0);
            Reg::LoadAlign(x2RegNorm0, x2OffsetNorm0);
            Reg::Sub(subRegNorm0, x1RegNorm0, x2RegNorm0, maskAllNorm0);
            Reg::Abs(absRegNorm0, subRegNorm0, maskAllNorm0);
            Reg::Truncate<float, RoundMode::CAST_CEIL, Reg::MaskMergeMode::ZEROING>(castRegNorm0, absRegNorm0,
                                                                                    maskAllNorm0);
            Reg::Mins(minRegNorm0, castRegNorm0, (float)1, maskAllNorm0);
            Reg::ReduceSum(sumRegNorm0, minRegNorm0, maskAllNorm0);
            Reg::Add(dstRegNorm0, dstRegNorm0, sumRegNorm0, maskOneNorm0);
        }
        __local_mem__ float* x1OffsetNorm0 = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * loopNumM;
        __local_mem__ float* x2OffsetNorm0 = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * loopNumM;
        Reg::LoadAlign(x1RegNorm0, x1OffsetNorm0);
        Reg::LoadAlign(x2RegNorm0, x2OffsetNorm0);
        Reg::Sub(subRegNorm0, x1RegNorm0, x2RegNorm0, maskTailNorm0);
        Reg::Abs(absRegNorm0, subRegNorm0, maskTailNorm0);
        Reg::Truncate<float, RoundMode::CAST_CEIL, Reg::MaskMergeMode::ZEROING>(castRegNorm0, absRegNorm0,
                                                                                maskTailNorm0);
        Reg::Mins(minRegNorm0, castRegNorm0, (float)1, maskTailNorm0);
        Reg::ReduceSum(sumRegNorm0, minRegNorm0, maskTailNorm0);
        Reg::Add(dstRegNorm0, dstRegNorm0, sumRegNorm0, maskOneNorm0);
        Reg::StoreUnAlign(yOffsetNorm0, dstRegNorm0, uRegNorm0, BASE_ONE);
        Reg::StoreUnAlignPost(yOffsetNorm0, uRegNorm0, 0);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputePNormInf(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM,
                                                 uint32_t tailNumM, __local_mem__ float* srcPtrX1,
                                                 __local_mem__ float* srcPtrX2, __local_mem__ float* dstPtr)
{
    uint32_t maksTailNumNormInf = tailNumM;
    uint32_t maskOneNumNormInf = BASE_ONE;
    __local_mem__ float* yOffsetNormInf = dstPtr + b * pSize_ * RAlign_ + p * RAlign_ + r;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x1RegNormInf;
        Reg::RegTensor<float> x2RegNormInf;
        Reg::RegTensor<float> subRegNormInf;
        Reg::RegTensor<float> absRegNormInf;
        Reg::RegTensor<float> maxRegNormInf;
        Reg::RegTensor<float> dstRegNormInf;
        Reg::MaskReg maskAllNormInf = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskTailNormInf;
        Reg::MaskReg maskOneNormInf;
        Reg::UnalignRegForStore uRegNormInf;
        maskTailNormInf = Reg::UpdateMask<float>(maksTailNumNormInf);
        maskOneNormInf = Reg::UpdateMask<float>(maskOneNumNormInf);
        Reg::Duplicate(dstRegNormInf, (float)0);
        for (uint16_t m = 0; m < loopNumM; m++) {
            __local_mem__ float* x1OffsetNormInf = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * m;
            __local_mem__ float* x2OffsetNormInf = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * m;
            Reg::LoadAlign(x1RegNormInf, x1OffsetNormInf);
            Reg::LoadAlign(x2RegNormInf, x2OffsetNormInf);
            Reg::Sub(subRegNormInf, x1RegNormInf, x2RegNormInf, maskAllNormInf);
            Reg::Abs(absRegNormInf, subRegNormInf, maskAllNormInf);
            Reg::ReduceMax(maxRegNormInf, absRegNormInf, maskAllNormInf);
            Reg::Max(dstRegNormInf, maxRegNormInf, dstRegNormInf, maskOneNormInf);
        }
        __local_mem__ float* x1OffsetNormInf = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * loopNumM;
        __local_mem__ float* x2OffsetNormInf = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * loopNumM;
        Reg::LoadAlign(x1RegNormInf, x1OffsetNormInf);
        Reg::LoadAlign(x2RegNormInf, x2OffsetNormInf);
        Reg::Sub(subRegNormInf, x1RegNormInf, x2RegNormInf, maskTailNormInf);
        Reg::Abs(absRegNormInf, subRegNormInf, maskAllNormInf);
        Reg::ReduceMax(maxRegNormInf, absRegNormInf, maskAllNormInf);
        Reg::Max(dstRegNormInf, maxRegNormInf, dstRegNormInf, maskOneNormInf);
        Reg::StoreUnAlign(yOffsetNormInf, dstRegNormInf, uRegNormInf, BASE_ONE);
        Reg::StoreUnAlignPost(yOffsetNormInf, uRegNormInf, 0);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ComputePNormOther(uint32_t b, uint32_t p, uint32_t r, uint16_t loopNumM,
                                                   uint32_t tailNumM, __local_mem__ float* srcPtrX1,
                                                   __local_mem__ float* srcPtrX2, __local_mem__ float* dstPtr)
{
    uint32_t maksTailNum = tailNumM;
    uint32_t maskOneNum = BASE_ONE;
    __local_mem__ float* yOffset = dstPtr + b * pSize_ * RAlign_ + p * RAlign_ + r;
    __VEC_SCOPE__
    {
        Reg::RegTensor<float> x1Reg;
        Reg::RegTensor<float> x2Reg;
        Reg::RegTensor<float> subReg;
        Reg::RegTensor<float> absReg;
        Reg::RegTensor<float> logReg;
        Reg::RegTensor<float> mulReg;
        Reg::RegTensor<float> expReg;
        Reg::RegTensor<float> sumReg;
        Reg::RegTensor<float> resultReg;
        Reg::RegTensor<float> dstReg;
        Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskTail;
        Reg::MaskReg maskOne;
        Reg::UnalignRegForStore uReg;
        maskTail = Reg::UpdateMask<float>(maksTailNum);
        maskOne = Reg::UpdateMask<float>(maskOneNum);
        Reg::Duplicate(dstReg, (float)0);
        static constexpr Reg::ExpSpecificMode modeexp = {Reg::MaskMergeMode::ZEROING,
                                                         ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
        static constexpr Reg::LogSpecificMode modelog = {Reg::MaskMergeMode::ZEROING,
                                                         LogAlgo::PRECISION_1ULP_FTZ_FALSE};
        for (uint16_t m = 0; m < loopNumM; m++) {
            __local_mem__ float* x1Offset = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * m;
            __local_mem__ float* x2Offset = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * m;
            Reg::LoadAlign(x1Reg, x1Offset);
            Reg::LoadAlign(x2Reg, x2Offset);
            Reg::Sub(subReg, x1Reg, x2Reg, maskAll);
            Reg::Abs(absReg, subReg, maskAll);
            Reg::Log<float, &modelog>(logReg, absReg, maskAll);
            Reg::Muls(mulReg, logReg, (float)p_, maskAll);
            Reg::Exp<float, &modeexp>(expReg, mulReg, maskAll);
            Reg::ReduceSum(sumReg, expReg, maskAll);
            Reg::Add(dstReg, dstReg, sumReg, maskOne);
        }
        __local_mem__ float* x1Offset = srcPtrX1 + b * pSize_ * MAlign_ + p * MAlign_ + vlLen_ * loopNumM;
        __local_mem__ float* x2Offset = srcPtrX2 + b * rSize_ * MAlign_ + r * MAlign_ + vlLen_ * loopNumM;
        Reg::LoadAlign(x1Reg, x1Offset);
        Reg::LoadAlign(x2Reg, x2Offset);
        Reg::Sub(subReg, x1Reg, x2Reg, maskTail);
        Reg::Abs(absReg, subReg, maskTail);
        Reg::Log<float, &modelog>(logReg, absReg, maskTail);
        Reg::Muls(mulReg, logReg, (float)p_, maskTail);
        Reg::Exp<float, &modeexp>(expReg, mulReg, maskTail);
        Reg::ReduceSum(sumReg, expReg, maskTail);
        Reg::Add(dstReg, dstReg, sumReg, maskOne);
        if (ubLoopNumM_ == 1) {
            Reg::Log<float, &modelog>(logReg, dstReg, maskOne);
            Reg::Muls(mulReg, logReg, (float)(1 / p_), maskOne);
            Reg::Exp<float, &modeexp>(expReg, mulReg, maskOne);
            Reg::StoreUnAlign(yOffset, expReg, uReg, BASE_ONE);
        } else {
            Reg::StoreUnAlign(yOffset, dstReg, uReg, BASE_ONE);
        }
        Reg::StoreUnAlignPost(yOffset, uReg, 0);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::CalSplitMResult(int32_t processNum)
{
    if (p_ == 2.0f) {
        if constexpr (sizeof(T) != sizeof(float)) {
            yFp32_ = yCastQueue_.DeQue<float>();
            Sqrt<float, sqrtConfig>(yFp32_, yFp32_, processNum);
            yCastQueue_.EnQue(yFp32_);
        } else {
            yFp32_ = yQueue_.DeQue<float>();
            Sqrt<float, sqrtConfig>(yFp32_, yFp32_, processNum);
            yQueue_.EnQue(yFp32_);
        }
    }
    if (p_ != 1.0f && p_ != 2.0f && p_ != static_cast<float>(INFINITY) && p_ != 0.0f) {
        if constexpr (sizeof(T) != sizeof(float)) {
            yFp32_ = yCastQueue_.DeQue<float>();
            Ln<float, lnConfig>(yFp32_, yFp32_, processNum);
            Muls(yFp32_, yFp32_, (float)(1 / p_), processNum);
            Exp<float, expConfig>(yFp32_, yFp32_, processNum);
            yCastQueue_.EnQue(yFp32_);
        } else {
            yFp32_ = yQueue_.DeQue<float>();
            Ln<float, lnConfig>(yFp32_, yFp32_, processNum);
            Muls(yFp32_, yFp32_, (float)(1 / p_), processNum);
            Exp<float, expConfig>(yFp32_, yFp32_, processNum);
            yQueue_.EnQue(yFp32_);
        }
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ProcessNoSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                                 uint32_t blockFactorR)
{
    uint32_t offsetX1 = 0;
    uint32_t offsetX2 = 0;
    uint32_t offsetY = 0;
    uint32_t rOffset = 0;
    offsetX1 = bOffset * P_ * M_ + pOffset * M_;
    MAlign_ = ((M_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
    CopyInX1(offsetX1);
    for (uint32_t rIdx = 0; rIdx < ubLoopNumR_; rIdx++) {
        rOffset = rOffsetBlock + rIdx * ubFactorR_;
        rSize_ = (rIdx == ubLoopNumR_ - 1) ? (blockFactorR - ubFactorR_ * rIdx) : ubFactorR_;
        if (rSize_ == 0) {
            continue;
        }
        RAlign_ = ((rSize_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
        offsetX2 = bOffset * R_ * M_ + rOffset * M_;
        offsetY = bOffset * P_ * R_ + pOffset * R_ + rOffset;
        CopyInX2(offsetX2);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventID);
        CastXToB32();
        Compute();
        CastY();
        CopyOut(offsetY);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::ProcessSplitM(uint32_t bOffset, uint32_t pOffset, uint32_t rOffsetBlock,
                                               uint32_t blockFactorR)
{
    uint32_t offsetX1 = 0;
    uint32_t offsetX2 = 0;
    uint32_t offsetY = 0;
    int32_t processNum = 0;
    uint32_t mOffset = 0;
    uint32_t rOffset = 0;
    for (uint32_t rIdx = 0; rIdx < ubLoopNumR_; rIdx++) {
        rOffset = rOffsetBlock + rIdx * ubFactorR_;
        rSize_ = (rIdx == ubLoopNumR_ - 1) ? (blockFactorR - ubFactorR_ * rIdx) : ubFactorR_;
        if (rSize_ == 0) {
            continue;
        }
        processNum = bSize_ * pSize_ * rSize_;
        offsetY = bOffset * P_ * R_ + pOffset * R_ + rOffset;
        if constexpr (sizeof(T) != sizeof(float)) {
            yFp32_ = yCastQueue_.DeQue<float>();
            Duplicate<float>(yFp32_, (float)0, processNum);
            yCastQueue_.EnQue(yFp32_);
        } else {
            yFp32_ = yQueue_.DeQue<float>();
            Duplicate<float>(yFp32_, (float)0, processNum);
            yQueue_.EnQue(yFp32_);
        }
        for (uint32_t mIdx = 0; mIdx < ubLoopNumM_; mIdx++) {
            mOffset = mIdx * ubFactorM_;
            mSize_ = (mIdx == ubLoopNumM_ - 1) ? ubTailFactorM_ : ubFactorM_;
            MAlign_ = ((mSize_ * sizeof(T) + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE) / sizeof(T);
            offsetX1 = bOffset * P_ * M_ + pOffset * M_ + mOffset;
            offsetX2 = bOffset * R_ * M_ + rOffset * M_ + mOffset;
            CopyInX1(offsetX1);
            CopyInX2(offsetX2);
            CastXToB32();
            ComputeSplitM();
        }
        CalSplitMResult(processNum);
        CastY();
        PipeBarrier<PIPE_ALL>();
        CopyOut(offsetY);
    }
}

template <typename T>
__aicore__ inline void Cdist<T>::Process()
{
    uint32_t bOffset = 0;
    uint32_t pOffset = 0;
    uint32_t bOffsetBlock = 0;
    uint32_t pOffsetBlock = 0;
    uint32_t rOffsetBlock = 0;
    uint32_t blockFactorB = 0;
    uint32_t blockFactorP = 0;
    uint32_t blockFactorR = 0;
    uint32_t blockNumP = blockMainNumP_ + blockTailNumP_;
    uint32_t blockNumR = blockMainNumR_ + blockTailNumR_;
    uint32_t bBlockIdx = blockIdx_ / (blockNumP * blockNumR);
    uint32_t prBlockIdx = blockIdx_ % (blockNumP * blockNumR);
    uint32_t pBlockIdx = prBlockIdx / blockNumR;
    uint32_t rBlockIdx = prBlockIdx % blockNumR;
    blockFactorB = (bBlockIdx < blockMainNumB_) ? blockMainFactorB_ : blockTailFactorB_;
    bOffsetBlock = (bBlockIdx < blockMainNumB_) ?
                       blockMainFactorB_ * bBlockIdx :
                       blockMainFactorB_ * blockMainNumB_ + (bBlockIdx - blockMainNumB_) * blockTailFactorB_;
    blockFactorR = (rBlockIdx < blockMainNumR_) ? blockMainFactorR_ : blockTailFactorR_;
    rOffsetBlock = (rBlockIdx < blockMainNumR_) ?
                       blockMainFactorR_ * rBlockIdx :
                       blockMainFactorR_ * blockMainNumR_ + (rBlockIdx - blockMainNumR_) * blockTailFactorR_;
    blockFactorP = (pBlockIdx < blockMainNumP_) ? blockMainFactorP_ : blockTailFactorP_;
    pOffsetBlock = (pBlockIdx < blockMainNumP_) ?
                       blockMainFactorP_ * pBlockIdx :
                       blockMainFactorP_ * blockMainNumP_ + (pBlockIdx - blockMainNumP_) * blockTailFactorP_;
    for (uint32_t bIdx = 0; bIdx < ubLoopNumB_; bIdx++) {
        bOffset = bOffsetBlock + bIdx * ubFactorB_;
        bSize_ = (bIdx == ubLoopNumB_ - 1) ? (blockFactorB - ubFactorB_ * bIdx) : ubFactorB_;
        if (bSize_ == 0) {
            continue;
        }
        for (uint32_t pIdx = 0; pIdx < ubLoopNumP_; pIdx++) {
            pOffset = pOffsetBlock + pIdx * ubFactorP_;
            pSize_ = (pIdx == ubLoopNumP_ - 1) ? (blockFactorP - ubFactorP_ * pIdx) : ubFactorP_;
            if (pSize_ == 0) {
                continue;
            }
            if (ubLoopNumM_ == 1) {
                ProcessNoSplitM(bOffset, pOffset, rOffsetBlock, blockFactorR);
            } else {
                ProcessSplitM(bOffset, pOffset, rOffsetBlock, blockFactorR);
            }
        }
    }
    yQueue_.FreeTensor(yFp32_);
    tmpQueue_.FreeTensor(tmpLocal_);
}
} // namespace NsCdist

#endif // CDIST_H
