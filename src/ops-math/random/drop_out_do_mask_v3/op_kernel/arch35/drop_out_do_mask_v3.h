/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_DO_MASK_V3_H
#define DROP_OUT_DO_MASK_V3_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace DropOutDoMaskV3 {
using namespace AscendC;
using namespace RandomKernelBase;

const uint32_t DOUBLE_BUFFER = 2;

template <typename T>
class DropOutDoMaskV3Op : public RandomKernelBaseOp {
public:
    __aicore__ inline DropOutDoMaskV3Op(TPipe* pipe, const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData), pipePtr_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void doMaskB32(AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf,
                                     AscendC::Reg::RegTensor<T> xInputVf, AscendC::Reg::RegTensor<T> yOutputVf,
                                     AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
                                     __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride,
                                     uint16_t index, uint32_t offset, float prob_reciprocal);
    __aicore__ inline void doMaskB16(AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf,
                                     AscendC::Reg::RegTensor<T> xInputVf, AscendC::Reg::RegTensor<T> yOutputVf,
                                     AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
                                     __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride,
                                     uint16_t index, uint32_t offset, float prob_reciprocal,
                                     AscendC::Reg::RegTensor<float> xFp32InputVf,
                                     AscendC::Reg::RegTensor<float> yFp32OutputVf);
    __aicore__ inline bool IsProbEqual(float a, float b);

    template <bool COPY_MASK>
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void CopyOutY(uint32_t loopIdx, uint32_t dataCount);

private:
    constexpr static uint32_t MASK_REG_LEN = 32;
    constexpr static uint32_t REG_COUNT = 2;
    constexpr static uint32_t INDEX_ONE = 1;
    constexpr static uint32_t ALIGN_128 = 128;

    AscendC::TPipe* pipePtr_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> xInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> maskInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> yOutputQueue_;
    AscendC::GlobalTensor<T> xInputGm_;
    AscendC::GlobalTensor<uint8_t> maskInputGm_;
    AscendC::GlobalTensor<T> yOutputGm_;

    float prob_ = 0.0f;
    uint64_t ubTailLoopSize_ = 0; // 当前coreUB尾循环搬运数据量
    uint64_t currLoopCount_ = 0;  // 当前core循环搬运数据次数

    static constexpr AscendC::Reg::CastTrait castTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline bool DropOutDoMaskV3Op<T>::IsProbEqual(float a, float b)
{
    return std::abs(a - b) <= std::numeric_limits<float>::epsilon();
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::doMaskB32(AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf,
                                                       AscendC::Reg::RegTensor<T> xInputVf,
                                                       AscendC::Reg::RegTensor<T> yOutputVf,
                                                       AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
                                                       __ubuf__ T* yOutputUbPtr, uint32_t& size,
                                                       uint32_t postUpdateStride, uint16_t index, uint32_t offset,
                                                       float prob_reciprocal)
{
    preg0 = AscendC::Reg::UpdateMask<float>(size);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(xInputVf, xInputUbPtr + index * postUpdateStride);
    AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xInputVf, xInputVf, prob_reciprocal, preg0);
    AscendC::Reg::Select<float>(yOutputVf, xInputVf, select0, maskInputVf);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(yOutputUbPtr + index * postUpdateStride,
                                                                          yOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::doMaskB16(
    AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf, AscendC::Reg::RegTensor<T> xInputVf,
    AscendC::Reg::RegTensor<T> yOutputVf, AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
    __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset,
    float prob_reciprocal, AscendC::Reg::RegTensor<float> xFp32InputVf, AscendC::Reg::RegTensor<float> yFp32OutputVf)
{
    preg0 = AscendC::Reg::UpdateMask<float>(size);
    AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xInputVf,
                                                                       xInputUbPtr + index * postUpdateStride);
    AscendC::Reg::Cast<float, T, castTraitB16ToB32>(xFp32InputVf, xInputVf, preg0);
    AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xFp32InputVf, xFp32InputVf, prob_reciprocal,
                                                                           preg0);
    AscendC::Reg::Select<float>(yFp32OutputVf, xFp32InputVf, select0, maskInputVf);
    AscendC::Reg::Cast<T, float, castTraitB32ToB16>(yOutputVf, yFp32OutputVf, preg0);
    AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yOutputUbPtr + index * postUpdateStride,
                                                                      yOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::Init(GM_ADDR x, GM_ADDR mask, GM_ADDR y)
{
    // Parsetiling data
    VarsInit();
    xInputGm_.SetGlobalBuffer((__gm__ T*)x);
    maskInputGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    yOutputGm_.SetGlobalBuffer((__gm__ T*)y);

    int64_t normBlockLoop = Ops::Base::CeilDiv(tiling_->normalCoreProNum, tiling_->singleBufferSize);
    int64_t normBlockTail = tiling_->normalCoreProNum - (normBlockLoop - 1) * tiling_->singleBufferSize;
    int64_t tailBlockLoop = Ops::Base::CeilDiv(tiling_->tailCoreProNum, tiling_->singleBufferSize);
    int64_t tailBlockTail = tiling_->tailCoreProNum - (tailBlockLoop - 1) * tiling_->singleBufferSize;

    if (blockIdx_ == tiling_->usedCoreNum - 1) {
        currLoopCount_ = tailBlockLoop;
        ubTailLoopSize_ = tailBlockTail;
    } else {
        currLoopCount_ = normBlockLoop;
        ubTailLoopSize_ = normBlockTail;
    }

    pipePtr_->InitBuffer(xInputQueue_, DOUBLE_BUFFER, tiling_->singleBufferSize * sizeof(T));
    pipePtr_->InitBuffer(
        maskInputQueue_, DOUBLE_BUFFER,
        Ops::Base::CeilAlign(static_cast<uint32_t>(tiling_->singleBufferSize), ALIGN_128) * sizeof(uint8_t));
    pipePtr_->InitBuffer(yOutputQueue_, DOUBLE_BUFFER, tiling_->singleBufferSize * sizeof(T));

    prob_ = tiling_->keepProb;
}

template <typename T>
template <bool COPY_MASK>
__aicore__ inline void DropOutDoMaskV3Op<T>::CopyIn(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.AllocTensor<T>();
    AscendC::DataCopyExtParams xCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(xInputUb_,
                         xInputGm_[blockIdx_ * tiling_->normalCoreProNum + loopIdx * tiling_->singleBufferSize],
                         xCopyParams, xPadParams);
    xInputQueue_.EnQue<T>(xInputUb_);

    event_t event_MTE2_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_MTE2_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_MTE2_MTE3);

    if constexpr (COPY_MASK) {
        AscendC::LocalTensor<uint8_t> maskInputUb_ = maskInputQueue_.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams maskCopyParams{
            1, (uint32_t)(Ops::Base::CeilAlign(dataCount, ALIGN_128) * sizeof(uint8_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> maskPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(maskInputUb_,
                             maskInputGm_[blockIdx_ * tiling_->normalCoreProNum + loopIdx * tiling_->singleBufferSize],
                             maskCopyParams, maskPadParams);
        maskInputQueue_.EnQue<uint8_t>(maskInputUb_);
    }
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::Compute(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.DeQue<T>();
    AscendC::LocalTensor<uint8_t> maskInputUb_ = maskInputQueue_.DeQue<uint8_t>();
    AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.AllocTensor<T>();
    constexpr uint16_t vRegSize = Ops::Base::GetVRegSize();
    uint16_t vfLoopNum = Ops::Base::CeilDiv(static_cast<uint32_t>(dataCount * sizeof(float)),
                                            static_cast<uint32_t>(vRegSize));
    __ubuf__ T* xInputUbPtr = (__ubuf__ T*)xInputUb_.GetPhyAddr();
    __ubuf__ T* yOutputUbPtr = (__ubuf__ T*)yOutputUb_.GetPhyAddr();
    __ubuf__ uint8_t* maskInputUbPtr = (__ubuf__ uint8_t*)maskInputUb_.GetPhyAddr();
    uint32_t postUpdateStride = vRegSize / sizeof(float); // 每次regbase计算的数据量
    float prob_reciprocal = static_cast<float>(1.0) / prob_;
    uint32_t size = dataCount;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> xInputVf;
        AscendC::Reg::RegTensor<float> xFp32InputVf;
        AscendC::Reg::RegTensor<T> yOutputVf;
        AscendC::Reg::RegTensor<float> yFp32OutputVf;
        AscendC::Reg::RegTensor<float> select0;
        AscendC::Reg::MaskReg preg0;
        AscendC::Reg::MaskReg maskInputVf0;
        AscendC::Reg::MaskReg maskInputVf1;
        AscendC::Reg::MaskReg newMask0;
        AscendC::Reg::MaskReg newMask1;
        AscendC::Reg::Duplicate(select0, 0.0f);

        if constexpr (AscendC::IsSameType<T, float>::value) {
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(
                    maskInputVf0, maskInputUbPtr + i * REG_COUNT * MASK_REG_LEN);
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(
                    maskInputVf1, maskInputUbPtr + (i * REG_COUNT + INDEX_ONE) * MASK_REG_LEN);
                AscendC::Reg::MaskDeInterleave<T>(newMask0, newMask1, maskInputVf0, maskInputVf1);
                doMaskB32(preg0, newMask0, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 0, prob_reciprocal);
            }
        } else {
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(
                    maskInputVf0, maskInputUbPtr + i * REG_COUNT * MASK_REG_LEN);
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(
                    maskInputVf1, maskInputUbPtr + (i * REG_COUNT + INDEX_ONE) * MASK_REG_LEN);
                AscendC::Reg::MaskDeInterleave<T>(newMask0, newMask1, maskInputVf0, maskInputVf1);
                doMaskB16(preg0, newMask0, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 0, prob_reciprocal, xFp32InputVf, yFp32OutputVf);
            }
        }
    }
    xInputQueue_.FreeTensor(xInputUb_);
    maskInputQueue_.FreeTensor(maskInputUb_);
    yOutputQueue_.EnQue<T>(yOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::CopyOutY(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.DeQue<T>();
    int64_t yOffset = blockIdx_ * tiling_->normalCoreProNum + loopIdx * tiling_->singleBufferSize;
    CopyOut(yOutputUb_, yOutputGm_, 1, (uint32_t)(dataCount * sizeof(T)), yOffset);
    yOutputQueue_.FreeTensor(yOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskV3Op<T>::Process()
{
    if (blockIdx_ >= tiling_->usedCoreNum) {
        return;
    }

    uint32_t dataCount = 0;

    if (IsProbEqual(prob_, 0.0f)) {
        for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
            dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                      static_cast<uint32_t>(tiling_->singleBufferSize);
            AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.AllocTensor<T>();
            AscendC::Duplicate(yOutputUb_, static_cast<T>(0), dataCount);
            yOutputQueue_.EnQue<T>(yOutputUb_);
            CopyOutY(idx, dataCount);
        }
        return;

    } else if (IsProbEqual(prob_, 1.0f)) {
        for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
            dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                      static_cast<uint32_t>(tiling_->singleBufferSize);
            CopyIn<false>(idx, dataCount);
            AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.DeQue<T>();
            int64_t yOffset = blockIdx_ * tiling_->normalCoreProNum + idx * tiling_->singleBufferSize;
            CopyOut(xInputUb_, yOutputGm_, 1, (uint32_t)(dataCount * sizeof(T)), yOffset);
            event_t event_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_MTE3_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_MTE3_MTE2);
            xInputQueue_.FreeTensor(xInputUb_);
        }
        return;
    }

    for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
        dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                  static_cast<uint32_t>(tiling_->singleBufferSize);
        CopyIn<true>(idx, dataCount);
        Compute(idx, dataCount);
        CopyOutY(idx, dataCount);
    }
}
} // namespace DropOutDoMaskV3
#endif // DROP_OUT_DO_MASK_V3_H
