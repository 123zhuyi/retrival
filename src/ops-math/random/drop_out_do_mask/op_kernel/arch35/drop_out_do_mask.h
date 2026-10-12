/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_DO_MASK_H
#define DROP_OUT_DO_MASK_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace DropOutDoMask {
using namespace Ops::Base;

const uint32_t DOUBLE_BUFFER = 2;

template <typename T>
class DropOutDoMaskImpl {
public:
    __aicore__ inline DropOutDoMaskImpl(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
                                const DropOutDoMaskForAscendCTilingData* tilingData, AscendC::TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void doMaskB32(AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf_n,
                                     AscendC::Reg::RegTensor<T> xInputVf, AscendC::Reg::RegTensor<T> yOutputVf,
                                     AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
                                     __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride,
                                     uint16_t index, uint32_t offset, float prob_reciprocal);
    __aicore__ inline void doMaskB16(AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf_n,
                                     AscendC::Reg::RegTensor<T> xInputVf, AscendC::Reg::RegTensor<T> yOutputVf,
                                     AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
                                     __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride,
                                     uint16_t index, uint32_t offset, float prob_reciprocal,
                                     AscendC::Reg::RegTensor<float> xFp32InputVf,
                                     AscendC::Reg::RegTensor<float> yFp32OutputVf);
    __aicore__ inline void ParseTilingData(const DropOutDoMaskForAscendCTilingData* tilingData);
    __aicore__ inline bool IsProbEqual(float a, float b);

    template <bool COPY_MASK>
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute_Prob0(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void CopyOutProb1(uint32_t loopIdx, uint32_t dataCount);

private:
    constexpr static uint32_t BYTE_BIT_RATIO = 8;
    constexpr static uint32_t MASK_REG_LEN = 32;
    constexpr static uint32_t MASK_LOOP = 4;
    constexpr static uint32_t ALIGN_128 = 128;

    AscendC::TPipe* pipePtr_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> xInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> maskInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> yOutputQueue_;
    AscendC::GlobalTensor<T> xInputGm_;
    AscendC::GlobalTensor<uint8_t> maskInputGm_;
    AscendC::GlobalTensor<T> probInputGm_;
    AscendC::GlobalTensor<T> yOutputGm_;

    float prob_ = 0.0f;
    int32_t ubBlock = GetUbBlockSize();

    uint64_t blockId_ = 0;
    uint64_t currBlockTilingSize_ = 0; // 当前core计算数据总量
    uint64_t ubTailLoopSize_ = 0;      // 当前coreUB尾循环搬运数据量
    uint64_t currLoopCount_ = 0;       // 当前core循环搬运数据次数
    const DropOutDoMaskForAscendCTilingData* tilingDataPtr_;

    static constexpr AscendC::Reg::CastTrait castTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::ParseTilingData(const DropOutDoMaskForAscendCTilingData* tilingData)
{
    blockId_ = AscendC::GetBlockIdx();

    if (blockId_ == tilingDataPtr_->usedCoreNum - 1) {
        currLoopCount_ = tilingDataPtr_->tailBlockLoop;
        currBlockTilingSize_ = tilingDataPtr_->tailBlockData;
        ubTailLoopSize_ = tilingDataPtr_->tailBlockTail;
    } else {
        currLoopCount_ = tilingDataPtr_->normBlockLoop;
        currBlockTilingSize_ = tilingDataPtr_->normBlockData;
        ubTailLoopSize_ = tilingDataPtr_->normBlockTail;
    }
}

template <typename T>
__aicore__ inline bool DropOutDoMaskImpl<T>::IsProbEqual(float a, float b)
{
    return std::abs(a - b) <= tilingDataPtr_->epsilon;
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::doMaskB32(
    AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf_n, AscendC::Reg::RegTensor<T> xInputVf,
    AscendC::Reg::RegTensor<T> yOutputVf, AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
    __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset,
    float prob_reciprocal)
{
    preg0 = AscendC::Reg::UpdateMask<float>(size);
    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
        xInputVf, xInputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride);
    AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xInputVf, xInputVf, prob_reciprocal, preg0);
    AscendC::Reg::Select<float>(yOutputVf, xInputVf, select0, maskInputVf_n);
    AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(
        yOutputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride, yOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::doMaskB16(
    AscendC::Reg::MaskReg& preg0, AscendC::Reg::MaskReg maskInputVf_n, AscendC::Reg::RegTensor<T> xInputVf,
    AscendC::Reg::RegTensor<T> yOutputVf, AscendC::Reg::RegTensor<float> select0, __ubuf__ T* xInputUbPtr,
    __ubuf__ T* yOutputUbPtr, uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset,
    float prob_reciprocal, AscendC::Reg::RegTensor<float> xFp32InputVf, AscendC::Reg::RegTensor<float> yFp32OutputVf)
{
    preg0 = AscendC::Reg::UpdateMask<float>(size);
    AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
        xInputVf, xInputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride);
    AscendC::Reg::Cast<float, T, castTraitB16ToB32>(xFp32InputVf, xInputVf, preg0);
    AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xFp32InputVf, xFp32InputVf, prob_reciprocal,
                                                                           preg0);
    AscendC::Reg::Select<float>(yFp32OutputVf, xFp32InputVf, select0, maskInputVf_n);
    AscendC::Reg::Cast<T, float, castTraitB32ToB16>(yOutputVf, yFp32OutputVf, preg0);
    AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(
        yOutputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride, yOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::Init(GM_ADDR x, GM_ADDR mask, GM_ADDR prob, GM_ADDR y, GM_ADDR workspace,
                                                  const DropOutDoMaskForAscendCTilingData* tilingData,
                                                  AscendC::TPipe* pipeIn)
{
    pipePtr_ = pipeIn;
    tilingDataPtr_ = tilingData;
    ParseTilingData(tilingDataPtr_);
    xInputGm_.SetGlobalBuffer((__gm__ T*)x);
    maskInputGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    probInputGm_.SetGlobalBuffer((__gm__ T*)prob);
    yOutputGm_.SetGlobalBuffer((__gm__ T*)y);

    pipePtr_->InitBuffer(xInputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor * sizeof(T));
    pipePtr_->InitBuffer(maskInputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor / BYTE_BIT_RATIO * sizeof(uint8_t));
    pipePtr_->InitBuffer(yOutputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor * sizeof(T));

    if constexpr (AscendC::IsSameType<T, float>::value) {
        prob_ = probInputGm_.GetValue(0);
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        prob_ = static_cast<float>(probInputGm_.GetValue(0));
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        prob_ = AscendC::ToFloat(probInputGm_.GetValue(0));
    }
}

template <typename T>
template <bool COPY_MASK>
__aicore__ inline void DropOutDoMaskImpl<T>::CopyIn(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.AllocTensor<T>();
    AscendC::DataCopyExtParams xCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> xPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(xInputUb_,
                         xInputGm_[blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor],
                         xCopyParams, xPadParams);
    xInputQueue_.EnQue<T>(xInputUb_);

    event_t event_MTE2_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_MTE2_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_MTE2_MTE3);

    if constexpr (COPY_MASK) {
        AscendC::LocalTensor<uint8_t> maskInputUb_ = maskInputQueue_.AllocTensor<uint8_t>();
        AscendC::DataCopyExtParams maskCopyParams{
            1, (uint32_t)(CeilAlign(dataCount, ALIGN_128) / BYTE_BIT_RATIO * sizeof(uint8_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint8_t> maskPadParams{false, 0, 0, 0};
        AscendC::DataCopyPad(
            maskInputUb_,
            maskInputGm_[(blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor) /
                         BYTE_BIT_RATIO],
            maskCopyParams, maskPadParams);
        maskInputQueue_.EnQue<uint8_t>(maskInputUb_);
    }
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::Compute_Prob0(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.AllocTensor<T>();
    AscendC::Duplicate(yOutputUb_, static_cast<T>(0), dataCount);
    yOutputQueue_.EnQue<T>(yOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::Compute(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.DeQue<T>();
    AscendC::LocalTensor<uint8_t> maskInputUb_ = maskInputQueue_.DeQue<uint8_t>();
    AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.AllocTensor<T>();

    constexpr uint16_t vRegSize = GetVRegSize();
    uint16_t vfLoopNum = CeilDiv(static_cast<uint32_t>(dataCount * sizeof(float)), static_cast<uint32_t>(vRegSize));
    uint16_t vfLoopNumMask = CeilDiv(static_cast<uint32_t>(vfLoopNum), MASK_LOOP);
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
        AscendC::Reg::MaskReg maskInputVf;
        AscendC::Reg::MaskReg maskInputVf1;
        AscendC::Reg::MaskReg maskInputVf2;
        AscendC::Reg::MaskReg maskInputVf3;
        AscendC::Reg::MaskReg maskInputVf4;

        AscendC::Reg::Duplicate(select0, 0.0f);

        if constexpr (AscendC::IsSameType<T, float>::value) {
            for (uint16_t i = 0; i < vfLoopNumMask; i++) {
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(maskInputVf,
                                                                                   maskInputUbPtr + i * MASK_REG_LEN);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf3, maskInputVf);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf1, maskInputVf3);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf2, maskInputVf3);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf3, maskInputVf4);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf4);

                doMaskB32(preg0, maskInputVf1, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 0, prob_reciprocal);
                doMaskB32(preg0, maskInputVf2, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 1, prob_reciprocal);
                doMaskB32(preg0, maskInputVf3, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 2, prob_reciprocal);
                doMaskB32(preg0, maskInputVf4, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 3, prob_reciprocal);
            }
        } else {
            for (uint16_t i = 0; i < vfLoopNumMask; i++) {
                AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::MaskDist::DIST_NORM>(maskInputVf,
                                                                                   maskInputUbPtr + i * MASK_REG_LEN);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf3, maskInputVf);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf1, maskInputVf3);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf2, maskInputVf3);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::LOWEST>(maskInputVf3, maskInputVf4);
                AscendC::Reg::MaskUnPack<AscendC::Reg::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf4);

                doMaskB16(preg0, maskInputVf1, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 0, prob_reciprocal, xFp32InputVf, yFp32OutputVf);
                doMaskB16(preg0, maskInputVf2, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 1, prob_reciprocal, xFp32InputVf, yFp32OutputVf);
                doMaskB16(preg0, maskInputVf3, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 2, prob_reciprocal, xFp32InputVf, yFp32OutputVf);
                doMaskB16(preg0, maskInputVf4, xInputVf, yOutputVf, select0, xInputUbPtr, yOutputUbPtr, size,
                          postUpdateStride, i, 3, prob_reciprocal, xFp32InputVf, yFp32OutputVf);
            }
        }
    }

    xInputQueue_.FreeTensor(xInputUb_);
    maskInputQueue_.FreeTensor(maskInputUb_);
    yOutputQueue_.EnQue<T>(yOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::CopyOut(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> yOutputUb_ = yOutputQueue_.DeQue<T>();
    AscendC::DataCopyExtParams yCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(yOutputGm_[blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor],
                         yOutputUb_, yCopyParams);
    yOutputQueue_.FreeTensor(yOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::CopyOutProb1(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> xInputUb_ = xInputQueue_.DeQue<T>();
    AscendC::DataCopyExtParams yCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(yOutputGm_[blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor],
                         xInputUb_, yCopyParams);
    xInputQueue_.FreeTensor(xInputUb_);
}

template <typename T>
__aicore__ inline void DropOutDoMaskImpl<T>::Process()
{
    if (blockId_ >= tilingDataPtr_->usedCoreNum) {
        return;
    }

    uint32_t dataCount = 0;

    if (IsProbEqual(prob_, 0.0f)) {
        for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
            dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                      static_cast<uint32_t>(tilingDataPtr_->ubFactor);
            Compute_Prob0(idx, dataCount);
            CopyOut(idx, dataCount);
        }
        return;

    } else if (IsProbEqual(prob_, 1.0f)) {
        for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
            dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                      static_cast<uint32_t>(tilingDataPtr_->ubFactor);
            CopyIn<false>(idx, dataCount);
            CopyOutProb1(idx, dataCount);
        }
        return;
    }

    for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
        dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                  static_cast<uint32_t>(tilingDataPtr_->ubFactor);
        CopyIn<true>(idx, dataCount);
        Compute(idx, dataCount);
        CopyOut(idx, dataCount);
    }
}

} // namespace DropOutDoMask

#endif // DROP_OUT_DO_MASK_H
