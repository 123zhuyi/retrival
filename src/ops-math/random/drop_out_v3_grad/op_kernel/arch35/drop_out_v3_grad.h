/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3_grad.h
 * \brief
 */

#ifndef DROP_OUT_V3_GRAD_H
#define DROP_OUT_V3_GRAD_H

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace DropOutV3Grad {
using namespace Ops::Base;

const uint32_t DOUBLE_BUFFER = 2;

template <typename T>
class DropOutV3GradImpl {
public:
    __aicore__ inline DropOutV3GradImpl(){};
    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR mask, GM_ADDR scale, GM_ADDR grad_x, GM_ADDR workspace,
                                const DropOutV3GradForAscendCTilingData* tilingData, AscendC::TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void doMaskB32(AscendC::MicroAPI::MaskReg& preg0, AscendC::MicroAPI::MaskReg maskInputVf_n,
                                     AscendC::MicroAPI::RegTensor<T> gradYInputVf,
                                     AscendC::MicroAPI::RegTensor<T> gradXOutputVf,
                                     AscendC::MicroAPI::RegTensor<float> select0,
                                     AscendC::MicroAPI::RegTensor<float> onesReg,
                                     AscendC::MicroAPI::RegTensor<float> maskFloatVf, __ubuf__ T* gradYInputUbPtr,
                                     __ubuf__ T* gradXOutputUbPtr, uint32_t& size, uint32_t postUpdateStride,
                                     uint16_t index, uint32_t offset, float scaleVal);
    __aicore__ inline void doMaskB16(
        AscendC::MicroAPI::MaskReg& preg0, AscendC::MicroAPI::MaskReg maskInputVf_n,
        AscendC::MicroAPI::RegTensor<T> gradYInputVf, AscendC::MicroAPI::RegTensor<T> gradXOutputVf,
        AscendC::MicroAPI::RegTensor<float> select0, AscendC::MicroAPI::RegTensor<float> onesReg,
        AscendC::MicroAPI::RegTensor<float> maskFloatVf, __ubuf__ T* gradYInputUbPtr, __ubuf__ T* gradXOutputUbPtr,
        uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset, float scaleVal,
        AscendC::MicroAPI::RegTensor<float> gradYFp32InputVf, AscendC::MicroAPI::RegTensor<float> gradXFp32OutputVf);
    __aicore__ inline void ParseTilingData(const DropOutV3GradForAscendCTilingData* tilingData);
    template <bool COPY_MASK>
    __aicore__ inline void CopyIn(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t loopIdx, uint32_t dataCount);

private:
    constexpr static uint32_t BYTE_BIT_RATIO = 8;
    constexpr static uint32_t MASK_REG_LEN = 32;
    constexpr static uint32_t MASK_LOOP = 4;
    constexpr static uint32_t ALIGN_128 = 128;

    AscendC::TPipe* pipePtr_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> gradYInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> maskInputQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> gradXOutputQueue_;
    AscendC::GlobalTensor<T> gradYInputGm_;
    AscendC::GlobalTensor<uint8_t> maskInputGm_;
    AscendC::GlobalTensor<float> scaleGm_;
    AscendC::GlobalTensor<T> gradXOutputGm_;

    float scale_ = 0.0f;
    int32_t ubBlock = GetUbBlockSize();

    uint64_t blockId_ = 0;
    uint64_t currBlockTilingSize_ = 0; // 当前core计算数据总量
    uint64_t ubTailLoopSize_ = 0;      // 当前coreUB尾循环搬运数据量
    uint64_t currLoopCount_ = 0;       // 当前core循环搬运数据次数
    const DropOutV3GradForAscendCTilingData* tilingDataPtr_;

    static constexpr AscendC::MicroAPI::CastTrait castTraitB16ToB32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitB32ToB16 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void DropOutV3GradImpl<T>::ParseTilingData(const DropOutV3GradForAscendCTilingData* tilingData)
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
__aicore__ inline void DropOutV3GradImpl<T>::doMaskB32(
    AscendC::MicroAPI::MaskReg& preg0, AscendC::MicroAPI::MaskReg maskInputVf_n,
    AscendC::MicroAPI::RegTensor<T> gradYInputVf, AscendC::MicroAPI::RegTensor<T> gradXOutputVf,
    AscendC::MicroAPI::RegTensor<float> select0, AscendC::MicroAPI::RegTensor<float> onesReg,
    AscendC::MicroAPI::RegTensor<float> maskFloatVf, __ubuf__ T* gradYInputUbPtr, __ubuf__ T* gradXOutputUbPtr,
    uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset, float scaleVal)
{
    preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
        gradYInputVf, gradYInputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride);
    // 完全对标 PyTorch native_dropout_backward 的纯乘法链 grad_x = grad_y * mask * scale，
    // 其中 mask 先物化为 0.0f/1.0f 的 float 再参与乘法，保证 inf*0=NaN 的传播行为与竞品一致
    AscendC::MicroAPI::Select<float>(maskFloatVf, onesReg, select0, maskInputVf_n);
    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(gradYInputVf, gradYInputVf, maskFloatVf,
                                                                             preg0);
    AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(gradXOutputVf, gradYInputVf,
                                                                                     scaleVal, preg0);
    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(
        gradXOutputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride, gradXOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutV3GradImpl<T>::doMaskB16(
    AscendC::MicroAPI::MaskReg& preg0, AscendC::MicroAPI::MaskReg maskInputVf_n,
    AscendC::MicroAPI::RegTensor<T> gradYInputVf, AscendC::MicroAPI::RegTensor<T> gradXOutputVf,
    AscendC::MicroAPI::RegTensor<float> select0, AscendC::MicroAPI::RegTensor<float> onesReg,
    AscendC::MicroAPI::RegTensor<float> maskFloatVf, __ubuf__ T* gradYInputUbPtr, __ubuf__ T* gradXOutputUbPtr,
    uint32_t& size, uint32_t postUpdateStride, uint16_t index, uint32_t offset, float scaleVal,
    AscendC::MicroAPI::RegTensor<float> gradYFp32InputVf, AscendC::MicroAPI::RegTensor<float> gradXFp32OutputVf)
{
    preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
        gradYInputVf, gradYInputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride);
    AscendC::MicroAPI::Cast<float, T, castTraitB16ToB32>(gradYFp32InputVf, gradYInputVf, preg0);
    // 完全对标 PyTorch native_dropout_backward 的纯乘法链 grad_x = grad_y * mask * scale，
    // 其中 mask 先物化为 0.0f/1.0f 的 float 再参与乘法，保证 inf*0=NaN 的传播行为与竞品一致
    AscendC::MicroAPI::Select<float>(maskFloatVf, onesReg, select0, maskInputVf_n);
    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(gradYFp32InputVf, gradYFp32InputVf,
                                                                             maskFloatVf, preg0);
    AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(gradYFp32InputVf, gradYFp32InputVf,
                                                                                     scaleVal, preg0);
    AscendC::MicroAPI::Cast<T, float, castTraitB32ToB16>(gradXOutputVf, gradYFp32InputVf, preg0);
    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
        gradXOutputUbPtr + (MASK_LOOP * index + offset) * postUpdateStride, gradXOutputVf, preg0);
}

template <typename T>
__aicore__ inline void DropOutV3GradImpl<T>::Init(GM_ADDR grad_y, GM_ADDR mask, GM_ADDR scale, GM_ADDR grad_x,
                                                  GM_ADDR workspace,
                                                  const DropOutV3GradForAscendCTilingData* tilingData,
                                                  AscendC::TPipe* pipeIn)
{
    pipePtr_ = pipeIn;
    tilingDataPtr_ = tilingData;
    ParseTilingData(tilingDataPtr_);
    gradYInputGm_.SetGlobalBuffer((__gm__ T*)grad_y);
    maskInputGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);
    gradXOutputGm_.SetGlobalBuffer((__gm__ T*)grad_x);

    pipePtr_->InitBuffer(gradYInputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor * sizeof(T));
    pipePtr_->InitBuffer(maskInputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor / BYTE_BIT_RATIO * sizeof(uint8_t));
    pipePtr_->InitBuffer(gradXOutputQueue_, DOUBLE_BUFFER, tilingDataPtr_->ubFactor * sizeof(T));

    scale_ = scaleGm_.GetValue(0);
}

template <typename T>
template <bool COPY_MASK>
__aicore__ inline void DropOutV3GradImpl<T>::CopyIn(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> gradYInputUb_ = gradYInputQueue_.AllocTensor<T>();
    AscendC::DataCopyExtParams gradYCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> gradYPadParams{false, 0, 0, 0};
    AscendC::DataCopyPad(gradYInputUb_,
                         gradYInputGm_[blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor],
                         gradYCopyParams, gradYPadParams);
    gradYInputQueue_.EnQue<T>(gradYInputUb_);

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
__aicore__ inline void DropOutV3GradImpl<T>::Compute(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> gradYInputUb_ = gradYInputQueue_.DeQue<T>();
    AscendC::LocalTensor<uint8_t> maskInputUb_ = maskInputQueue_.DeQue<uint8_t>();
    AscendC::LocalTensor<T> gradXOutputUb_ = gradXOutputQueue_.AllocTensor<T>();

    constexpr uint16_t vRegSize = GetVRegSize();
    uint16_t vfLoopNum = CeilDiv(static_cast<uint32_t>(dataCount * sizeof(float)), static_cast<uint32_t>(vRegSize));
    uint16_t vfLoopNumMask = CeilDiv(static_cast<uint32_t>(vfLoopNum), MASK_LOOP);
    __ubuf__ T* gradYInputUbPtr = (__ubuf__ T*)gradYInputUb_.GetPhyAddr();
    __ubuf__ T* gradXOutputUbPtr = (__ubuf__ T*)gradXOutputUb_.GetPhyAddr();
    __ubuf__ uint8_t* maskInputUbPtr = (__ubuf__ uint8_t*)maskInputUb_.GetPhyAddr();

    uint32_t postUpdateStride = vRegSize / sizeof(float); // 每次regbase计算的数据量
    float scaleVal = scale_;
    uint32_t size = dataCount;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> gradYInputVf;
        AscendC::MicroAPI::RegTensor<float> gradYFp32InputVf;
        AscendC::MicroAPI::RegTensor<T> gradXOutputVf;
        AscendC::MicroAPI::RegTensor<float> gradXFp32OutputVf;
        AscendC::MicroAPI::RegTensor<float> select0;
        AscendC::MicroAPI::RegTensor<float> onesReg;     // 全 1.0f，用于把 mask 物化为 0/1 float
        AscendC::MicroAPI::RegTensor<float> maskFloatVf; // mask 物化后的 0.0f/1.0f float 向量
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg maskInputVf;
        AscendC::MicroAPI::MaskReg maskInputVf1;
        AscendC::MicroAPI::MaskReg maskInputVf2;
        AscendC::MicroAPI::MaskReg maskInputVf3;
        AscendC::MicroAPI::MaskReg maskInputVf4;

        AscendC::MicroAPI::Duplicate(select0, 0.0f);
        AscendC::MicroAPI::Duplicate(onesReg, 1.0f);

        if constexpr (AscendC::IsSameType<T, float>::value) {
            for (uint16_t i = 0; i < vfLoopNumMask; i++) {
                AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::MaskDist::DIST_NORM>(
                    maskInputVf, maskInputUbPtr + i * MASK_REG_LEN);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf3, maskInputVf);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf1, maskInputVf3);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf2, maskInputVf3);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf3, maskInputVf4);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf4);

                doMaskB32(preg0, maskInputVf1, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 0, scaleVal);
                doMaskB32(preg0, maskInputVf2, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 1, scaleVal);
                doMaskB32(preg0, maskInputVf3, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 2, scaleVal);
                doMaskB32(preg0, maskInputVf4, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 3, scaleVal);
            }
        } else {
            for (uint16_t i = 0; i < vfLoopNumMask; i++) {
                AscendC::MicroAPI::DataCopy<uint8_t, AscendC::MicroAPI::MaskDist::DIST_NORM>(
                    maskInputVf, maskInputUbPtr + i * MASK_REG_LEN);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf3, maskInputVf);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf1, maskInputVf3);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf2, maskInputVf3);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskInputVf3, maskInputVf4);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskInputVf4, maskInputVf4);

                doMaskB16(preg0, maskInputVf1, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 0, scaleVal, gradYFp32InputVf,
                          gradXFp32OutputVf);
                doMaskB16(preg0, maskInputVf2, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 1, scaleVal, gradYFp32InputVf,
                          gradXFp32OutputVf);
                doMaskB16(preg0, maskInputVf3, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 2, scaleVal, gradYFp32InputVf,
                          gradXFp32OutputVf);
                doMaskB16(preg0, maskInputVf4, gradYInputVf, gradXOutputVf, select0, onesReg, maskFloatVf,
                          gradYInputUbPtr, gradXOutputUbPtr, size, postUpdateStride, i, 3, scaleVal, gradYFp32InputVf,
                          gradXFp32OutputVf);
            }
        }
    }

    gradYInputQueue_.FreeTensor(gradYInputUb_);
    maskInputQueue_.FreeTensor(maskInputUb_);
    gradXOutputQueue_.EnQue<T>(gradXOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutV3GradImpl<T>::CopyOut(uint32_t loopIdx, uint32_t dataCount)
{
    AscendC::LocalTensor<T> gradXOutputUb_ = gradXOutputQueue_.DeQue<T>();
    AscendC::DataCopyExtParams gradXCopyParams{1, (uint32_t)(dataCount * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(gradXOutputGm_[blockId_ * tilingDataPtr_->normBlockData + loopIdx * tilingDataPtr_->ubFactor],
                         gradXOutputUb_, gradXCopyParams);
    gradXOutputQueue_.FreeTensor(gradXOutputUb_);
}

template <typename T>
__aicore__ inline void DropOutV3GradImpl<T>::Process()
{
    if (blockId_ >= tilingDataPtr_->usedCoreNum) {
        return;
    }

    uint32_t dataCount = 0;

    for (uint32_t idx = 0; idx < currLoopCount_; idx++) {
        dataCount = (idx == currLoopCount_ - 1) ? static_cast<uint32_t>(ubTailLoopSize_) :
                                                  static_cast<uint32_t>(tilingDataPtr_->ubFactor);
        CopyIn<true>(idx, dataCount);
        Compute(idx, dataCount);
        CopyOut(idx, dataCount);
    }
}

} // namespace DropOutV3Grad

#endif // DROP_OUT_V3_GRAD_H
