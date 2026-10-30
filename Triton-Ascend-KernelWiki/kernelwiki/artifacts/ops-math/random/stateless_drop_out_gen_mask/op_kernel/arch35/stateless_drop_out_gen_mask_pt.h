/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stateless_drop_out_gen_mask_pt.h
 * \brief
 */
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

#ifndef STATELESS_DROP_OUT_GEN_MASK_PT_H
#define STATELESS_DROP_OUT_GEN_MASK_PT_H
#pragma once

namespace StatelessDropOutGenMask {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr static uint32_t RESULT_ELEMENT_CNT = 4;
constexpr static float CURAND_2POW32_INV = 2.3283064365386963e-10;
constexpr static float COEFF_2POW32_INV = 2.0f;
constexpr static uint32_t gainCoeff = 2;
constexpr static uint32_t byteBitRatio = 8;
constexpr static uint32_t RoundUpByte256 = 256;
constexpr static int64_t BUFFER_NUM = 2;

template <typename T>
class StatelessDropOutGenMaskPt : public RandomKernelBaseOp {
public:
    __aicore__ inline StatelessDropOutGenMaskPt(TPipe* pipe, const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset, GM_ADDR y);
    __aicore__ inline void Process();

protected:
    template <typename TK>
    __aicore__ inline TK AlignUp256(TK param)
    {
        return (param + RoundUpByte256 - 1) / RoundUpByte256 * RoundUpByte256;
    };

private:
    __aicore__ inline void Uint32ToFloat(uint32_t calCount);
    __aicore__ inline void CompareMask(uint32_t calCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t calCount);

private:
    TPipe* pipe_;
    GlobalTensor<T> probInputGm_;
    GlobalTensor<uint8_t> outputGm_;
    GlobalTensor<uint64_t> counterGm_; // offset tensor 的 GM 访问句柄

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> calcuDataBuf_;
    TBuf<QuePosition::VECCALC> tempDataBuf_;

    float prob_ = 0.0f;
    uint32_t curUbProNum = 0;
    uint32_t blockOffset_ = 0;
    uint32_t singleBufferProNum = 0;

    static constexpr Reg::CastTrait castTraitPt = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                   Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Init(GM_ADDR shape, GM_ADDR prob, GM_ADDR seed, GM_ADDR offset,
                                                          GM_ADDR y)
{
    // VarsInit 从 tiling_ 拷贝 key/counter（此时 tiling 中 key 存元数据，counter 已置零）
    VarsInit();

    // 读取 tiling 编码的元数据：key_[0]=offsetElemCount, key_[1]=seedByteSize
    uint32_t offsetElemCount = key_[0];
    uint32_t seedByteSize = key_[1];

    // 从 GM 直接读取 seed → key_[]
    constexpr uint32_t SHIFT_BITS = 32;
    int64_t keyVal = 0;
    if (seedByteSize == sizeof(int32_t)) {
        GlobalTensor<int32_t> seedGm;
        seedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seed), 1);
        keyVal = static_cast<int64_t>(seedGm(0));
    } else {
        GlobalTensor<int64_t> seedGm;
        seedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(seed), 1);
        keyVal = seedGm(0);
    }
    key_[0] = static_cast<uint32_t>(keyVal); // 截断取低32位，符号位不影响
    // 先转 uint64_t 再右移：keyVal 是有符号 int64_t，负数直接右移是算术右移（高位补1），
    // 转成无符号后右移才是逻辑右移（高位补0），保证拿到原始数据的高32位
    key_[1] = static_cast<uint32_t>(static_cast<uint64_t>(keyVal) >> SHIFT_BITS);

    // 从 GM 直接读取 offset → counter_[]
    uint64_t counterVal0 = 0;
    uint64_t counterVal1 = 0;
    if (offsetElemCount > 0) {
        counterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(offset), offsetElemCount);
        if (offsetElemCount == 1) {
            counterVal1 = counterGm_(0);
        } else {
            counterVal0 = counterGm_(0);
            counterVal1 = counterGm_(1);
        }
    }
    counter_[0] = static_cast<uint32_t>(counterVal0);
    counter_[1] = static_cast<uint32_t>(counterVal0 >> SHIFT_BITS);
    counter_[2] = static_cast<uint32_t>(counterVal1);
    counter_[3] = static_cast<uint32_t>(counterVal1 >> SHIFT_BITS);

    singleBufferProNum = tiling_->singleBufferSize;
    blockOffset_ = blockIdx_ * tiling_->normalCoreProNum;
    if (curCoreProNum_ < singleBufferProNum) {
        singleBufferProNum = curCoreProNum_;
    }

    // SetBuffer
    probInputGm_.SetGlobalBuffer((__gm__ T*)prob);
    outputGm_.SetGlobalBuffer((__gm__ uint8_t*)y + (blockOffset_ / byteBitRatio));

    // InitBuffer
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, singleBufferProNum * sizeof(float));
    pipe_->InitBuffer(philoxQueBuf_, singleBufferProNum * sizeof(uint32_t));
    pipe_->InitBuffer(calcuDataBuf_, singleBufferProNum * sizeof(float));
    pipe_->InitBuffer(tempDataBuf_, singleBufferProNum * sizeof(float));

    // GetValue
    T currProb = probInputGm_.GetValue(0);
    if constexpr (AscendC::IsSameType<T, half>::value) {
        prob_ = static_cast<float>(currProb);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        prob_ = static_cast<T>(currProb);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        prob_ = AscendC::ToFloat(currProb);
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Uint32ToFloat(uint32_t calCount)
{
    /*
    [calculation formula]
    (x * CURAND_2POW32_INV) + (CURAND_2POW32_INV / 2.0f)
    */

    // philox result saved in philoxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int64_t* ubPhilox = (__ubuf__ int64_t*)philoxRes.GetPhyAddr();
    LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    __ubuf__ float* ubOut = (__ubuf__ float*)caluData.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, vfLen);

    __VEC_SCOPE__
    {
        Reg::RegTensor<int64_t> vReg0;
        Reg::RegTensor<float> vReg1;
        Reg::RegTensor<float> vReg2;
        Reg::RegTensor<float> vReg3;
        Reg::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount) * gainCoeff;
        float sReg3 = static_cast<float>(CURAND_2POW32_INV);
        float sReg4 = static_cast<float>(CURAND_2POW32_INV / COEFF_2POW32_INV);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<int32_t>(sReg1);
            Reg::DataCopy<int64_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B32>(
                vReg0, ubPhilox, offset / gainCoeff);
            Reg::Cast<float, int64_t, castTraitPt>(vReg1, vReg0, mask);
            Reg::Muls<float, float, Reg::MaskMergeMode::ZEROING>(vReg2, vReg1, sReg3, mask);
            Reg::Adds<float, float, Reg::MaskMergeMode::ZEROING>(vReg3, vReg2, sReg4, mask);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B64>(ubOut, vReg3,
                                                                                                    offset, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::CompareMask(uint32_t calCount)
{
    LocalTensor<float> caluData = calcuDataBuf_.Get<float>();
    LocalTensor<float> tempData = tempDataBuf_.Get<float>();
    LocalTensor<uint8_t> yOutput = outQueY_.AllocTensor<uint8_t>();

    uint32_t countAlign256 = this->AlignUp256(calCount);
    AscendC::Duplicate(tempData, static_cast<float>(1.0), countAlign256);
    AscendC::Adds(tempData, caluData, static_cast<float>(0.0), calCount);
    AscendC::CompareScalar(yOutput, tempData, static_cast<float>(prob_), CMPMODE::LT, countAlign256);
    outQueY_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Compute(uint32_t loopIdx, uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    AscendC::PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                              calCount);
    Uint32ToFloat(calCount);
    CompareMask(calCount);
}

template <typename T>
__aicore__ inline void StatelessDropOutGenMaskPt<T>::Process()
{
    auto groupCnt = Ops::Base::CeilDiv(blockOffset_, RESULT_ELEMENT_CNT);
    Skip(groupCnt);

    for (uint32_t idx = 0; idx < ubRepeatimes_; idx++) {
        curUbProNum = singleBufferProNum;
        if ((idx == ubRepeatimes_ - 1) && (curCoreProNum_ % singleBufferProNum != 0)) {
            curUbProNum = curCoreProNum_ % singleBufferProNum;
        }
        Compute(idx, curUbProNum);

        LocalTensor<uint8_t> yOutput = outQueY_.DeQue<uint8_t>();
        int64_t yOffset = (idx * singleBufferProNum) / byteBitRatio;
        uint32_t copyLength = static_cast<uint32_t>(Ops::Base::CeilDiv(curUbProNum, byteBitRatio) * sizeof(uint8_t));
        CopyOut(yOutput, outputGm_, 1, copyLength, yOffset);
        outQueY_.FreeTensor(yOutput);

        groupCnt = Ops::Base::CeilDiv(curUbProNum, RESULT_ELEMENT_CNT);
        Skip(groupCnt);
    }
}
} // namespace StatelessDropOutGenMask
#endif // STATELESS_DROP_OUT_GEN_MASK_PT_H
