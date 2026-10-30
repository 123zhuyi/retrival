/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace StatelessRandomUniformV3Simd {
using namespace AscendC;

template <typename T>
class StatelessRandomUniformV3 : public RandomKernelBase::RandomKernelBaseOp {
public:
    __aicore__ inline StatelessRandomUniformV3(TPipe* pipeIn,
                                               const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData), pipe_(pipeIn){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter, GM_ADDR from, GM_ADDR to);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ScaleRangeUniform(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void ScaleRangeRandom(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void CopyOut();

private:
    TPipe* pipe_;

    static constexpr uint16_t BUFFER_NUM = 2;
    static constexpr uint16_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    static constexpr uint16_t RESULT_ELEMENT_CNT = 4;

    GlobalTensor<T> outputGm_;
    GlobalTensor<float> fromGm_;
    GlobalTensor<float> toGm_;
    GlobalTensor<uint64_t> keyGm_;
    GlobalTensor<uint64_t> counterGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;

    uint32_t ubTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint64_t blockOffSet_ = 0;
    uint64_t currOffSet_ = 0;
    float from_{0};
    float to_{0};

    static constexpr AscendC::Reg::CastTrait castTraitB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    static constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void StatelessRandomUniformV3Simd::StatelessRandomUniformV3<T>::Init(GM_ADDR y, GM_ADDR key,
                                                                                       GM_ADDR counter, GM_ADDR from,
                                                                                       GM_ADDR to)
{
    // 基类完成：blockIdx_、curCoreProNum_、ubRepeatimes_ 初始化
    // 注意：VarsInit 也会从 tiling_ 拷贝 key/counter，但 tiling 中已置零，
    // 后续会用 GM 读取的值覆盖基类的 key_[] 和 counter_[]
    VarsInit();

    // 从 GM 直接读取 key/counter（替代原来从 tilingData 读取）
    keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(key), 1);
    counterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(counter), 2);
    uint64_t keyVal = keyGm_(0);
    uint64_t counterVal0 = counterGm_(0);
    uint64_t counterVal1 = counterGm_(1);

    constexpr uint32_t SHIFT_BITS = 32;
    key_[0] = static_cast<uint32_t>(keyVal);
    key_[1] = static_cast<uint32_t>(keyVal >> SHIFT_BITS);
    counter_[0] = static_cast<uint32_t>(counterVal0);
    counter_[1] = static_cast<uint32_t>(counterVal0 >> SHIFT_BITS);
    counter_[2] = static_cast<uint32_t>(counterVal1);
    counter_[3] = static_cast<uint32_t>(counterVal1 >> SHIFT_BITS);

    // V3 特有：计算 GM 偏移、UB 切分收缩
    blockOffSet_ = static_cast<int64_t>(tiling_->normalCoreProNum) * blockIdx_;
    ubTilingSize_ = tiling_->singleBufferSize;
    if (curCoreProNum_ < ubTilingSize_) {
        ubTilingSize_ = curCoreProNum_;
    }

    // V3 特有：GM 绑定 + 读取 from/to 标量
    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    fromGm_.SetGlobalBuffer((__gm__ float*)from);
    toGm_.SetGlobalBuffer((__gm__ float*)to);
    from_ = fromGm_(0);
    to_ = toGm_(0);

    // V3 特有：Buffer 初始化
    pipe_->InitBuffer(
        outQueY_, BUFFER_NUM,
        Ops::Base::CeilAlign(ubTilingSize_ * static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(BLOCK_SIZE)));
    pipe_->InitBuffer(philoxQueBuf_, Ops::Base::CeilAlign(ubTilingSize_ * static_cast<uint32_t>(sizeof(uint32_t)),
                                                          static_cast<uint32_t>(BLOCK_SIZE)));
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV3Simd::StatelessRandomUniformV3<T>::Process()
{
    auto groupCnt = (blockOffSet_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    for (auto idx = 0; idx < ubRepeatimes_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == ubRepeatimes_ - 1) && (curCoreProNum_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = curCoreProNum_ % ubTilingSize_;
        }
        currOffSet_ = blockOffSet_ + idx * ubTilingSize_;
        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        LocalTensor<T> yOutput = outQueY_.AllocTensor<T>();
        GenRandomSIMD(philoxRes, currUbTilingSize_);
        RandomKernelBase::U32Conversion(yOutput, philoxRes, currUbTilingSize_);
        if (tiling_->v3KernelMode == 0) {
            ScaleRangeUniform(yOutput, currUbTilingSize_);
        } else {
            ScaleRangeRandom(yOutput, currUbTilingSize_);
        }
        outQueY_.EnQue(yOutput);
        CopyOut();
        groupCnt = (currUbTilingSize_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
}

// 取代 `aclnnInplaceUniform` 中 `V2 → Muls → Add` 3 算子链 result = x * (to_ - from_) + from_
template <typename T>
__aicore__ inline void StatelessRandomUniformV3Simd::StatelessRandomUniformV3<T>::ScaleRangeUniform(
    LocalTensor<T>& yOutput, const uint32_t calCount)
{
    float rangeVal = to_ - from_;
    float offsetVal = from_;
    uint32_t Count = calCount;
    // bf16 经 UNPACK 后占 32-bit 寄存器槽位，loopsize 按 float 计算；float/half 按自身 sizeof 计算
    constexpr uint32_t elemSize = (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) ?
                                      sizeof(T) :
                                      sizeof(float);
    uint32_t loopsize = Ops::Base::GetVRegSize() / elemSize;
    uint16_t looptimes = Ops::Base::CeilDiv(Count, loopsize);
    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();
    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) {
            Reg::RegTensor<T> vReg0;
            Reg::RegTensor<T> vReg1;
            Reg::RegTensor<T> vReg2;
            Reg::MaskReg curMask;
            Reg::AddrReg aReg;
            for (uint16_t i = 0; i < looptimes; i++) {
                curMask = AscendC::Reg::UpdateMask<T>(Count);
                aReg = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::DataCopy(vReg0, yOutputAddr, aReg);
                AscendC::Reg::Muls(vReg1, vReg0, rangeVal, curMask);
                AscendC::Reg::Adds(vReg2, vReg1, offsetVal, curMask);
                AscendC::Reg::DataCopy(yOutputAddr, vReg2, aReg, curMask);
            }
        } else {
            // bfloat16 路径：在 float32 中计算，与 V2 链对 bf16 selfRef 走 float32 路径一致
            Reg::RegTensor<T> vReg0;
            Reg::RegTensor<float> vReg1;
            Reg::RegTensor<float> vReg2;
            Reg::RegTensor<float> xFp32Inputvf;
            Reg::RegTensor<T> vRegOut;
            Reg::MaskReg curMask;
            Reg::AddrReg aReg;
            for (uint16_t i = 0; i < looptimes; i++) {
                curMask = AscendC::Reg::UpdateMask<float>(Count);
                aReg = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vReg0, yOutputAddr, aReg);
                AscendC::Reg::Cast<float, T, castTraitB16ToB32>(xFp32Inputvf, vReg0, curMask);
                AscendC::Reg::Muls(vReg1, xFp32Inputvf, rangeVal, curMask);
                AscendC::Reg::Adds(vReg2, vReg1, offsetVal, curMask);
                AscendC::Reg::Cast<T, float, castTraitB32ToB16>(vRegOut, vReg2, curMask);
                AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vRegOut, aReg, curMask);
            }
        }
    }
}

// 取代 `aclnnInplaceRandom` 中 `V2 → Muls → Sub → Muls → Sub` 5 算子链 result = x*to_ - (x*from_ - from_)
template <typename T>
__aicore__ inline void StatelessRandomUniformV3Simd::StatelessRandomUniformV3<T>::ScaleRangeRandom(
    LocalTensor<T>& yOutput, const uint32_t calCount)
{
    float fromVal = from_;
    float toVal = to_;
    float negFromVal = -from_;
    uint32_t Count = calCount;
    constexpr uint32_t elemSize = (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) ?
                                      sizeof(T) :
                                      sizeof(float);
    uint32_t loopsize = Ops::Base::GetVRegSize() / elemSize;
    uint16_t looptimes = Ops::Base::CeilDiv(Count, loopsize);
    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();

    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, half>::value) {
            Reg::RegTensor<T> vReg0, vReg1, vReg2, vReg3;
            Reg::MaskReg curMask;
            Reg::AddrReg aReg;
            for (uint16_t i = 0; i < looptimes; i++) {
                curMask = AscendC::Reg::UpdateMask<T>(Count);
                aReg = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::DataCopy(vReg0, yOutputAddr, aReg);
                AscendC::Reg::Muls(vReg1, vReg0, fromVal, curMask);
                AscendC::Reg::Adds(vReg2, vReg1, negFromVal, curMask);
                AscendC::Reg::Muls(vReg3, vReg0, toVal, curMask);
                AscendC::Reg::Sub(vReg1, vReg3, vReg2, curMask);
                AscendC::Reg::DataCopy(yOutputAddr, vReg1, aReg, curMask);
            }
        } else {
            Reg::RegTensor<T> vReg0;
            Reg::RegTensor<float> xFp32Inputvf;
            Reg::RegTensor<float> vReg1, vReg2, vReg3;
            Reg::RegTensor<T> vRegOut;
            Reg::MaskReg curMask;
            Reg::AddrReg aReg;
            for (uint16_t i = 0; i < looptimes; i++) {
                curMask = AscendC::Reg::UpdateMask<float>(Count);
                aReg = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vReg0, yOutputAddr, aReg);
                AscendC::Reg::Cast<float, T, castTraitB16ToB32>(xFp32Inputvf, vReg0, curMask);
                AscendC::Reg::Muls(vReg1, xFp32Inputvf, fromVal, curMask);
                AscendC::Reg::Adds(vReg2, vReg1, negFromVal, curMask);
                AscendC::Reg::Muls(vReg3, xFp32Inputvf, toVal, curMask);
                AscendC::Reg::Sub(vReg1, vReg3, vReg2, curMask);
                AscendC::Reg::Cast<T, float, castTraitB32ToB16>(vRegOut, vReg1, curMask);
                AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vRegOut, aReg, curMask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV3Simd::StatelessRandomUniformV3<T>::CopyOut()
{
    LocalTensor<T> yOutput = outQueY_.DeQue<T>();
    RandomKernelBase::CopyOut(yOutput, outputGm_, 1, static_cast<uint32_t>(currUbTilingSize_ * sizeof(T)), currOffSet_);
    outQueY_.FreeTensor(yOutput);
}
} // namespace StatelessRandomUniformV3Simd
