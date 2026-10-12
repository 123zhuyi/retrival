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
 * \file stateless_random_normal_v3.h
 * \brief
 */

#ifndef STATELESS_RANDOM_NORMAL_V3_H
#define STATELESS_RANDOM_NORMAL_V3_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace StatelessRandomNormalV3Simd {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr uint16_t BUFFER_NUM = 2;
constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
constexpr uint16_t RESULT_ELEMENT_CNT = 4;
static constexpr uint16_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();

template <typename T>
class StatelessRandomNormalV3 : public RandomKernelBaseOp {
public:
    __aicore__ inline StatelessRandomNormalV3(TPipe* pipeIn, const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : RandomKernelBaseOp(tilingData), pipe_(pipeIn){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter, GM_ADDR mean, GM_ADDR stdev);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInMeanStdev(const uint32_t calCount);
    __aicore__ static inline T ScalarCastToT(float val);
    __aicore__ inline void ApplyMeanStdevBothTensor(const uint32_t calCount);
    __aicore__ inline void ApplyMeanStdevMeanScalar(const uint32_t calCount);
    __aicore__ inline void ApplyMeanStdevStdevScalar(const uint32_t calCount);
    __aicore__ inline void ApplyMeanStdevBothScalar(const uint32_t calCount);

private:
    TPipe* pipe_;

    static constexpr uint32_t MEAN_SCALAR_FLAG = 1;
    static constexpr uint32_t STDEV_SCALAR_FLAG = 2;

    GlobalTensor<T> outputGm_;
    GlobalTensor<uint64_t> keyGm_;
    GlobalTensor<uint64_t> counterGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> stdevGm_;

    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> philoxQueBufY_;
    TBuf<QuePosition::VECCALC> uniformResult_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;
    TQue<QuePosition::VECIN, 1> meanInQue_;
    TQue<QuePosition::VECIN, 1> stdevInQue_;

    uint32_t ubTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t currOffset_ = 0;
    bool meanIsTensor_ = false;
    bool stdevIsTensor_ = false;
    float meanScalar_ = 0.0f;
    float stdevScalar_ = 0.0f;

    static constexpr AscendC::Reg::CastTrait castTraitB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
};

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter, GM_ADDR mean,
                                                        GM_ADDR stdev)
{
    VarsInit();
    keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(key), 1);
    counterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(counter), 2);
    uint64_t keyVal = keyGm_.GetValue(0);
    uint64_t counterVal0 = counterGm_.GetValue(0);
    uint64_t counterVal1 = counterGm_.GetValue(1);
    constexpr uint32_t SHIFT_BITS = 32;

    key_[0] = static_cast<uint32_t>(keyVal);
    key_[1] = static_cast<uint32_t>(keyVal >> SHIFT_BITS);
    counter_[0] = static_cast<uint32_t>(counterVal0);
    counter_[1] = static_cast<uint32_t>(counterVal0 >> SHIFT_BITS);
    counter_[2] = static_cast<uint32_t>(counterVal1);
    counter_[3] = static_cast<uint32_t>(counterVal1 >> SHIFT_BITS);

    blockOffset_ = static_cast<int64_t>(tiling_->normalCoreProNum) * blockIdx_;
    ubTilingSize_ = tiling_->singleBufferSize;

    if (ubTilingSize_ > curCoreProNum_) {
        ubTilingSize_ = curCoreProNum_;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    meanGm_.SetGlobalBuffer((__gm__ float*)mean);
    stdevGm_.SetGlobalBuffer((__gm__ float*)stdev);

    uint32_t bufSize = Ops::Base::CeilAlign(ubTilingSize_ * static_cast<uint32_t>(sizeof(float)),
                                            static_cast<uint32_t>(BLOCK_SIZE));
    pipe_->InitBuffer(philoxQueBuf_, bufSize);
    pipe_->InitBuffer(philoxQueBufY_, bufSize);
    pipe_->InitBuffer(uniformResult_, bufSize);
    pipe_->InitBuffer(outQue_, BUFFER_NUM, bufSize);
    meanIsTensor_ = !(tiling_->v3KernelMode & MEAN_SCALAR_FLAG);
    stdevIsTensor_ = !(tiling_->v3KernelMode & STDEV_SCALAR_FLAG);

    if (meanIsTensor_) {
        pipe_->InitBuffer(meanInQue_, 1, bufSize);
    } else {
        meanScalar_ = meanGm_.GetValue(0);
    }
    if (stdevIsTensor_) {
        pipe_->InitBuffer(stdevInQue_, 1, bufSize);
    } else {
        stdevScalar_ = stdevGm_.GetValue(0);
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::Process()
{
    auto groupCnt = (blockOffset_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    for (auto idx = 0; idx < ubRepeatimes_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == ubRepeatimes_ - 1) && (curCoreProNum_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = curCoreProNum_ % ubTilingSize_;
        }
        currOffset_ = blockOffset_ + idx * ubTilingSize_;
        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        LocalTensor<float> yOutputTmp = uniformResult_.Get<float>();
        uint32_t uniformResCount = Ops::Base::CeilDiv(currUbTilingSize_, static_cast<uint32_t>(DOUBLE_UNIFORM_RESULT)) *
                                   DOUBLE_UNIFORM_RESULT;
        GenRandomSIMD(philoxRes, uniformResCount);
        Uint32ToFloat(yOutputTmp, philoxRes, uniformResCount);

        LocalTensor<float> v1Result = philoxQueBuf_.Get<float>();
        LocalTensor<float> u2Result = philoxQueBufY_.Get<float>();
        BoxMullerFloatSIMD<T>(yOutputTmp, v1Result, u2Result, uniformResCount);

        LocalTensor<float> yOutput = outQue_.AllocTensor<float>();
        BoxMullerMulSIMD<T>(yOutputTmp, v1Result, u2Result, yOutput, uniformResCount);
        outQue_.EnQue(yOutput);
        yOutput = outQue_.DeQue<float>();
        outQue_.FreeTensor(yOutput);

        LocalTensor<float> normalFloatResult = philoxQueBuf_.Get<float>();
        LocalTensor<T> typedOutput = outQue_.AllocTensor<T>();
        Float32Conversion(typedOutput, normalFloatResult, currUbTilingSize_);
        outQue_.EnQue(typedOutput);
        PipeBarrier<PIPE_ALL>();
        CopyInMeanStdev(currUbTilingSize_);

        if (meanIsTensor_ && stdevIsTensor_) {
            ApplyMeanStdevBothTensor(currUbTilingSize_);
        } else if (meanIsTensor_) {
            ApplyMeanStdevStdevScalar(currUbTilingSize_);
        } else if (stdevIsTensor_) {
            ApplyMeanStdevMeanScalar(currUbTilingSize_);
        } else {
            ApplyMeanStdevBothScalar(currUbTilingSize_);
        }

        LocalTensor<T> copyOutput = outQue_.DeQue<T>();
        CopyOut(copyOutput, outputGm_, static_cast<uint32_t>(1), static_cast<uint32_t>(currUbTilingSize_ * sizeof(T)),
                currOffset_);
        outQue_.FreeTensor(copyOutput);

        groupCnt = (currUbTilingSize_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::CopyInMeanStdev(const uint32_t calCount)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(calCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    if (meanIsTensor_) {
        LocalTensor<float> meanLocal = meanInQue_.AllocTensor<float>();
        DataCopyPad(meanLocal, meanGm_[currOffset_], copyParams, padParams);
        meanInQue_.EnQue(meanLocal);
    }
    if (stdevIsTensor_) {
        LocalTensor<float> stdevLocal = stdevInQue_.AllocTensor<float>();
        DataCopyPad(stdevLocal, stdevGm_[currOffset_], copyParams, padParams);
        stdevInQue_.EnQue(stdevLocal);
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::ApplyMeanStdevBothTensor(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    LocalTensor<float> meanLocal = meanInQue_.DeQue<float>();
    LocalTensor<float> stdevLocal = stdevInQue_.DeQue<float>();

    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal.GetPhyAddr();
    __local_mem__ float* stdevAddr = (__local_mem__ float*)stdevLocal.GetPhyAddr();
    uint32_t loopsize = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t looptimes = Ops::Base::CeilDiv(calCount, loopsize);
    uint32_t Count = calCount;
    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            Reg::RegTensor<float> vRegY, vRegStdev, vRegMean, vRegTmp, vRegResult;
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<T>(Count);
                AscendC::Reg::AddrReg aReg = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy(vRegY, yOutputAddr, aReg);
                Reg::DataCopy(vRegStdev, stdevAddr, aReg);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::DataCopy(vRegMean, meanAddr, aReg);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy(yOutputAddr, vRegResult, aReg, curMask);
            }
        } else {
            Reg::RegTensor<T> vRegY, vRegStdev, vRegMean, vRegTmp, vRegResult;
            Reg::RegTensor<float> vRegStdevF, vRegMeanF;
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aRegT = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::AddrReg aRegF = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vRegY, yOutputAddr, aRegT);
                Reg::DataCopy(vRegStdevF, stdevAddr, aRegF);
                Reg::Cast<T, float, castTraitB32ToB16>(vRegStdev, vRegStdevF, curMask);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::DataCopy(vRegMeanF, meanAddr, aRegF);
                Reg::Cast<T, float, castTraitB32ToB16>(vRegMean, vRegMeanF, curMask);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vRegResult, aRegT, curMask);
            }
        }
    }
    meanInQue_.FreeTensor(meanLocal);
    stdevInQue_.FreeTensor(stdevLocal);
    outQue_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::ApplyMeanStdevMeanScalar(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    LocalTensor<float> stdevLocal = stdevInQue_.DeQue<float>();

    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();
    __local_mem__ float* stdevAddr = (__local_mem__ float*)stdevLocal.GetPhyAddr();

    T scaleMean = ScalarCastToT(meanScalar_);
    uint32_t loopsize = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t looptimes = Ops::Base::CeilDiv(calCount, loopsize);
    uint32_t Count = calCount;

    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            Reg::RegTensor<float> vRegY, vRegStdev, vRegTmp, vRegResult, vRegMean;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vRegMean, scaleMean, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aReg = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy(vRegY, yOutputAddr, aReg);
                Reg::DataCopy(vRegStdev, stdevAddr, aReg);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy(yOutputAddr, vRegResult, aReg, curMask);
            }
        } else {
            Reg::RegTensor<T> vRegY, vRegStdev, vRegTmp, vRegResult, vRegMean;
            Reg::RegTensor<float> vRegStdevF;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vRegMean, scaleMean, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aRegT = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::AddrReg aRegF = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vRegY, yOutputAddr, aRegT);
                Reg::DataCopy(vRegStdevF, stdevAddr, aRegF);
                Reg::Cast<T, float, castTraitB32ToB16>(vRegStdev, vRegStdevF, curMask);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vRegResult, aRegT, curMask);
            }
        }
    }
    stdevInQue_.FreeTensor(stdevLocal);
    outQue_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::ApplyMeanStdevStdevScalar(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    LocalTensor<float> meanLocal = meanInQue_.DeQue<float>();

    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();
    __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal.GetPhyAddr();

    T scaleStdev = ScalarCastToT(stdevScalar_);
    uint32_t loopsize = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t looptimes = Ops::Base::CeilDiv(calCount, loopsize);
    uint32_t Count = calCount;

    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            Reg::RegTensor<float> vRegY, vRegMean, vRegTmp, vRegResult, vRegStdev;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vRegStdev, scaleStdev, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aReg = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy(vRegY, yOutputAddr, aReg);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::DataCopy(vRegMean, meanAddr, aReg);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy(yOutputAddr, vRegResult, aReg, curMask);
            }
        } else {
            Reg::RegTensor<T> vRegY, vRegMean, vRegTmp, vRegResult, vRegStdev;
            Reg::RegTensor<float> vRegMeanF;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vRegStdev, scaleStdev, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aRegT = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                AscendC::Reg::AddrReg aRegF = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vRegY, yOutputAddr, aRegT);
                Reg::Mul(vRegTmp, vRegY, vRegStdev, curMask);
                Reg::DataCopy(vRegMeanF, meanAddr, aRegF);
                Reg::Cast<T, float, castTraitB32ToB16>(vRegMean, vRegMeanF, curMask);
                Reg::Add(vRegResult, vRegTmp, vRegMean, curMask);
                Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vRegResult, aRegT, curMask);
            }
        }
    }
    meanInQue_.FreeTensor(meanLocal);
    outQue_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV3<T>::ApplyMeanStdevBothScalar(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    __local_mem__ T* yOutputAddr = (__local_mem__ T*)yOutput.GetPhyAddr();

    T scaleStdev = ScalarCastToT(stdevScalar_);
    T scaleMean = ScalarCastToT(meanScalar_);

    uint32_t loopsize = Ops::Base::GetVRegSize() / sizeof(float);
    uint16_t looptimes = Ops::Base::CeilDiv(calCount, loopsize);
    uint32_t Count = calCount;

    __VEC_SCOPE__
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            Reg::RegTensor<float> vReg0, vReg1, vReg2, vReg3, vRegMean;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vReg1, scaleStdev, maskAll);
            Reg::Duplicate(vRegMean, scaleMean, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aReg = AscendC::Reg::CreateAddrReg<float>(i, loopsize);
                Reg::DataCopy(vReg0, yOutputAddr, aReg);
                Reg::Mul(vReg2, vReg0, vReg1, curMask);
                Reg::Add(vReg3, vReg2, vRegMean, curMask);
                Reg::DataCopy(yOutputAddr, vReg3, aReg, curMask);
            }
        } else {
            Reg::RegTensor<T> vReg0, vReg1, vReg2, vReg3, vRegMean;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::Duplicate(vReg1, scaleStdev, maskAll);
            Reg::Duplicate(vRegMean, scaleMean, maskAll);
            for (uint16_t i = 0; i < looptimes; i++) {
                Reg::MaskReg curMask = Reg::UpdateMask<float>(Count);
                AscendC::Reg::AddrReg aRegT = AscendC::Reg::CreateAddrReg<T>(i, loopsize);
                Reg::DataCopy<T, Reg::LoadDist::DIST_UNPACK_B16>(vReg0, yOutputAddr, aRegT);
                Reg::Mul(vReg2, vReg0, vReg1, curMask);
                Reg::Add(vReg3, vReg2, vRegMean, curMask);
                Reg::DataCopy<T, Reg::StoreDist::DIST_PACK_B32>(yOutputAddr, vReg3, aRegT, curMask);
            }
        }
    }
    outQue_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline T StatelessRandomNormalV3<T>::ScalarCastToT(float val)
{
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return Cast(val);
    } else {
        return static_cast<T>(val);
    }
}
} // namespace StatelessRandomNormalV3Simd

#endif // STATELESS_RANDOM_NORMAL_V3_H
