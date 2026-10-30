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
 * \file stateless_random_normal_v2.h
 * \brief
 */

#ifndef STATELESS_RANDOM_NORMAL_V2_H
#define STATELESS_RANDOM_NORMAL_V2_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace StatelessRandomNormalV2Simd {
using namespace AscendC;

constexpr uint16_t BUFFER_NUM = 2;
constexpr uint16_t BLOCK_SIZE = 32;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
constexpr uint16_t RESULT_ELEMENT_CNT = 4;
constexpr float DOUBLE_MULTIPLE = 2.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t INT32_FLOAT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);

template <typename T>
class StatelessRandomNormalV2 {
public:
    __aicore__ inline StatelessRandomNormalV2(){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter,
                                const StatelessRandomNormalV2TilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const StatelessRandomNormalV2TilingData* __restrict tilingData);
    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void Uint32ToFloat(LocalTensor<float>& yOutputTmp, const uint32_t calCount);
    __aicore__ inline void BoxMullerFloat(LocalTensor<float>& yOutputTmp, const uint32_t calCount);
    __aicore__ inline void BoxMullerMul(LocalTensor<float>& yOutputTmp, const uint32_t calCount);
    __aicore__ inline void DataTypeHandle(const uint32_t calCount);
    __aicore__ inline void CopyOut();
    __aicore__ inline uint32_t ROUND_UP32(const uint32_t x) const;
    __aicore__ inline uint32_t CeilDiv(const uint32_t a, const uint32_t b);

private:
    TPipe* pipe_;

    GlobalTensor<T> outputGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> philoxQueBufY_;
    TBuf<QuePosition::VECCALC> uniformResult_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    uint32_t blockNum_ = 0;
    uint32_t blockTilingSize_ = 0;
    uint32_t tailBlockTilingSize_ = 0;
    uint32_t ubTilingSize_ = 0;
    uint32_t currBlockTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t currOffset_ = 0;
    uint32_t ubLoopCnt_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    GlobalTensor<uint64_t> keyGm_;
    GlobalTensor<uint64_t> counterGm_;
};

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter,
                                                        const StatelessRandomNormalV2TilingData* __restrict tilingData,
                                                        TPipe* pipe)
{
    ParseTilingData(tilingData);

    // 从 GM 直接读取 key/counter，覆盖 ParseTilingData 中的零值
    keyGm_.SetGlobalBuffer((__gm__ uint64_t*)key);
    counterGm_.SetGlobalBuffer((__gm__ uint64_t*)counter);
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

    auto blockIdx = GetBlockIdx();
    blockOffset_ = blockIdx * blockTilingSize_;

    if (blockIdx == blockNum_ - 1) {
        currBlockTilingSize_ = tailBlockTilingSize_;
    } else {
        currBlockTilingSize_ = blockTilingSize_;
    }

    // ubTiling size less than block tiling size
    if (ubTilingSize_ > currBlockTilingSize_) {
        ubTilingSize_ = currBlockTilingSize_;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_ = pipe;
    pipe_->InitBuffer(philoxQueBuf_, ROUND_UP32(ubTilingSize_ * sizeof(float)));
    pipe_->InitBuffer(philoxQueBufY_, ROUND_UP32(ubTilingSize_ * sizeof(float)));
    pipe_->InitBuffer(uniformResult_, ROUND_UP32(ubTilingSize_ * sizeof(float)));
    pipe_->InitBuffer(outQue_, BUFFER_NUM, ROUND_UP32(ubTilingSize_ * sizeof(float)));
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::Process()
{
    auto groupCnt = (blockOffset_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    ubLoopCnt_ = (currBlockTilingSize_ + ubTilingSize_ - 1) / ubTilingSize_;
    for (auto idx = 0; idx < ubLoopCnt_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == ubLoopCnt_ - 1) && (currBlockTilingSize_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = currBlockTilingSize_ % ubTilingSize_;
        }
        currOffset_ = blockOffset_ + idx * ubTilingSize_;
        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        LocalTensor<float> yOutputTmp = uniformResult_.Get<float>();
        uint16_t uniformResCount = CeilDiv(currUbTilingSize_, DOUBLE_UNIFORM_RESULT) * DOUBLE_UNIFORM_RESULT;
        PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         uniformResCount);
        Uint32ToFloat(yOutputTmp, uniformResCount);
        BoxMullerFloat(yOutputTmp, uniformResCount);
        BoxMullerMul(yOutputTmp, uniformResCount);
        DataTypeHandle(currUbTilingSize_);
        CopyOut();
        groupCnt = (currUbTilingSize_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::ParseTilingData(
    const StatelessRandomNormalV2TilingData* __restrict tilingData)
{
    blockNum_ = tilingData->blockNum;
    blockTilingSize_ = tilingData->blockTilingSize;
    tailBlockTilingSize_ = tilingData->tailBlockTilingSize;
    ubTilingSize_ = tilingData->ubTilingSize;
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tilingData->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tilingData->counter[i];
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::Skip(const uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> 32);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[2] == 0) {
            ++counter_[3];
        }
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::Uint32ToFloat(LocalTensor<float>& yOutputTmp,
                                                                 const uint32_t calCount)
{
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)yOutputTmp.GetPhyAddr();

    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t repeatTimes = CeilDiv(calCount, vfLen);

    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vReg0;
        Reg::RegTensor<int32_t> vReg1;
        Reg::RegTensor<int32_t> vReg2;
        Reg::RegTensor<int32_t> vReg3;
        Reg::RegTensor<int32_t> vReg4;
        Reg::RegTensor<float> vReg5;
        Reg::RegTensor<float> vReg6;
        Reg::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint32_t sReg2 = static_cast<uint32_t>(0x7fffffu);
        uint32_t exp = static_cast<uint32_t>(127);
        uint32_t sReg3 = exp << 23;

        Reg::MaskReg maskAll = Reg::CreateMask<int32_t>();
        Reg::Duplicate<int32_t, Reg::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll);
        Reg::Duplicate<int32_t, Reg::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll);

        float sReg4 = static_cast<float>(-1.0);
        int32_t offset = static_cast<int32_t>(vfLen);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sReg1);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(vReg0, ubPhilox,
                                                                                                 offset);
            Reg::And<int32_t, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            Reg::Or<int32_t, Reg::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (Reg::RegTensor<float>&)vReg4;
            Reg::Adds<float, float, Reg::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B32>(ubOut, vReg6,
                                                                                                    offset, mask);
        }
    }
}

/*
 *  Formula: Box-Muller
 *   X = sqrt(-2 * ln(U1)) * cos(2 * PI * U2)
 *   Y = sqrt(-2 * ln(U1)) * sin(2 * PI * U2)
 */
template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::BoxMullerFloat(LocalTensor<float>& yOutputTmp,
                                                                  const uint32_t calCount)
{
    __ubuf__ float* uniformRes = (__ubuf__ float*)yOutputTmp.GetPhyAddr();
    LocalTensor<float> v1Result = philoxQueBuf_.Get<float>();
    LocalTensor<float> u2Result = philoxQueBufY_.Get<float>();
    __ubuf__ float* ubV1Out = (__ubuf__ float*)v1Result.GetPhyAddr();
    __ubuf__ float* ubU2Out = (__ubuf__ float*)u2Result.GetPhyAddr();
    uint32_t repeatTimes = CeilDiv(calCount / DOUBLE_UNIFORM_RESULT, static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));
    uint32_t sreg1 = calCount;

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vreg0;
        Reg::RegTensor<float> vreg1;
        Reg::RegTensor<float> vreg2;
        Reg::RegTensor<float> vreg3;
        Reg::RegTensor<float> vreg4;
        Reg::RegTensor<float> vreg5;
        Reg::RegTensor<float> vreg6;
        Reg::MaskReg mask;

        float epsScalar = 1.0e-7f;
        float doublePiScalar = DOUBLE_MULTIPLE * PI;
        int32_t offset = static_cast<int32_t>(INT32_FLOAT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sreg1);
            Reg::DataCopy<float, Reg::LoadDist::DIST_DINTLV_B32>(vreg0, vreg1,
                                                                 uniformRes + i * offset * DOUBLE_UNIFORM_RESULT);
            Reg::Maxs(vreg2, vreg0, epsScalar, mask);
            Reg::Ln(vreg3, vreg2, mask);
            Reg::Muls(vreg4, vreg1, doublePiScalar, mask);
            Reg::Muls(vreg5, vreg3, -DOUBLE_MULTIPLE, mask);
            Reg::Sqrt(vreg6, vreg5, mask);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(ubV1Out, vreg4, offset, mask);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(ubU2Out, vreg6, offset, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::BoxMullerMul(LocalTensor<float>& yOutputTmp, const uint32_t calCount)
{
    LocalTensor<float> v1Result = philoxQueBuf_.Get<float>();
    LocalTensor<float> u2Result = philoxQueBufY_.Get<float>();
    LocalTensor<float> yOutput = outQue_.AllocTensor<float>();

    Cos<float, false>(yOutputTmp, v1Result);
    Sin<float, false>(yOutput, v1Result);
    uint32_t repeatTimes = CeilDiv(calCount / DOUBLE_UNIFORM_RESULT, static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));

    __ubuf__ float* ubSinResult = (__ubuf__ float*)yOutput.GetPhyAddr();
    __ubuf__ float* ubCosResult = (__ubuf__ float*)yOutputTmp.GetPhyAddr();
    __ubuf__ float* ubU2Result = (__ubuf__ float*)u2Result.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)v1Result.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::RegTensor<float> vreg0;
        Reg::RegTensor<float> vreg1;
        Reg::RegTensor<float> vreg2;
        Reg::RegTensor<float> vreg3;
        Reg::RegTensor<float> vreg4;
        Reg::RegTensor<float> vreg5;
        Reg::MaskReg mask;

        uint32_t sreg1 = static_cast<uint32_t>(calCount);
        int32_t offset = static_cast<int32_t>(INT32_FLOAT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sreg1);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(vreg0, ubSinResult, offset);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(vreg1, ubCosResult, offset);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(vreg2, ubU2Result, offset);

            Reg::Mul(vreg3, vreg0, vreg2, mask);
            Reg::Mul(vreg4, vreg1, vreg2, mask);

            Reg::DataCopy<int32_t, Reg::StoreDist::DIST_INTLV_B32>(
                reinterpret_cast<__ubuf__ int32_t*>(ubOut + i * offset * DOUBLE_UNIFORM_RESULT),
                (Reg::RegTensor<int32_t>&)(vreg3), (Reg::RegTensor<int32_t>&)(vreg4), mask);
        }
    }
    outQue_.EnQue(yOutput);
    yOutput = outQue_.DeQue<float>();
    outQue_.FreeTensor(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::DataTypeHandle(const uint32_t calCount)
{
    LocalTensor<float> normalFloatResult = philoxQueBuf_.Get<float>();
    LocalTensor<T> yOutput = outQue_.AllocTensor<T>();
    if constexpr (AscendC::IsSameType<T, float>::value) {
        DataCopy(yOutput, normalFloatResult, ROUND_UP32(calCount));
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_NONE, calCount);
    } else {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_RINT, calCount);
    }
    outQue_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomNormalV2<T>::CopyOut()
{
    LocalTensor<T> yOutput = outQue_.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currUbTilingSize_ * sizeof(T));
    DataCopyPad(outputGm_[currOffset_], yOutput, copyParams);
    outQue_.FreeTensor(yOutput);
}

template <typename T>
__aicore__ inline uint32_t StatelessRandomNormalV2<T>::ROUND_UP32(const uint32_t x) const
{
    if (x % BLOCK_SIZE != 0) {
        return (x / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    return x;
}

template <typename T>
__aicore__ inline uint32_t StatelessRandomNormalV2<T>::CeilDiv(const uint32_t a, const uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}
} // namespace StatelessRandomNormalV2Simd

#endif // STATELESS_RANDOM_NORMAL_V2_H
