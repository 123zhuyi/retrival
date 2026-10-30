/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
namespace StatelessRandom {
using namespace AscendC;

template <typename T>
__aicore__ inline T CeilDiv(const T a, const T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T>
class StatelessRandomUniformV2 {
public:
    __aicore__ inline StatelessRandomUniformV2(){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR key, GM_ADDR counter,
                                const StatelessRandomUniformV2TilingData* __restrict tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const StatelessRandomUniformV2TilingData* __restrict tilingData);
    __aicore__ inline uint32_t ROUND_UP32(const uint32_t x) const;
    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void DataTypeHandle(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void Uint32ToFloat(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void Uint16ToHalf(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void Uint16ToBfloat16(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void CopyOut();

private:
    TPipe* pipe;

    static constexpr uint16_t BUFFER_NUM = 2;
    static constexpr uint16_t BLOCK_SIZE = 32;
    static constexpr uint16_t ALG_KEY_SIZE = 2;
    static constexpr uint16_t ALG_COUNTER_SIZE = 4;
    static constexpr uint16_t RESULT_ELEMENT_CNT = 4;
    static constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);

    GlobalTensor<T> outputGm_;
    GlobalTensor<uint64_t> keyGm_;
    GlobalTensor<uint64_t> counterGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;

    uint32_t blockNum_ = 0;
    uint32_t blockTilingSize_ = 0;
    uint32_t currBlockTilingSize_ = 0;
    uint32_t tailBlockTilingSize_ = 0;
    uint32_t ubTilingSize_ = 0;
    uint32_t currUbTilingSize_ = 0;
    uint64_t blockOffSet_ = 0;
    uint64_t currOffSet_ = 0;
    uint32_t ubLoopCnt_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    static constexpr Reg::CastTrait castTraitTf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                   Reg::MaskMergeMode::ZEROING};
};

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::Init(
    GM_ADDR y, GM_ADDR key, GM_ADDR counter, const StatelessRandomUniformV2TilingData* __restrict tilingData,
    TPipe* pipeIn)
{
    ParseTilingData(tilingData);

    // 从 GM 直接读取 key/counter（替代 ParseTilingData 中从 tilingData 读取的零值）
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
    blockOffSet_ = blockTilingSize_ * blockIdx;

    if (blockIdx == blockNum_ - 1) {
        currBlockTilingSize_ = tailBlockTilingSize_;
    } else {
        currBlockTilingSize_ = blockTilingSize_;
    }

    // ubTiling size less than block tiling size
    if (currBlockTilingSize_ < ubTilingSize_) {
        ubTilingSize_ = currBlockTilingSize_;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe = pipeIn;
    pipe->InitBuffer(outQueY_, BUFFER_NUM, ROUND_UP32(ubTilingSize_ * sizeof(T)));
    pipe->InitBuffer(philoxQueBuf_, ROUND_UP32(ubTilingSize_ * sizeof(uint32_t)));
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::Process()
{
    auto groupCnt = (blockOffSet_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    ubLoopCnt_ = (currBlockTilingSize_ + ubTilingSize_ - 1) / ubTilingSize_;
    for (auto idx = 0; idx < ubLoopCnt_; idx++) {
        currUbTilingSize_ = ubTilingSize_;
        if ((idx == ubLoopCnt_ - 1) && (currBlockTilingSize_ % ubTilingSize_ != 0)) {
            currUbTilingSize_ = currBlockTilingSize_ % ubTilingSize_;
        }
        currOffSet_ = blockOffSet_ + idx * ubTilingSize_;
        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        LocalTensor<T> yOutput = outQueY_.AllocTensor<T>();
        PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         currUbTilingSize_);
        DataTypeHandle(yOutput, currUbTilingSize_);
        outQueY_.EnQue(yOutput);
        CopyOut();
        groupCnt = (currUbTilingSize_ + RESULT_ELEMENT_CNT - 1) / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
}

template <typename T>
__aicore__ inline uint32_t StatelessRandomUniformV2<T>::ROUND_UP32(const uint32_t x) const
{
    if (x % BLOCK_SIZE != 0) {
        return (x / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    return x;
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::ParseTilingData(
    const StatelessRandomUniformV2TilingData* __restrict tilingData)
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
__aicore__ inline void StatelessRandomUniformV2<T>::Skip(const uint64_t count)
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
__aicore__ inline void StatelessRandomUniformV2<T>::DataTypeHandle(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Uint16ToHalf(yOutput, calCount);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        Uint32ToFloat(yOutput, calCount);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Uint16ToBfloat16(yOutput, calCount);
    }
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::CopyOut()
{
    LocalTensor<T> yOutput = outQueY_.DeQue<T>();
    __ubuf__ T* ubPhilox = (__ubuf__ T*)yOutput.GetPhyAddr();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currUbTilingSize_ * sizeof(T));
    DataCopyPad(outputGm_[currOffSet_], yOutput, copyParams);
    outQueY_.FreeTensor(yOutput);
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::Uint32ToFloat(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    /*
    |1|_____8____|___________23___________|
    |s|exponent  | mantissa               |

    |_param__|___type___|________value_________|
    |__man___|_uint32_t_|_____x&0x7fffffu______|
    |__exp___|_uint32_t_|_________127__________|
    |__val___|_uint32_t_|___(exp << 23)|man____|
    |_result_|___float__|memcpy(&result,&val..)|
    |return result - 1.0f                      |
    */

    // philox result saved in PhiloxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

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

        uint32_t sReg1 = static_cast<uint32_t>(calCount);  // x
        uint32_t sReg2 = static_cast<uint32_t>(0x7fffffu); // 23 bit mantissa
        uint32_t exp = static_cast<uint32_t>(127);
        uint32_t sReg3 = exp << 23;

        Reg::MaskReg maskAll = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
        Reg::Duplicate<int32_t, Reg::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll); //  1 = 0x7fffffu
        Reg::Duplicate<int32_t, Reg::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll); // 3 = 127 << 23
        float sReg4 = static_cast<float>(-1.0);
        int32_t offSet = static_cast<int32_t>(INT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sReg1);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(vReg0, ubPhilox,
                                                                                                 offSet);
            Reg::And<int32_t, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            Reg::Or<int32_t, Reg::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (Reg::RegTensor<float>&)vReg4;
            Reg::Adds<float, float, Reg::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B32>(ubOut, vReg6,
                                                                                                    offSet, mask);
        }
    }
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::Uint16ToHalf(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    /*
      |1|____5____|__________10___________|
      |s|exponent  | mantissa             |

      |_param__|___type___|________value_________|
      |__man___|_uint16_t_|_______x&0x3ffu_______|
      |__exp___|_uint16_t_|__________15__________|
      |__val___|_uint16_t_|____(exp<<10)|man_____|
      |_result_|__float16_|memcpy(&result,&val..)|
      |return result - half(1.0)                 |
    */

    // philox result saved in PhiloxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ half* ubOut = (__ubuf__ half*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    set_ctrl(sbitset0(get_ctrl(), 60));
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vRegT;
        Reg::RegTensor<int16_t> vReg0;
        Reg::RegTensor<int16_t> vReg1;
        Reg::RegTensor<int16_t> vReg2;
        Reg::RegTensor<int16_t> vReg3;
        Reg::RegTensor<int16_t> vReg4;
        Reg::RegTensor<half> vReg5;
        Reg::RegTensor<half> vReg6;
        Reg::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint16_t sReg2 = static_cast<uint16_t>(0x3ffu);
        uint16_t exp = static_cast<uint16_t>(15);
        uint16_t sReg3 = exp << 10;

        Reg::MaskReg maskAll = Reg::CreateMask<int32_t>();
        Reg::Duplicate<int16_t, Reg::MaskMergeMode::ZEROING>(vReg1, sReg2, maskAll);
        Reg::Duplicate<int16_t, Reg::MaskMergeMode::ZEROING>(vReg3, sReg3, maskAll);

        half sReg4 = static_cast<half>(-1.0);
        int32_t offset = static_cast<int32_t>(INT32_ONE_REPEAT);

        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sReg1);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(vRegT, ubPhilox,
                                                                                                 offset);
            Reg::Cast<int16_t, int32_t, castTraitTf>(vReg0, vRegT, mask);
            Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            Reg::Or<int16_t, Reg::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (Reg::RegTensor<half>&)vReg4;
            Reg::Adds<half, half, Reg::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            Reg::DataCopy<half, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(ubOut, vReg6, offset,
                                                                                                   mask);
        }
    }

    set_ctrl(sbitset1(get_ctrl(), 60));
}

template <typename T>
__aicore__ inline void StatelessRandomUniformV2<T>::Uint16ToBfloat16(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    /*
      |1|______8______|_____7_____|
      |s|exponent     | mantissa   |

      |_param__|____type____|_________value________|
      |__man___|__uint16_t__|_________x&0x7fu______|
      |__exp___|__uint16_t__|__________127_________|
      |__val___|__uint16_t__|_____(exp<<7)|man_____|
      |_result_|__bfloat16__|memcpy(&result,&val..)|
      |return result - bfloat16(1.0)               |
    */

    // philox result saved in PhiloxQueBuf
    LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ bfloat16_t* ubOut = (__ubuf__ bfloat16_t*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    set_ctrl(sbitset0(get_ctrl(), 60));
    __VEC_SCOPE__
    {
        Reg::RegTensor<int32_t> vRegT;
        Reg::RegTensor<int16_t> vReg0;
        Reg::RegTensor<int16_t> vReg1;
        Reg::RegTensor<int16_t> vReg2;
        Reg::RegTensor<int16_t> vReg3;
        Reg::RegTensor<int16_t> vReg4;
        Reg::RegTensor<bfloat16_t> vReg5;
        Reg::RegTensor<bfloat16_t> vReg6;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        uint16_t sReg2 = static_cast<uint16_t>(0x7fu); // 7 bit mantissa
        uint16_t exp = static_cast<uint16_t>(127);
        uint16_t sReg3 = exp << 7;

        Reg::MaskReg maskAll = Reg::CreateMask<half, Reg::MaskPattern::ALL>();
        Reg::Duplicate(vReg1, sReg2, maskAll);
        Reg::Duplicate(vReg3, sReg3, maskAll);
        bfloat16_t sReg4 = static_cast<bfloat16_t>(-1.0);
        Reg::MaskReg mask;
        int32_t offSet = static_cast<int32_t>(INT32_ONE_REPEAT);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<float>(sReg1);
            Reg::DataCopy<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(vRegT, ubPhilox,
                                                                                                 offSet);
            Reg::Cast<int16_t, int32_t, castTraitTf>(vReg0, vRegT, mask);
            Reg::And<int16_t, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg1, mask);
            Reg::Or<int16_t, Reg::MaskMergeMode::ZEROING>(vReg4, vReg2, vReg3, mask);
            vReg5 = (Reg::RegTensor<bfloat16_t>&)vReg4;
            Reg::Adds<bfloat16_t, bfloat16_t, Reg::MaskMergeMode::ZEROING>(vReg6, vReg5, sReg4, mask);
            Reg::DataCopy<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B32>(ubOut, vReg6,
                                                                                                         offSet, mask);
        }
    }
    set_ctrl(sbitset1(get_ctrl(), 60));
}
} // namespace StatelessRandom
