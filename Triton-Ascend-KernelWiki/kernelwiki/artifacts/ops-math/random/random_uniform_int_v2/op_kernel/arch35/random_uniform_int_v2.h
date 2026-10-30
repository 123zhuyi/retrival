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
#include "op_kernel/math_util.h"

#include "simt_api/asc_simt.h"
namespace RandomUniformIntV2 {
using namespace AscendC;

template <typename T>
class RandomUniformIntV2Op {
public:
    __aicore__ inline RandomUniformIntV2Op(TPipe* pipe,
                                           const RandomUniformIntV2TilingData4RegBase* __restrict tilingData)
        : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR offset);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void DataTypeHandle(const uint32_t calCount);
    template <typename U, typename V>
    __aicore__ inline void UintToInt(LocalTensor<T>& yOutput, const uint32_t calCount);
    __aicore__ inline void CopyOut(int64_t yOffset, int64_t yCount);
    __aicore__ inline void offsetCopyOut(int64_t offsetValue);
    __aicore__ inline void InitKeyAndCounter();

private:
    TPipe* pipe_;
    const RandomUniformIntV2TilingData4RegBase* tiling_;

    constexpr static int8_t SAT_POS = 60;
    static constexpr uint16_t BUFFER_NUM = 2;
    static constexpr uint16_t ALG_KEY_SIZE = 2;
    static constexpr uint16_t ALG_COUNTER_SIZE = 4;
    static constexpr uint16_t RESULT_ELEMENT_CNT = 4;
    static constexpr uint16_t INT64_RATIO = 2;
    static constexpr uint64_t K_RESERVEED_PER_OUTPUT = 256;
    static constexpr uint32_t RIGHT_SHIFT = 32;

    GlobalTensor<T> outputGm_;
    GlobalTensor<int64_t> offsetGm_;
    TBuf<QuePosition::VECCALC> philoxQueBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueY_;

    int64_t curCoreProNum_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    uint32_t blockIdx_;
    static constexpr Reg::CastTrait castTraitTf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                   Reg::MaskMergeMode::ZEROING};
};

constexpr uint32_t THREAD_DIM = 512;

template <typename T, typename UINT_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void UintToIntSimt(const uint32_t calCount, T low, UINT_T range,
                                                                          const __ubuf__ UINT_T* philox,
                                                                          __ubuf__ T* yOutput)
{
    for (uint32_t index = threadIdx.x; index < calCount; index = index + blockDim.x) {
        UINT_T randomRes = philox[index];
        UINT_T divRes = randomRes / range;
        UINT_T b = randomRes - divRes * range;
        UINT_T b_div_2 = b >> 1;
        yOutput[index] = low + static_cast<T>(b_div_2) + static_cast<T>(b - b_div_2);
    }
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::Init(GM_ADDR y, GM_ADDR offset)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > tiling_->blockNum) {
        return;
    }

    if (blockIdx_ == tiling_->blockNum - 1) {
        curCoreProNum_ = tiling_->tailCoreProNum;
    } else {
        curCoreProNum_ = tiling_->normalCoreProNum;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)y);
    offsetGm_.SetGlobalBuffer((__gm__ int64_t*)offset);
    pipe_->InitBuffer(outQueY_, BUFFER_NUM, tiling_->singleUbSize * sizeof(T));
    pipe_->InitBuffer(philoxQueBuf_, tiling_->singleUbSize * sizeof(uint32_t));
    pipe_->InitBuffer(offsetBuf_, Ops::Base::GetUbBlockSize());
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::InitKeyAndCounter()
{
    key_[0] = static_cast<uint32_t>(tiling_->seed);
    key_[1] = static_cast<uint32_t>(tiling_->seed >> RIGHT_SHIFT);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(tiling_->seed2);
    counter_[3] = static_cast<uint32_t>(tiling_->seed2 >> RIGHT_SHIFT);
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::Process()
{
    if (blockIdx_ > tiling_->blockNum) {
        return;
    }

    InitKeyAndCounter();
    auto offsetValue = offsetGm_.GetValue(0);
    if (offsetValue > 0) {
        Skip(offsetValue);
    }
    SyncAll();
    if (blockIdx_ == 0) {
        if (offsetValue < 0) {
            offsetValue = 0;
        }
        offsetValue = offsetValue + tiling_->outputSize * K_RESERVEED_PER_OUTPUT;
        offsetCopyOut(offsetValue);
    }
    uint16_t dtypeRatio = 1;
    if constexpr (AscendC::IsSameType<T, int64_t>::value) {
        dtypeRatio = INT64_RATIO;
    }
    auto blockOffSet = tiling_->normalCoreProNum * blockIdx_;
    auto resultElementCnt = RESULT_ELEMENT_CNT / dtypeRatio;
    auto groupCnt = Ops::Base::CeilDiv(blockOffSet, static_cast<int64_t>(resultElementCnt));

    Skip(groupCnt);
    int64_t singleUbEleNum = tiling_->singleUbSize;
    int64_t ubRepeatimes = Ops::Base::CeilDiv(curCoreProNum_, singleUbEleNum);
    for (auto idx = 0; idx < ubRepeatimes; idx++) {
        int64_t curUbEleNum = idx == (ubRepeatimes - 1) ? curCoreProNum_ - (ubRepeatimes - 1) * singleUbEleNum :
                                                          singleUbEleNum;
        int64_t philoxNumPro = curUbEleNum * dtypeRatio;
        int64_t philoxNumOffset = idx * singleUbEleNum;

        LocalTensor<uint32_t> philoxRes = philoxQueBuf_.Get<uint32_t>();
        PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         philoxNumPro);

        DataTypeHandle(curUbEleNum);
        int64_t yOffset = blockOffSet + philoxNumOffset;
        CopyOut(yOffset, curUbEleNum);
        groupCnt = Ops::Base::CeilDiv(curUbEleNum, static_cast<int64_t>(resultElementCnt));
        Skip(groupCnt);
    }
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::Skip(const uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> RIGHT_SHIFT);

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
__aicore__ inline void RandomUniformIntV2Op<T>::DataTypeHandle(const uint32_t calCount)
{
    LocalTensor<T> yOutput = outQueY_.AllocTensor<T>();
    if constexpr (AscendC::IsSameType<T, int32_t>::value) {
        UintToInt<int32_t, uint32_t>(yOutput, calCount);
    } else if constexpr (AscendC::IsSameType<T, int64_t>::value) {
        LocalTensor<uint64_t> philoxRes = philoxQueBuf_.Get<uint64_t>();
        asc_vf_call<UintToIntSimt<int64_t, uint64_t>>(dim3{THREAD_DIM}, calCount, tiling_->lo, tiling_->range,
                                                      (__ubuf__ uint64_t*)philoxRes.GetPhyAddr(),
                                                      (__ubuf__ int64_t*)yOutput.GetPhyAddr());
    }
    outQueY_.EnQue(yOutput);
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::offsetCopyOut(int64_t offsetValue)
{
    LocalTensor<int64_t> offsetOutput = offsetBuf_.Get<int64_t>();
    offsetOutput.SetValue(0, offsetValue);
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(sizeof(int64_t));
    event_t SToMTE3Event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(SToMTE3Event);
    WaitFlag<HardEvent::S_MTE3>(SToMTE3Event);
    DataCopyPad(offsetGm_[0], offsetOutput, copyParams);
}

template <typename T>
__aicore__ inline void RandomUniformIntV2Op<T>::CopyOut(int64_t yOffset, int64_t yCount)
{
    LocalTensor<T> yOutput = outQueY_.DeQue<T>();
    __ubuf__ T* ubPhilox = (__ubuf__ T*)yOutput.GetPhyAddr();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(yCount * sizeof(T));
    DataCopyPad(outputGm_[yOffset], yOutput, copyParams);
    outQueY_.FreeTensor(yOutput);
}

/*
  将uint随机数转化生成[minVal, minVal + range)内整数
  计算逻辑
  1. 对Philox算法生成的随机数取模运算：b = philoxRandom % range
  2. 计算第一步结果/2的结果暂存在b_div_2：b_div_2 = b >> 1
  3. 获得最终输出：res = minVal + (int)b_div_2 + (int)(b-b_div_2)
*/
template <typename T>
template <typename U, typename V>
__aicore__ inline void RandomUniformIntV2Op<T>::UintToInt(LocalTensor<T>& yOutput, const uint32_t calCount)
{
    LocalTensor<V> philoxRes = philoxQueBuf_.Get<V>();
    __ubuf__ V* ubPhilox = (__ubuf__ V*)philoxRes.GetPhyAddr();
    __ubuf__ U* ubOut = (__ubuf__ U*)yOutput.GetPhyAddr();
    U minVal = tiling_->lo;
    V range = tiling_->range;
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(Ops::Base::GetVRegSize() / sizeof(U)));

    SetCtrlSpr<SAT_POS, SAT_POS>(0);
    __VEC_SCOPE__
    {
        Reg::RegTensor<V> vReg0;
        Reg::RegTensor<U> vReg1;
        Reg::RegTensor<V> vReg2;
        Reg::RegTensor<U> vReg3;
        Reg::RegTensor<U> vReg4;
        Reg::RegTensor<U> vReg5;
        Reg::RegTensor<V> vReg6;
        Reg::RegTensor<V> vReg7;
        Reg::RegTensor<V> vReg8;
        Reg::MaskReg mask;

        uint32_t sReg1 = static_cast<uint32_t>(calCount);
        V sReg2 = range;
        U sReg3 = minVal;
        int16_t sReg4 = 1;

        Reg::MaskReg maskAll = Reg::CreateMask<U, Reg::MaskPattern::ALL>();
        Reg::Duplicate<U, Reg::MaskMergeMode::ZEROING>(vReg1, minVal, maskAll);
        Reg::Duplicate<V, Reg::MaskMergeMode::ZEROING>(vReg8, sReg2, maskAll);

        int32_t offSet = static_cast<int32_t>(Ops::Base::GetVRegSize() / sizeof(T));
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTimes); ++i) {
            mask = Reg::UpdateMask<U>(sReg1);
            Reg::DataCopy<V, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(vReg0, ubPhilox, offSet);
            Reg::Div<V, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg8, mask);
            Reg::Mul<V, Reg::MaskMergeMode::ZEROING>(vReg2, vReg2, vReg8, mask);
            Reg::Sub<V, Reg::MaskMergeMode::ZEROING>(vReg2, vReg0, vReg2, mask);
            Reg::ShiftRights<V, int16_t>(vReg6, vReg2, sReg4, mask);
            Reg::Sub<V, Reg::MaskMergeMode::ZEROING>(vReg7, vReg2, vReg6, mask);
            vReg4 = (Reg::RegTensor<U>&)vReg6;
            vReg5 = (Reg::RegTensor<U>&)vReg7;
            Reg::Add(vReg3, vReg1, vReg4, mask);
            Reg::Add(vReg3, vReg3, vReg5, mask);

            Reg::DataCopy<U, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B32>(ubOut, vReg3, offSet,
                                                                                                mask);
        }
    }
    SetCtrlSpr<SAT_POS, SAT_POS>(1);
}

} // namespace RandomUniformIntV2
