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
 * \file sim_thread_exponential.h
 * \brief
 */

#ifndef SIM_THREAD_EXPONENTIAL_H
#define SIM_THREAD_EXPONENTIAL_H

#include "kernel_operator.h"
#include <limits>

namespace {
constexpr int32_t MAX_UB_SIZE = 192 * 1024;
constexpr uint32_t PHILOX_W32_0 = 2654435769;
constexpr uint32_t PHILOX_W32_1 = 3144134277;
constexpr uint64_t PHILOX_M4x32_0 = 3528531795;
constexpr uint64_t PHILOX_M4x32_1 = 3449720151;
constexpr float CURAND_2POW32_INV = 2.3283064e-10;
constexpr uint32_t PER_UINT8_8BITS = 8;
constexpr uint32_t REPEAT_NUM_64 = 64;
constexpr float DOUBLE = 2;
constexpr uint32_t UINT_DOUBLE = 2;
constexpr uint32_t UINT_THREE = 3;
constexpr uint32_t UINT_FOUR = 4;
constexpr uint32_t UINT_FIVE = 5;
constexpr uint32_t UINT_EIGHT = 8;
constexpr uint32_t UINT_NINE = 9;
constexpr uint32_t SHIFT_LEFT_16 = 16;
constexpr uint32_t SHIFT_LEFT_32 = 32;
constexpr uint32_t SHIFT_LEFT_31 = 31;
constexpr float EPSILON_ = std::numeric_limits<float>::epsilon();

template <typename T1, typename T2>
__aicore__ inline void myAdd(
    const AscendC::LocalTensor<T1>& zLocal, const AscendC::LocalTensor<T1>& xLocal,
    const AscendC::LocalTensor<T1>& yLocal, uint32_t len)
{
    AscendC::Add<T2>(
        zLocal.template ReinterpretCast<T2>(), xLocal.template ReinterpretCast<T2>(), yLocal.template ReinterpretCast<T2>(), len);
}

__aicore__ inline void myAdd2(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    const AscendC::LocalTensor<int32_t>& yLocal, uint32_t len)
{
    AscendC::Add<int32_t>(
        zLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int32_t>(), yLocal, len);
}

__aicore__ inline void myAdds(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    const uint32_t yScalar, uint32_t len)
{
    AscendC::Adds<int32_t>(
        zLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int32_t>(), yScalar, len);
}

template <typename T1, typename T2>
__aicore__ inline void myMul(
    const AscendC::LocalTensor<T1>& zLocal, const AscendC::LocalTensor<T1>& xLocal,
    const AscendC::LocalTensor<T1>& yLocal, uint32_t len)
{
    AscendC::Mul<T2>(
        zLocal.template ReinterpretCast<T2>(), xLocal.template ReinterpretCast<T2>(), yLocal.template ReinterpretCast<T2>(), len);
}

__aicore__ inline void myMuls(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    const uint32_t yScalar, uint32_t len)
{
    AscendC::Muls<int32_t>(
        zLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int32_t>(), yScalar, len);
}

__aicore__ inline void myEqual(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    const AscendC::LocalTensor<uint32_t>& yLocal, AscendC::TBuf<>& tensorBuf,
    uint32_t len)
{
    uint32_t offset = 0;
    AscendC::LocalTensor<uint32_t> onesLocal = tensorBuf.GetWithOffset<uint32_t>(len, offset);
    offset += len * sizeof(uint32_t);
    AscendC::LocalTensor<uint8_t> maskLocal = tensorBuf.GetWithOffset<uint8_t>(len / UINT_EIGHT / UINT_EIGHT, offset);

    AscendC::Duplicate<uint32_t>(onesLocal, 1, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Compare<float, uint8_t>(maskLocal, xLocal.template ReinterpretCast<float>(), yLocal.template ReinterpretCast<float>(), AscendC::CMPMODE::EQ, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Select<float, uint8_t>(
        zLocal.template ReinterpretCast<float>(), maskLocal,
        onesLocal.template ReinterpretCast<float>(), 0,
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
}

__aicore__ inline void myEqualZeroScalar(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    AscendC::TBuf<>& tensorBuf, uint32_t len)
{
    uint32_t offset = 0;
    AscendC::LocalTensor<uint32_t> onesLocal = tensorBuf.GetWithOffset<uint32_t>(len, offset);
    offset += len * sizeof(uint32_t);
    AscendC::LocalTensor<uint8_t> maskLocal = tensorBuf.GetWithOffset<uint8_t>(len / UINT_EIGHT / UINT_EIGHT, offset);

    AscendC::Duplicate<uint32_t>(onesLocal, 1, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast<float, int32_t>(
        zLocal.template ReinterpretCast<float>(), xLocal.template ReinterpretCast<int32_t>(),
        AscendC::RoundMode::CAST_NONE, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::CompareScalar<float, uint8_t>(maskLocal, zLocal.template ReinterpretCast<float>(), 0.0f, AscendC::CMPMODE::EQ, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Select<float, uint8_t>(
        zLocal.template ReinterpretCast<float>(), maskLocal,
        onesLocal.template ReinterpretCast<float>(), 0,
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
}

__aicore__ inline void myLess(
    const AscendC::LocalTensor<uint32_t>& zLocal, const AscendC::LocalTensor<uint32_t>& xLocal,
    const AscendC::LocalTensor<uint32_t>& yLocal, AscendC::TBuf<>& tensorBuf,
    uint32_t len)
{
    uint32_t offset = 0;
    AscendC::LocalTensor<uint32_t> onesLocal = tensorBuf.GetWithOffset<uint32_t>(len, offset);
    offset += len * sizeof(uint32_t);
    AscendC::LocalTensor<uint32_t> xu16Local = tensorBuf.GetWithOffset<uint32_t>(len, offset);
    offset += len * sizeof(uint32_t);
    AscendC::LocalTensor<uint32_t> yu16Local = tensorBuf.GetWithOffset<uint32_t>(len, offset);
    offset += len * sizeof(uint32_t);
    AscendC::LocalTensor<uint8_t> maskHiLessLocal = tensorBuf.GetWithOffset<uint8_t>(len / UINT_EIGHT, offset);
    offset += len / UINT_EIGHT;
    AscendC::LocalTensor<uint8_t> maskHiEqLocal = tensorBuf.GetWithOffset<uint8_t>(len / UINT_EIGHT, offset);
    offset += len / UINT_EIGHT;
    AscendC::LocalTensor<uint8_t> maskLoLessLocal = tensorBuf.GetWithOffset<uint8_t>(len / UINT_EIGHT, offset);

    AscendC::Duplicate<float>(onesLocal.template ReinterpretCast<float>(), static_cast<float>(1), len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::ShiftRight(xu16Local, xLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(yu16Local, yLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<float, int32_t>(
        xu16Local.template ReinterpretCast<float>(), xu16Local.template ReinterpretCast<int32_t>(),
        AscendC::RoundMode::CAST_NONE, len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<float, int32_t>(
        yu16Local.template ReinterpretCast<float>(), yu16Local.template ReinterpretCast<int32_t>(),
        AscendC::RoundMode::CAST_NONE, len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(maskHiLessLocal, xu16Local.template ReinterpretCast<float>(), yu16Local.template ReinterpretCast<float>(), AscendC::CMPMODE::LT, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::ShiftRight(xu16Local, xLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(yu16Local, yLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(maskHiEqLocal, xu16Local.template ReinterpretCast<int32_t>(), yu16Local.template ReinterpretCast<int32_t>(), AscendC::CMPMODE::EQ, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::ShiftLeft(xu16Local, xLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(xu16Local, xu16Local, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftLeft(yu16Local, yLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(yu16Local, yu16Local, static_cast<uint32_t>(SHIFT_LEFT_16), len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<float, int32_t>(
        xu16Local.template ReinterpretCast<float>(), xu16Local.template ReinterpretCast<int32_t>(),
        AscendC::RoundMode::CAST_NONE, len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<float, int32_t>(
        yu16Local.template ReinterpretCast<float>(), yu16Local.template ReinterpretCast<int32_t>(),
        AscendC::RoundMode::CAST_NONE, len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(maskLoLessLocal, xu16Local.template ReinterpretCast<float>(), yu16Local.template ReinterpretCast<float>(), AscendC::CMPMODE::LT, len);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::And(maskLoLessLocal.template ReinterpretCast<uint16_t>(), maskHiEqLocal.template ReinterpretCast<uint16_t>(), maskLoLessLocal.template ReinterpretCast<uint16_t>(), len/UINT_EIGHT/UINT_DOUBLE);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Or(maskHiLessLocal.template ReinterpretCast<uint16_t>(), maskHiLessLocal.template ReinterpretCast<uint16_t>(), maskLoLessLocal.template ReinterpretCast<uint16_t>(), len/UINT_EIGHT/UINT_DOUBLE);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Select<float, uint8_t>(
        zLocal.template ReinterpretCast<float>(), maskHiLessLocal,
        onesLocal.template ReinterpretCast<float>(), static_cast<float>(0),
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<int32_t, float>(
        zLocal.template ReinterpretCast<int32_t>(), zLocal.template ReinterpretCast<float>(),
        AscendC::RoundMode::CAST_RINT, len);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void myCast(
    const AscendC::LocalTensor<uint32_t>& yLocal, const AscendC::LocalTensor<uint64_t>& xLocal, uint32_t len)
{
    AscendC::Cast<int32_t, int64_t>(
        yLocal.template ReinterpretCast<int32_t>(), xLocal.template ReinterpretCast<int64_t>(),
        AscendC::RoundMode::CAST_NONE, len);
}

}

template <typename T1>
class SimThreadExponential {
public:
    __aicore__ inline SimThreadExponential(AscendC::TPipe* p) : pipe(p) {};

    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR workspace, const SimThreadExponentialTilingData* tiling_data)
    {
        ParseTilingData(tiling_data);

        uint32_t blockIdx = AscendC::GetBlockIdx();
        blockIdxoffset = batchNumPerCore * blockIdx;
        outputOffset = batchNumPerCore * blockIdx * blockSize;
        if(blockIdx < useCoreNum - 1) {
            curCoreBatchNum = batchNumPerCore;
        } else {                                                                    // blockIdx == useCoreNum - 1
            curCoreBatchNum = batchNumTailCore;
        }

        dstGlobal.SetGlobalBuffer((__gm__ T1 *)dstGm + outputOffset);

        pipe->InitBuffer(ctrTensorBuf, batchNumPerHandle * unroll_factor * blockSize * sizeof(uint32_t));
        pipe->InitBuffer(outputBuf, batchNumPerHandle * unroll_factor * blockSize * sizeof(uint32_t));
        pipe->InitBuffer(ctrTensorTmpAndTmpBuf, UINT_DOUBLE * batchNumPerHandle * unroll_factor * blockSize * sizeof(uint32_t));
        pipe->InitBuffer(largeTmpTensorBuf, UINT_DOUBLE * batchNumPerHandle * unroll_factor * blockSize * sizeof(uint32_t));

        pipe->InitBuffer(tmpTensorBuf, UINT_FIVE * batchNumPerHandle * blockSize * sizeof(uint32_t));

        this->ctrTensor = ctrTensorBuf.Get<uint32_t>();
        this->output = outputBuf.Get<uint32_t>();
        uint32_t offset = 0;
        this->ctrTensorTmp = ctrTensorTmpAndTmpBuf.GetWithOffset<uint32_t>(batchNumPerHandle * unroll_factor * blockSize, offset);
        offset += batchNumPerHandle * unroll_factor * blockSize * sizeof(uint32_t);
        this->tmp = ctrTensorTmpAndTmpBuf.GetWithOffset<uint32_t>(batchNumPerHandle * unroll_factor * blockSize, offset);

        offset = 0;
        this->tmp0Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumPerHandle * blockSize, offset);
        offset += batchNumPerHandle * blockSize * sizeof(uint32_t);
        this->tmp1Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumPerHandle * blockSize, offset);
        offset += batchNumPerHandle * blockSize * sizeof(uint32_t);
        this->tmp2Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumPerHandle * blockSize, offset);
        offset += batchNumPerHandle * blockSize * sizeof(uint32_t);
        this->tmp3Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumPerHandle * blockSize, offset);
        offset += batchNumPerHandle * blockSize * sizeof(uint32_t);
        this->tmp4Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumPerHandle * blockSize, offset);
    }

    __aicore__ inline void genRandomData(uint32_t idx, uint32_t batchNumPerHandle, uint32_t batchNumCurHandle)
    {
        uint32_t threadIdx = blockIdxoffset + idx * batchNumPerHandle;

        // UB置0
        uint32_t zeroVal(0);
        AscendC::Duplicate<uint32_t>(ctrTensor, zeroVal, batchNumCurHandle * unroll_factor * blockSize);
        AscendC::PipeBarrier<PIPE_V>();

        setnlo0nhi0(ctrTensor, tmp.template ReinterpretCast<int32_t>(), tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, largeTmpTensorBuf, threadIdx, batchNumCurHandle * blockSize);

        setnlo1nhi1(ctrTensor, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, largeTmpTensorBuf, batchNumCurHandle * blockSize);

        AscendC::DataCopy(output, ctrTensor, batchNumCurHandle * unroll_factor * blockSize);
        AscendC::PipeBarrier<PIPE_V>();
        keyTmp0 = key0;
        keyTmp1 = key1;
        for(uint32_t idx2 = 0; idx2 < UINT_NINE; idx2++) {
            rand_Philox4x32_10(output, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, 
                largeTmpTensorBuf, keyTmp0, keyTmp1, batchNumCurHandle * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            keyTmp0 += PHILOX_W32_0;
            keyTmp1 += PHILOX_W32_1;
        }
        rand_Philox4x32_10(output, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, 
            largeTmpTensorBuf, keyTmp0, keyTmp1, batchNumCurHandle * blockSize);
        AscendC::PipeBarrier<PIPE_V>();

        for(uint32_t linearIndexBlock = threadIdx; linearIndexBlock < roundedSizeBlock; linearIndexBlock += stepBlock) {
            uint32_t linearIndexBlockCurCore = linearIndexBlock - blockIdxoffset;
            AscendC::DataCopy(tmp, output, batchNumCurHandle * unroll_factor * blockSize);

            setCtrTensor(ctrTensor, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, largeTmpTensorBuf, batchNumCurHandle * blockSize);
            AscendC::DataCopy(output, ctrTensor, batchNumCurHandle * unroll_factor * blockSize);

            keyTmp0 = key0;
            keyTmp1 = key1;
            for(uint32_t idx2 = 0; idx2 < UINT_NINE; idx2++) {
                rand_Philox4x32_10(output, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, 
                    largeTmpTensorBuf, keyTmp0, keyTmp1, batchNumCurHandle * blockSize);
                AscendC::PipeBarrier<PIPE_V>();
                keyTmp0 += PHILOX_W32_0;
                keyTmp1 += PHILOX_W32_1;
            }
            rand_Philox4x32_10(output, tmp0Tensor, tmp1Tensor, tmp2Tensor, tmp3Tensor, tmp4Tensor, 
                largeTmpTensorBuf, keyTmp0, keyTmp1, batchNumCurHandle * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(ctrTensorTmp, output, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<uint32_t> xTensor = largeTmpTensorBuf.GetWithOffset<uint32_t>(UINT_DOUBLE * batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));
            AscendC::Duplicate<uint32_t>(xTensor, zeroVal, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            setXTensor(xTensor, tmp, ctrTensorTmp, batchNumCurHandle * blockSize);

            AscendC::PipeBarrier<PIPE_ALL>();
            ctrTensorTmp = ctrTensorTmpAndTmpBuf.GetWithOffset<uint32_t>(UINT_DOUBLE* batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));
            AscendC::LocalTensor<uint32_t> u32Tof32Tensor = tmpTensorBuf.GetWithOffset<uint32_t>(batchNumCurHandle * unroll_factor * blockSize, static_cast<uint32_t>(0));

            uint32_to_float_func(u32Tof32Tensor.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>(), xTensor, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(ctrTensorTmp.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>(), static_cast<float>(CURAND_2POW32_INV), batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ctrTensorTmp.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>(), static_cast<float>(1.1641532e-10f), batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            uniform_func(ctrTensorTmp.template ReinterpretCast<float>(), tmp0Tensor.template ReinterpretCast<uint8_t>(), tmp.template ReinterpretCast<float>(), start, end, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PipeBarrier<PIPE_ALL>();

            uint32_t maskOffset = 0;
            AscendC::LocalTensor<uint8_t> zeroMask = tmpTensorBuf.GetWithOffset<uint8_t>(batchNumCurHandle * unroll_factor * blockSize / PER_UINT8_8BITS, maskOffset);
            AscendC::CompareScalar<float, uint8_t>(zeroMask, ctrTensorTmp.template ReinterpretCast<float>(), 0, AscendC::CMPMODE::NE, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<float, uint8_t>(
                ctrTensorTmp.template ReinterpretCast<float>(), zeroMask, ctrTensorTmp.template ReinterpretCast<float>(), 1, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                REPEAT_NUM_64, batchNumCurHandle * unroll_factor * blockSize / REPEAT_NUM_64, { 1, 1, 1, 8, 8, 8 });
            AscendC::PipeBarrier<PIPE_V>();
            halfEpsilon = EPSILON_ / DOUBLE;
            AscendC::CompareScalar<float, uint8_t>(zeroMask, ctrTensorTmp.template ReinterpretCast<float>(), 1 - halfEpsilon, AscendC::CMPMODE::LT, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Log<float>(ctrTensorTmp.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>());
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select<float, uint8_t>(
                ctrTensorTmp.template ReinterpretCast<float>(), zeroMask, ctrTensorTmp.template ReinterpretCast<float>(), -halfEpsilon, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                REPEAT_NUM_64, batchNumCurHandle * unroll_factor * blockSize / REPEAT_NUM_64, { 1, 1, 1, 8, 8, 8 });
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls<float>(ctrTensorTmp.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>(), -1, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate<float>(u32Tof32Tensor.template ReinterpretCast<float>(), lambda, batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div<float>(ctrTensorTmp.template ReinterpretCast<float>(), ctrTensorTmp.template ReinterpretCast<float>(), u32Tof32Tensor.template ReinterpretCast<float>(), batchNumCurHandle * unroll_factor * blockSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PipeBarrier<PIPE_ALL>();
            
            if (std::is_same<T1, float>::value == false) {
                AscendC::Cast<T1, float>(
                    xTensor.template ReinterpretCast<T1>(), ctrTensorTmp.template ReinterpretCast<float>(),
                    AscendC::RoundMode::CAST_RINT, batchNumCurHandle * unroll_factor * blockSize);
                AscendC::PipeBarrier<PIPE_ALL>();

                if(linearIndexBlock * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize], xTensor.template ReinterpretCast<T1>(), batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + batchNumTotal * blockSize], xTensor.template ReinterpretCast<T1>()[batchNumCurHandle * blockSize], batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + UINT_DOUBLE * batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + UINT_DOUBLE * batchNumTotal * blockSize], xTensor.template ReinterpretCast<T1>()[UINT_DOUBLE * batchNumCurHandle * blockSize], batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + UINT_THREE * batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + UINT_THREE * batchNumTotal * blockSize], xTensor.template ReinterpretCast<T1>()[UINT_THREE * batchNumCurHandle * blockSize], batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            } else {
                if(linearIndexBlock * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize], ctrTensorTmp.template ReinterpretCast<T1>(), batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + batchNumTotal * blockSize], ctrTensorTmp[batchNumCurHandle * blockSize].template ReinterpretCast<T1>(), batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + UINT_DOUBLE * batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + UINT_DOUBLE * batchNumTotal * blockSize], ctrTensorTmp[UINT_DOUBLE * batchNumCurHandle * blockSize].template ReinterpretCast<T1>(), batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                if(linearIndexBlock * blockSize + UINT_THREE * batchNumTotal * blockSize <= numel) {
                    AscendC::DataCopy(dstGlobal[linearIndexBlockCurCore * blockSize + UINT_THREE * batchNumTotal * blockSize], ctrTensorTmp[UINT_THREE * batchNumCurHandle * blockSize].template ReinterpretCast<T1>(), batchNumCurHandle * blockSize);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        for(uint32_t idx=0; idx<handleNumLoop; idx++) {
            genRandomData(idx, batchNumPerHandle, batchNumPerHandle);
        }
        if (handleNumTail > 0) {
            genRandomData(handleNumLoop, batchNumPerHandle, handleNumTail);
        }
    }

    __aicore__ inline void setnlo0nhi0(
        const AscendC::LocalTensor<uint32_t>& ctrTensor, const AscendC::LocalTensor<int32_t>& arangeTensor, 
        const AscendC::LocalTensor<uint32_t>& blockNumTensor, const AscendC::LocalTensor<uint32_t>& iTensor, 
        const AscendC::LocalTensor<uint32_t>& nlo0Tensor, const AscendC::LocalTensor<uint32_t>& nhi0Tensor, 
        const AscendC::LocalTensor<uint32_t>& nhi0TmpTensor, AscendC::TBuf<>& compareTensorBuf, uint32_t curThreadIdx, uint32_t len)
    {
        AscendC::Duplicate<uint32_t>(blockNumTensor, static_cast<uint32_t>(curThreadIdx), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ArithProgression<int32_t>(arangeTensor, static_cast<int32_t>(0), static_cast<int32_t>(1), len);
        AscendC::PipeBarrier<PIPE_V>();
        myMuls(iTensor, blockNumTensor, static_cast<uint32_t>(blockSize), len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd2(iTensor, iTensor, arangeTensor, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::DataCopy(nlo0Tensor, iTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(nhi0Tensor, iTensor, static_cast<uint32_t>(SHIFT_LEFT_32), len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_DOUBLE * len], ctrTensor[UINT_DOUBLE * len], nlo0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myLess(nhi0TmpTensor, ctrTensor[UINT_DOUBLE * len], nlo0Tensor, compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();

        myAdd<uint32_t, int32_t>(nhi0Tensor, nhi0Tensor, nhi0TmpTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_THREE * len], ctrTensor[UINT_THREE * len], nhi0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void setnlo1nhi1(
        const AscendC::LocalTensor<uint32_t>& ctrTensor, const AscendC::LocalTensor<uint32_t>& nlo1Tensor, 
        const AscendC::LocalTensor<uint32_t>& nhi1Tensor, const AscendC::LocalTensor<uint32_t>& nhi1TmpTensor,
        const AscendC::LocalTensor<uint32_t>& nhi1TmpGTTensor, const AscendC::LocalTensor<uint32_t>& ctrTensorEQ0Tensor,
        AscendC::TBuf<>& compareTensorBuf, uint32_t len)
    {
        AscendC::Duplicate<uint32_t>(nlo1Tensor, offset_t_low, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<uint32_t>(nhi1Tensor, offset_t_high, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor, ctrTensor, nlo1Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myLess(nhi1TmpTensor, ctrTensor, nlo1Tensor, compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(nhi1Tensor, nhi1Tensor, nhi1TmpTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[len], ctrTensor[len], nhi1Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myLess(nhi1TmpGTTensor, ctrTensor[len], nhi1Tensor, compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_DOUBLE * len], ctrTensor[UINT_DOUBLE * len], nhi1TmpGTTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myEqualZeroScalar(ctrTensorEQ0Tensor, ctrTensor[UINT_DOUBLE * len], compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(nhi1TmpGTTensor, nhi1TmpGTTensor, ctrTensorEQ0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_THREE * len], ctrTensor[UINT_THREE * len], nhi1TmpGTTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void setCtrTensor(
        const AscendC::LocalTensor<uint32_t>& ctrTensor, const AscendC::LocalTensor<uint32_t>& ctr0EQ0Tensor,
        const AscendC::LocalTensor<uint32_t>& ctr1EQ0Tensor, const AscendC::LocalTensor<uint32_t>& ctrMulEQ01Tensor,
        const AscendC::LocalTensor<uint32_t>& ctr2EQ0Tensor, const AscendC::LocalTensor<uint32_t>& ctrMulEQ012Tensor,
        AscendC::TBuf<>& compareTensorBuf, uint32_t len)
    {
        myAdds(ctrTensor, ctrTensor, static_cast<uint32_t>(1), len);
        AscendC::PipeBarrier<PIPE_V>();
        myEqualZeroScalar(ctr0EQ0Tensor, ctrTensor, compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[len], ctrTensor[len], ctr0EQ0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myEqualZeroScalar(ctr1EQ0Tensor, ctrTensor[len], compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(ctrMulEQ01Tensor, ctr0EQ0Tensor, ctr1EQ0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_DOUBLE * len], ctrTensor[UINT_DOUBLE * len], ctrMulEQ01Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myEqualZeroScalar(ctr2EQ0Tensor, ctrTensor[UINT_DOUBLE * len], compareTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(ctrMulEQ012Tensor, ctrMulEQ01Tensor, ctr2EQ0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(ctrTensor[UINT_THREE * len], ctrTensor[UINT_THREE * len], ctrMulEQ012Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void setXTensor(
        const AscendC::LocalTensor<uint32_t>& xTensor, const AscendC::LocalTensor<uint32_t>& tmp, 
        const AscendC::LocalTensor<uint32_t>& output, uint32_t len)
    {
        if(state == UINT_DOUBLE) {
            AscendC::DataCopy(xTensor, tmp[UINT_DOUBLE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[len], tmp[UINT_THREE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_DOUBLE * len], output, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_THREE * len], output[len], len);
            AscendC::PipeBarrier<PIPE_V>();
        } else if(state == UINT_THREE) {
            AscendC::DataCopy(xTensor, tmp[UINT_THREE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[len], output, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_DOUBLE * len], output[1 * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_THREE * len], output[UINT_DOUBLE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::DataCopy(xTensor, tmp, len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[len], tmp[1 * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_DOUBLE * len], tmp[UINT_DOUBLE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::DataCopy(xTensor[UINT_THREE * len], tmp[UINT_THREE * len], len);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void uniform_func(
        const AscendC::LocalTensor<float>& randVal, const AscendC::LocalTensor<uint8_t>& randValEQEndBitMask, const AscendC::LocalTensor<float>& startTensor, float start, float end, uint32_t len)
    {
        float range = end - start;
        AscendC::Muls(randVal, randVal, static_cast<float>(range), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(randVal, randVal, static_cast<float>(start), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(randValEQEndBitMask, randVal, static_cast<float>(end), AscendC::CMPMODE::EQ, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<float>(startTensor, static_cast<float>(start), len);
        AscendC::PipeBarrier<PIPE_V>();
        Select(randVal, randValEQEndBitMask, startTensor, randVal, AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, len);
    }

    __aicore__ inline void uint32_to_float_func(
        const AscendC::LocalTensor<float>& dstValTmp, const AscendC::LocalTensor<float>& dstVal, const AscendC::LocalTensor<uint32_t>& u32Value, 
        uint32_t len)
    {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(dstValTmp.template ReinterpretCast<uint32_t>(), u32Value, static_cast<uint32_t>(SHIFT_LEFT_31), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstVal.template ReinterpretCast<int64_t>(), u32Value.template ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(u32Value.template ReinterpretCast<int64_t>(), dstValTmp.template ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(u32Value.template ReinterpretCast<uint32_t>(), u32Value.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(SHIFT_LEFT_31), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Or(u32Value.template ReinterpretCast<uint16_t>(), dstVal.template ReinterpretCast<uint16_t>(), u32Value.template ReinterpretCast<uint16_t>(), UINT_FOUR * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<uint32_t>(dstVal.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t mask[UINT_DOUBLE] = { 0xCCCCCCCCCCCCCCCC, 0xCCCCCCCCCCCCCCCC };
        AscendC::And(u32Value.template ReinterpretCast<uint16_t>(), u32Value.template ReinterpretCast<uint16_t>(), dstVal.template ReinterpretCast<uint16_t>(), mask, len / SHIFT_LEFT_32, { 1, 1, 1, 8, 8, 8 });
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(dstVal, u32Value.template ReinterpretCast<int64_t>(), AscendC::RoundMode::CAST_RINT, len);
    }

    __aicore__ inline void multiply_uint64(
        const AscendC::LocalTensor<uint32_t>& ctrTensor, const AscendC::LocalTensor<uint32_t>& tensor, 
        const AscendC::LocalTensor<uint32_t>& hiTensor, const AscendC::LocalTensor<uint32_t>& loTensor, 
        AscendC::TBuf<>& hlAndpTensorBuf, uint32_t len)
    {
        uint32_t offset = 0;
        AscendC::LocalTensor<uint32_t> aHLocal = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> aLLocal = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> bHLocal = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> bLLocal = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);

        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> p0Local = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> p1Local = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> p2Local = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);
        offset += len * sizeof(uint32_t);
        AscendC::LocalTensor<uint32_t> p3Local = hlAndpTensorBuf.GetWithOffset<uint32_t>(len, offset);

        AscendC::ShiftRight(aHLocal, tensor, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(aLLocal, tensor, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(aLLocal, aLLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(bHLocal, ctrTensor, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(bLLocal, ctrTensor, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(bLLocal, bLLocal, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();

        myMul<uint32_t, int32_t>(p0Local, aLLocal, bLLocal, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(p1Local, aLLocal, bHLocal, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(p2Local, aHLocal, bLLocal, len);
        AscendC::PipeBarrier<PIPE_V>();
        myMul<uint32_t, int32_t>(p3Local, aHLocal, bHLocal, len);
        AscendC::PipeBarrier<PIPE_V>();

        myAdd<uint32_t, int32_t>(p2Local, p1Local, p2Local, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(aLLocal, p2Local, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(loTensor, p0Local, aLLocal, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(hiTensor, p2Local, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        myLess(p2Local, p2Local, p1Local, hlAndpTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(p2Local, p2Local, static_cast<uint32_t>(SHIFT_LEFT_16), len);
        AscendC::PipeBarrier<PIPE_V>();
        myLess(p0Local, loTensor, p0Local, hlAndpTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(hiTensor, p3Local, hiTensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(hiTensor, hiTensor, p2Local, len);
        AscendC::PipeBarrier<PIPE_V>();
        myAdd<uint32_t, int32_t>(hiTensor, hiTensor, p0Local, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void rand_Philox4x32_10(
        const AscendC::LocalTensor<uint32_t>& ctrTensor,
        AscendC::LocalTensor<uint32_t>& tensor0, 
        AscendC::LocalTensor<uint32_t>& hi0Tensor, AscendC::LocalTensor<uint32_t>& lo0Tensor, 
        const AscendC::LocalTensor<uint32_t>& hi1Tensor, 
        const AscendC::LocalTensor<uint32_t>& lo1Tensor, AscendC::TBuf<>& largeTmpTensorBuf,
        uint32_t key0, uint32_t key1, uint32_t len)
    {
        AscendC::Duplicate<uint32_t>(tensor0, static_cast<uint32_t>(PHILOX_M4x32_0), len);
        AscendC::PipeBarrier<PIPE_V>();
        multiply_uint64(ctrTensor, tensor0, hi0Tensor, lo0Tensor, largeTmpTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Duplicate<uint32_t>(tensor0, static_cast<uint32_t>(PHILOX_M4x32_1), len);
        AscendC::PipeBarrier<PIPE_V>();
        multiply_uint64(ctrTensor[UINT_DOUBLE * len], tensor0, hi1Tensor, lo1Tensor, largeTmpTensorBuf, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Xor(tensor0.template ReinterpretCast<uint16_t>(), hi1Tensor.template ReinterpretCast<uint16_t>(), ctrTensor[len].template ReinterpretCast<uint16_t>(), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<uint32_t>(ctrTensor[len], static_cast<uint32_t>(key0), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Xor(ctrTensor.template ReinterpretCast<uint16_t>(), tensor0.template ReinterpretCast<uint16_t>(), ctrTensor[len].template ReinterpretCast<uint16_t>(), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(ctrTensor[len], lo1Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Xor(tensor0.template ReinterpretCast<uint16_t>(), hi0Tensor.template ReinterpretCast<uint16_t>(), ctrTensor[UINT_THREE * len].template ReinterpretCast<uint16_t>(), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<uint32_t>(ctrTensor[UINT_THREE * len], static_cast<uint32_t>(key1), len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Xor(ctrTensor[UINT_DOUBLE * len].template ReinterpretCast<uint16_t>(), tensor0.template ReinterpretCast<uint16_t>(), ctrTensor[UINT_THREE * len].template ReinterpretCast<uint16_t>(), UINT_DOUBLE * len);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(ctrTensor[UINT_THREE * len], lo0Tensor, len);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ParseTilingData(const SimThreadExponentialTilingData* tilingData)
    {
        key0 = tilingData->key0;
        key1 = tilingData->key1;
        offset_t_low = tilingData->offset_t_low;
        offset_t_high = tilingData->offset_t_high;
        useCoreNum = tilingData->useCoreNum;
        batchNumPerCore = tilingData->batchNumPerCore;
        batchNumTailCore = tilingData->batchNumTailCore;
        batchNumTotal = tilingData->batchNumTotal;
        numel = tilingData->numel;
        stepBlock = tilingData->stepBlock;
        roundedSizeBlock = tilingData->roundedSizeBlock;
        range = tilingData->range;
        handleNumLoop = tilingData->handleNumLoop;
        handleNumTail = tilingData->handleNumTail;
        state = tilingData->state;
        start = tilingData->start;
        end = tilingData->end;
        lambda = tilingData->lambda;
    }

private:
    AscendC::TPipe* pipe{nullptr};

    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctrTensorBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outputBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> ctrTensorTmpAndTmpBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> largeTmpTensorBuf;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpTensorBuf;

    AscendC::GlobalTensor<T1> dstGlobal;

    AscendC::LocalTensor<uint32_t> ctrTensor;
    AscendC::LocalTensor<uint32_t> ctrTensorTmp;
    AscendC::LocalTensor<uint32_t> tmp;
    AscendC::LocalTensor<uint32_t> output;

    AscendC::LocalTensor<uint32_t> tmp0Tensor;
    AscendC::LocalTensor<uint32_t> tmp1Tensor;
    AscendC::LocalTensor<uint32_t> tmp2Tensor;
    AscendC::LocalTensor<uint32_t> tmp3Tensor;
    AscendC::LocalTensor<uint32_t> tmp4Tensor;

    // 可设置为常量
    uint32_t blockSize{256};
    uint32_t batchNumPerHandle = 6;
    uint32_t unroll_factor{4};
    uint32_t lenPerHandle = batchNumPerHandle * blockSize;

    uint32_t key0 = 5;
    uint32_t key1 = 0;
    uint32_t offset_t_low = 0;
    uint32_t offset_t_high = 0;
    uint32_t useCoreNum = 48;
    uint32_t batchNumPerCore = 18;
    uint32_t batchNumTailCore = 18;
    uint32_t batchNumTotal = 864;
    int64_t numel = 250000;

    uint64_t state = 0;

    float start = 0;
    float end = 1;
    float lambda = 1.0;
    float halfEpsilon = 0;
    uint64_t outputOffset = 0;
    uint32_t keyTmp0 = 5;
    uint32_t keyTmp1 = 0;
    uint32_t blockIdx = 0;
    uint32_t blockIdxoffset;
    uint32_t curCoreBatchNum = 0;
    uint32_t handleNumLoop = 0;
    uint32_t handleNumTail = 0;
    uint32_t stepBlock = 0;
    uint32_t roundedSizeBlock = 0;
    float range = 0;
};
#endif // SIM_THREAD_EXPONENTIAL_H