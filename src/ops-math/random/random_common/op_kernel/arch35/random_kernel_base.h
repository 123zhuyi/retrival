/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANDOM_KERNEL_BASE_H
#define RANDOM_KERNEL_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "random_unified_tiling_data_arch35.h"
#include "simt_api/asc_simt.h"

namespace RandomKernelBase {
using namespace AscendC;

static constexpr Reg::CastTrait castTraitTf = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING};

static constexpr uint16_t ALG_KEY_SIZE = 2;
static constexpr uint16_t ALG_COUNTER_SIZE = 4;
static constexpr uint32_t INT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
static constexpr uint32_t RIGHT_SHIFT = 32;
static constexpr uint16_t BLOCK_SIZE = 32;
static constexpr uint16_t DOUBLE_UNIFORM_RESULT = 2;
static constexpr uint32_t INT32_FLOAT32_ONE_REPEAT = Ops::Base::GetVRegSize() / sizeof(int32_t);
static constexpr float DOUBLE_MULTIPLE = 2.0f;
static constexpr float PI = 3.14159265358979323846f;

static constexpr int IDX_2 = 2;
static constexpr int IDX_3 = 3;
static constexpr int32_t CONTINUOUS_USE = 0;
static constexpr int32_t DIS_CONTINUOUS_USE = 1;
static constexpr uint32_t PHILOX_W32_A = 0x9E3779B9;
static constexpr uint32_t PHILOX_W32_B = 0xBB67AE85;
static constexpr uint32_t PHILOX_M4X32_A = 0xD2511F53;
static constexpr uint32_t PHILOX_M4X32_B = 0xCD9E8D57;
static constexpr float RAND_2POW32_INV = 2.3283064e-10f;
static constexpr float RAND_2POW32_INV_HALF = RAND_2POW32_INV / 2.0f;

constexpr uint64_t VEC_2 = 2;
constexpr uint64_t VEC_4 = 4;
constexpr uint64_t VEC_8 = 8;
constexpr uint64_t VEC_16 = 16;

template <typename T>
__aicore__ inline void CopyOut(LocalTensor<T> yLocal, GlobalTensor<T> yGm, uint32_t burstNum, uint32_t busrtLength,
                               int64_t gmOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = burstNum;
    copyParams.blockLen = busrtLength;
    DataCopyPad(yGm[gmOffset], yLocal, copyParams);
}

template <typename T>
__aicore__ inline void Uint32ToFloat(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ float* ubOut = (__ubuf__ float*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

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
__aicore__ inline void Uint16ToHalf(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ half* ubOut = (__ubuf__ half*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60, 60>(0);
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

    SetCtrlSpr<60, 60>(1);
}

template <typename T>
__aicore__ inline void Uint16ToBfloat16(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes,
                                        const uint32_t calCount)
{
    __ubuf__ int32_t* ubPhilox = (__ubuf__ int32_t*)philoxRes.GetPhyAddr();
    __ubuf__ bfloat16_t* ubOut = (__ubuf__ bfloat16_t*)yOutput.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount, static_cast<uint32_t>(INT32_ONE_REPEAT));

    SetCtrlSpr<60, 60>(0);
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

    SetCtrlSpr<60, 60>(1);
}

template <typename T>
__aicore__ inline void U32Conversion(LocalTensor<T>& yOutput, LocalTensor<uint32_t>& philoxRes, const uint32_t calCount)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Uint16ToHalf(yOutput, philoxRes, calCount);
    } else if constexpr (AscendC::IsSameType<T, float>::value) {
        Uint32ToFloat(yOutput, philoxRes, calCount);
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        Uint16ToBfloat16(yOutput, philoxRes, calCount);
    }
}

template <typename T>
__aicore__ inline void Float32Conversion(LocalTensor<T> yOutput, LocalTensor<float> normalFloatResult,
                                         const uint32_t calCount)
{
    if constexpr (AscendC::IsSameType<T, float>::value) {
        DataCopy(yOutput, normalFloatResult, Ops::Base::CeilAlign(calCount, static_cast<uint32_t>(BLOCK_SIZE)));
    } else if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_NONE, calCount);
    } else {
        Cast(yOutput, normalFloatResult, RoundMode::CAST_RINT, calCount);
    }
}

/*
 *  Formula: Box-Muller
 *   X = sqrt(-2 * ln(U1)) * cos(2 * PI * U2)
 *   Y = sqrt(-2 * ln(U1)) * sin(2 * PI * U2)
 */
template <typename T>
__aicore__ inline void BoxMullerFloatSIMD(LocalTensor<float>& yOutputTmp, LocalTensor<float> v1Result,
                                          LocalTensor<float> u2Result, const uint32_t calCount)
{
    __ubuf__ float* uniformRes = (__ubuf__ float*)yOutputTmp.GetPhyAddr();
    __ubuf__ float* ubV1Out = (__ubuf__ float*)v1Result.GetPhyAddr();
    __ubuf__ float* ubU2Out = (__ubuf__ float*)u2Result.GetPhyAddr();
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount / DOUBLE_UNIFORM_RESULT,
                                              static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));
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
__aicore__ inline void BoxMullerMulSIMD(LocalTensor<float>& yOutputTmp, LocalTensor<float> v1Result,
                                        LocalTensor<float> u2Result, LocalTensor<float> yOutput,
                                        const uint32_t calCount)
{
    Cos<float, false>(yOutputTmp, v1Result);
    Sin<float, false>(yOutput, v1Result);
    uint32_t repeatTimes = Ops::Base::CeilDiv(calCount / DOUBLE_UNIFORM_RESULT,
                                              static_cast<uint32_t>(INT32_FLOAT32_ONE_REPEAT));

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
}

// Box-Muller transform（Use BoxMullerFloatSafe for eps-protected version)
__simt_callee__ __aicore__ inline void BoxMullerFloatSafe(float u1, const float u2, float* z0, float* z1)
{
    const float eps = 1.0e-7f;
    if (u1 < eps) {
        u1 = eps;
    }
    float v = static_cast<float>(DOUBLE_MULTIPLE * PI * u2);
    float r = sqrtf(-DOUBLE_MULTIPLE * logf(u1));
    sincosf(v, z0, z1);
    *z0 *= r;
    *z1 *= r;
}

// Box-Muller transform (Use BoxMullerFloat for torch-aligned, no eps clamping. ）
__simt_callee__ __aicore__ inline void BoxMullerFloat(float u1, const float u2, float* z0, float* z1)
{
    float v = static_cast<float>(DOUBLE_MULTIPLE * PI * u2);
    float r = sqrtf(-DOUBLE_MULTIPLE * logf(u1));
    sincosf(v, z0, z1);
    *z0 *= r;
    *z1 *= r;
}

template <uint16_t COPY_SIZE>
__simt_callee__ __aicore__ inline void CopyArray(uint32_t* dst, const uint32_t* src)
{
#pragma unroll
    for (uint16_t i = 0; i < COPY_SIZE; i++) {
        dst[i] = src[i];
    }
}

__simt_callee__ __aicore__ inline void SkipOne(uint32_t* counter)
{
    if (++counter[0])
        return;
    if (++counter[1])
        return;
    if (++counter[IDX_2])
        return;
    ++counter[IDX_3];
}

__simt_callee__ __aicore__ inline void SkipLo(uint32_t* counter, uint64_t n)
{
    const uint32_t nlo = static_cast<uint32_t>(n);
    uint32_t nhi = static_cast<uint32_t>(n >> RIGHT_SHIFT);

    counter[0] += nlo;
    if (counter[0] < nlo) {
        nhi++;
    }
    counter[1] += nhi;
    if (nhi <= counter[1])
        return;
    if (++counter[IDX_2])
        return;
    ++counter[IDX_3];
}

__simt_callee__ __aicore__ inline void SkipHi(uint32_t* counter, uint64_t n)
{
    const uint32_t countLo = static_cast<uint32_t>(n);
    uint32_t countHi = static_cast<uint32_t>(n >> RIGHT_SHIFT);

    counter[IDX_2] += countLo;
    if (counter[IDX_2] < countLo) {
        countHi++;
    }
    counter[IDX_3] += countHi;
}

__simt_callee__ __aicore__ inline void FlashCounter(uint64_t globalThreadIdx, uint64_t offset, uint32_t* counter)
{
    SkipHi(counter, globalThreadIdx);
    SkipLo(counter, offset);
}

__simt_callee__ __aicore__ inline void PhiloxAlgParsInit(uint32_t* key, uint32_t* counter, int64_t seed, int64_t offset)
{
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(seed >> RIGHT_SHIFT);

    offset = (offset + VEC_4 - 1) / VEC_4;
    SkipLo(counter, offset);
}

__simt_callee__ __aicore__ inline void Philox4x32Round(uint32_t* counter, const uint32_t* key)
{
    uint64_t res0_u64 = __mul_i32toi64(PHILOX_M4X32_A, counter[0]);
    uint64_t res1_u64 = __mul_i32toi64(PHILOX_M4X32_B, counter[IDX_2]);

    // 这里使用uint2和uint32功能等价，但是编译器的编译结果不一样，导致性能差异很大，使用性能更好的uint2
    uint2& res0 = reinterpret_cast<uint2&>(res0_u64);
    uint2& res1 = reinterpret_cast<uint2&>(res1_u64);

    counter[0] = res1.y ^ counter[1] ^ key[0];
    counter[1] = res1.x;
    counter[IDX_2] = res0.y ^ counter[IDX_3] ^ key[1];
    counter[IDX_3] = res0.x;
}

__simt_callee__ __aicore__ inline void KeyInc(uint32_t* key)
{
    key[0] += PHILOX_W32_A;
    key[1] += PHILOX_W32_B;
}

// 算法内部在迭代时使用临时变量，不会修改传入的key 和 counter
__simt_callee__ __aicore__ inline void PhiloxRandomSimt(const uint32_t* key, const uint32_t* counter, uint32_t* results)
{
    uint32_t keyTmp[ALG_KEY_SIZE];
    uint32_t counterTmp[ALG_COUNTER_SIZE];
    CopyArray<ALG_KEY_SIZE>(keyTmp, key);
    CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);

    Philox4x32Round(counterTmp, keyTmp); // 1
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 2
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 3
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 4
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 5
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 6
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 7
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 8
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 9
    KeyInc(keyTmp);
    Philox4x32Round(counterTmp, keyTmp); // 10
    CopyArray<ALG_COUNTER_SIZE>(results, counterTmp);
}

// 算法内部在迭代时使用临时变量，不会修改传入的key 和 counter
__simt_callee__ __aicore__ inline void PhiloxRandomSimt(const uint32_t* key, const uint32_t* counter, float* results)
{
    uint32_t resultU32[ALG_COUNTER_SIZE];
    PhiloxRandomSimt(key, counter, resultU32);
#pragma unroll
    for (uint16_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        results[i] = resultU32[i] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
    }
}

// 除数 (gridDimx * blockDim) 使用uint64快除接口， 提升性能
template <int32_t STEP, int32_t ARANGE_MODE>
__simt_callee__ __aicore__ inline void ThreadMappingAndSkip(uint64_t idx, uint32_t* counter, uint64_t magic,
                                                            uint64_t shift, uint64_t totalThreads)
{
    static_assert(!(ARANGE_MODE == DIS_CONTINUOUS_USE && STEP != VEC_4),
                  "When ARANGE_MODE is DIS_CONTINUOUS_USE, STEP must be 4");
    uint64_t idxTmp = idx / STEP;
    uint64_t globalThreadIdx = 0;
    uint64_t repeat = Simt::UintDiv(idxTmp, magic, shift);
    // 排列方式 0000 1111 ...0000 1111
    if constexpr (ARANGE_MODE == CONTINUOUS_USE) {
        globalThreadIdx = idxTmp - repeat * totalThreads;
    } else {
        // 排列方式 0123 4567 ... 0123 4567 ...
        auto repeatTmp = Simt::UintDiv(idx, magic, shift);
        globalThreadIdx = idx - repeatTmp * totalThreads;
    }
    // DIS_CONTINUOUS_USE模式下 STEP一定要求是4
    if constexpr (STEP == VEC_8 || STEP == VEC_16) {
        repeat = repeat * (STEP / VEC_4) + (idx / VEC_4 % (STEP / VEC_4));
    }

    FlashCounter(globalThreadIdx, repeat, counter);
}

/*
使用说明
        uint32_t key[ALG_KEY_SIZE] = {0, 0};
        uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        PhiloxAlgParsInit(key, counter, seed, offset);
        int32_t step = 4;

        uint64_t totalThreads = gridDimx(动态计算) * blockDim（固定值）;
        uint64_t magic, shift;
        GetUintDivMagicAndShift(magic, shift, totalThreads);
        // 方式1：
        for (int64_t i = (blockIdx.x * blockDim.x + threadIdx.x) * step; i < outputLength;
            i += gridDim.x * blockDim.x * step)
         {
            uint32_t results[ALG_COUNTER_SIZE];   // 或者 float类型
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
            ThreadMappingAndSkip<step, CONTINUOUS_USE>(i, counterTmp, magic, shift, totalThreads);
            PhiloxRandomSimt(key, counterTmp, results);
            // 使用results 对连续的4个索引做操作
         }

        // 方式2：
        for (int64_t i = (blockIdx.x * blockDim.x + threadIdx.x); i < outputLength;
            i += gridDim.x * blockDim.x * step)
         {
            uint32_t results[ALG_COUNTER_SIZE];   // 或者 float类型
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
            ThreadMappingAndSkip<step, DIS_CONTINUOUS_USE>(i, counterTmp, magic, shift, totalThreads);
            PhiloxRandomSimt(key, counterTmp, results);
            // 使用results 对非连续的4个索引做操作，stride为totalThreads
         }
*/

class RandomKernelBaseOp {
public:
    __aicore__ inline RandomKernelBaseOp(const RandomUnifiedTilingDataStruct* __restrict tilingData)
        : tiling_(tilingData){};
    __aicore__ inline void VarsInit();

    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void GenRandomSIMD(LocalTensor<uint32_t> randomLocal, const uint64_t count);

    const RandomUnifiedTilingDataStruct* tiling_;

    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};

    uint32_t blockIdx_;
    int64_t curCoreProNum_ = 0;
    int64_t ubRepeatimes_ = 0;
};

__aicore__ inline void RandomKernelBaseOp::VarsInit()
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > tiling_->usedCoreNum) {
        return;
    }

    if (blockIdx_ == tiling_->usedCoreNum - 1) {
        curCoreProNum_ = tiling_->tailCoreProNum;
    } else {
        curCoreProNum_ = tiling_->normalCoreProNum;
    }

    ubRepeatimes_ = Ops::Base::CeilDiv(curCoreProNum_, tiling_->singleBufferSize);
    // InitKeyAndCounter
    for (uint32_t i = 0; i < ALG_KEY_SIZE; i++) {
        key_[i] = tiling_->key[i];
    }
    for (uint32_t i = 0; i < ALG_COUNTER_SIZE; i++) {
        counter_[i] = tiling_->counter[i];
    }

    /*    算子自管理
        GlobalTensor   SetGlobalBuffer
        Que  InitBuffer
    */
}

__aicore__ inline void RandomKernelBaseOp::Skip(const uint64_t count)
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

__aicore__ inline void RandomKernelBaseOp::GenRandomSIMD(LocalTensor<uint32_t> randomLocal, const uint64_t count)
{
    constexpr uint16_t kPhiloxRounds = 10;
    PhiloxRandom<kPhiloxRounds>(randomLocal, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                                count);
}

constexpr uint32_t SIMT_STEP = 4;
constexpr uint32_t DEFAULT_SIMT_THREAD_NUM = 512;

struct ExecutionPolicyKernel {
    uint64_t magic;
    uint64_t shift;
};

template <typename T, typename TransformFunc, uint32_t ThreadNum = DEFAULT_SIMT_THREAD_NUM, uint32_t Vec = SIMT_STEP>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void PhiloxSimtKernelContinuous(
    __gm__ volatile T* outputGm, int64_t offset, int64_t seed, uint64_t outputSize, uint64_t magic, uint64_t shift,
    uint64_t totalThreads, TransformFunc transform)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    PhiloxAlgParsInit(key, counter, seed, offset);

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t randRounds = (Vec + SIMT_STEP - 1) / SIMT_STEP;
    uint32_t resultsAll[16] = {0U};

    for (uint64_t i = idx * Vec; i < outputSize; i += blockDim.x * gridDim.x * Vec) {
        for (uint32_t randIdx = 0; randIdx < randRounds; randIdx++) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);

            ThreadMappingAndSkip<Vec, CONTINUOUS_USE>(i + randIdx * SIMT_STEP, counterTmp, magic, shift, totalThreads);

            uint32_t results[SIMT_STEP];
            PhiloxRandomSimt(key, counterTmp, results);
            for (uint32_t j = 0; j < SIMT_STEP && randIdx * SIMT_STEP + j < Vec; j++) {
                resultsAll[randIdx * SIMT_STEP + j] = results[j];
            }
        }

        for (uint32_t j = 0; j < Vec; j++) {
            uint64_t li = i + j;
            if (li < outputSize) {
                transform(outputGm, li, resultsAll, j);
            }
        }
    }
}

template <typename T, typename TransformFunc, uint32_t ThreadNum = DEFAULT_SIMT_THREAD_NUM, uint32_t UNROLL = 4>
__simt_vf__ __aicore__ LAUNCH_BOUND(ThreadNum) inline void PhiloxSimtKernelDiscontinuous(
    __gm__ volatile T* outputGm, int64_t offset, int64_t seed, uint64_t outputSize, uint64_t magic, uint64_t shift,
    uint64_t totalThreads, TransformFunc transform)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    PhiloxAlgParsInit(key, counter, seed, offset);

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t repeatTime = (outputSize + totalThreads * UNROLL - 1) / (totalThreads * UNROLL);

    for (uint64_t loopIdx = 0; loopIdx < repeatTime; loopIdx++) {
        for (uint64_t linearIndex = idx; linearIndex < totalThreads; linearIndex += blockDim.x * gridDim.x) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);

            FlashCounter(linearIndex, loopIdx, counterTmp);

            uint32_t results[SIMT_STEP];
            PhiloxRandomSimt(key, counterTmp, results);
            for (uint32_t iStep = 0; iStep < UNROLL; iStep++) {
                uint64_t li = linearIndex + loopIdx * totalThreads * UNROLL + totalThreads * iStep;
                if (li < outputSize) {
                    transform(outputGm, li, results, iStep, SIMT_STEP / UNROLL);
                }
            }
        }
    }
}

template <typename Launcher>
__aicore__ inline void ProcessWithSplitBlocks(const RandomUnifiedSimtTilingDataStruct* __restrict tilingData,
                                              Launcher& launcher)
{
    for (int64_t blockIdx = 0; blockIdx < tilingData->splitBlockCount; blockIdx++) {
        const SplitBlockInfo& block = tilingData->splitBlocks[blockIdx];

        ExecutionPolicyKernel policy;
        GetUintDivMagicAndShift(policy.magic, policy.shift, static_cast<uint64_t>(block.totalThreads));

        launcher(policy, block.gmOffset, block.kernelOffset, block.numel, block.grid, block.totalThreads);
    }
}

} // namespace RandomKernelBase

/*
 * 其他算子接入SIMT Kernel模板说明：
 *
 * 1. Tiling侧配置：
 *    OpTilingConfig config;
 *    config.kernelMode = RandomKernelMode::SIMT;
 *    config.enableSplitBlocks = true;
 *
 * 2. Kernel侧定义TransformFunc：
 *    template <typename T>
 *    struct MyTransform {
 *        float param_;
 *        __aicore__ MyTransform(float param) : param_(param) {}
 *        __simt_callee__ __aicore__ inline void operator()(
 *            __gm__ volatile T* outputGm, uint64_t li, const uint32_t* results, uint32_t iStep,
 *            [[maybe_unused]] uint32_t randPerOutput = 1)
 *        {
 *            // results[iStep] 是 uint32 随机数，需自行转换为 float
 *            float u = results[iStep] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
 *            float val = logf(u);
 *            outputGm[li] = static_cast<T>(val * param_);
 *        }
 *    };
 *
 * 3. TransformFunc 参数说明：
 *    - outputGm: 输出 Global Memory 地址
 *    - li: 当前写入的线性索引
 *    - results: Philox 生成的 uint32 随机数数组（大小为 SIMT_STEP=4）
 *    - iStep: 当前处理的元素索引（0 ~ UNROLL-1）
 *    - randPerOutput: 每个输出消耗的 Philox 输出数量（SIMT_STEP/UNROLL，默认=1）
 *
 * 4. 调用模板：
 *    AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, MyTransform<T>>>(
 *        AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM),
 *        outputGm, kernelOffset, seed, numel,
 *        policy.magic, policy.shift, totalThreads, transform);
 *
 * 5. 切分执行（如果 enableSplitBlocks=true）：
 *    struct MyLauncher {
 *        __aicore__ inline void operator()(
 *            const ExecutionPolicyKernel& policy, int64_t gmOffset, int64_t kernelOffset,
 *            int64_t numel, int64_t grid, int64_t totalThreads)
 *        {
 *            __gm__ volatile T* gmPtr = (__gm__ volatile T*)baseAddr + gmOffset;
 *            MyTransform<T> transform(param);
 *            AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, MyTransform<T>>>(
 *                AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM),
 *                gmPtr, kernelOffset, seed, numel,
 *                policy.magic, policy.shift, totalThreads, transform);
 *        }
 *    };
 *    ProcessWithSplitBlocks(tilingData, launcher);
 */

#endif
