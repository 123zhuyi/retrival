/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_V3_SIMD_IMPL_H
#define DROP_OUT_V3_SIMD_IMPL_H

#include "../../random_common/arch35/random_kernel_base.h"
#include "drop_out_v3_tiling_data_arch35.h"
#include "simt_api/asc_simt.h"
#include "adv_api/math/philox.h"

namespace DropOutV3 {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr static uint32_t IDX_0 = 0;
constexpr static uint32_t IDX_1 = 1;
constexpr static uint32_t NUM_3 = 3;
constexpr static uint32_t NUM_16 = 16;
constexpr static int64_t ALIGNMENT_32 = 32;
constexpr static uint16_t CORE_THREAD_NUMBER = 512;
constexpr static float HIFLOAT_MULS = 65536.0f;

// ==================== SIMT Kernels ====================

template <typename T, int32_t VEC>
__simt_vf__ __aicore__ LAUNCH_BOUND(CORE_THREAD_NUMBER) inline void SimtDropOutComputeContinuous(
    __ubuf__ volatile T* inputUb, __ubuf__ volatile T* outputUb, __ubuf__ volatile float* randomFloatUb,
    uint64_t totalThreads, uint64_t magic, uint64_t shift, int64_t elementNum, int64_t baseLinearIndex, int64_t seed,
    int64_t offset, float p)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    float scale = 1.0f / p;
    PhiloxAlgParsInit(key, counter, seed, offset);
    int64_t idx = threadIdx.x;
    int64_t launchThreads = blockDim.x;
    constexpr int32_t pkgElemNum = NUM_4;
    constexpr int32_t randNum = NUM_4 - (VEC % NUM_4);
    constexpr int32_t randLoop = pkgElemNum / randNum;
    for (int64_t ubIdx = idx * pkgElemNum; ubIdx < elementNum; ubIdx += launchThreads * pkgElemNum) {
        T input[pkgElemNum] = {0.0};
        T output[pkgElemNum] = {0.0};
        float randValues[pkgElemNum] = {0.0f};
        if (sizeof(T) == NUM_2) {
            reinterpret_cast<float2*>(input)[0] = reinterpret_cast<__ubuf__ volatile float2*>(inputUb + ubIdx)[0];
        } else {
            reinterpret_cast<float4*>(input)[0] = reinterpret_cast<__ubuf__ volatile float4*>(inputUb + ubIdx)[0];
        }

        for (uint8_t randIdx = 0; randIdx < randLoop; randIdx++) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
            ThreadMappingAndSkip<VEC, CONTINUOUS_USE>(baseLinearIndex + ubIdx + randIdx * randNum, counterTmp, magic,
                                                      shift, totalThreads);
            float results[ALG_COUNTER_SIZE];
            PhiloxRandomSimt(key, counterTmp, results);
            for (uint8_t i = 0; i < randNum; i++) {
                float fMaskBit = (results[i] < p) ? 1.0f : 0.0f;
                output[randIdx * randNum + i] = input[randIdx * randNum + i] * fMaskBit * scale;
                randValues[randIdx * randNum + i] = results[i];
            }
        }

        if (sizeof(T) == NUM_2) {
            reinterpret_cast<__ubuf__ volatile float2*>(outputUb + ubIdx)[0] = reinterpret_cast<float2*>(output)[0];
        } else {
            reinterpret_cast<__ubuf__ volatile float4*>(outputUb + ubIdx)[0] = reinterpret_cast<float4*>(output)[0];
        }
        reinterpret_cast<__ubuf__ volatile float4*>(randomFloatUb +
                                                    ubIdx)[0] = reinterpret_cast<float4*>(randValues)[0];
    }
}

// ==================== SIMD Kernels ====================

static constexpr Reg::CastTrait castTraitB16ToB32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
static constexpr Reg::CastTrait castTraitB32ToB16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitI32ToF32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                     Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
__simd_callee__ inline void PhiloxCtrConvertAndDropout(Reg::RegTensor<uint32_t>& currCtr,
                                                       Reg::RegTensor<uint32_t>& mask16Reg,
                                                       Reg::RegTensor<float>& zeroFloatReg, uint32_t currCount,
                                                       float scale, float prob, __ubuf__ float* randomFloatPtr,
                                                       __ubuf__ T* inputPtr, __ubuf__ T* outputPtr)
{
    Reg::MaskReg mask = Reg::UpdateMask<float>(currCount);

    Reg::RegTensor<uint32_t> highReg, lowReg;
    Reg::ShiftRights(highReg, currCtr, static_cast<int16_t>(NUM_16), mask);
    Reg::And(lowReg, currCtr, mask16Reg, mask);

    Reg::RegTensor<float> highFloat, lowFloat;
    Reg::Cast<float, int32_t, castTraitI32ToF32>(highFloat, (Reg::RegTensor<int32_t>&)highReg, mask);
    Reg::Cast<float, int32_t, castTraitI32ToF32>(lowFloat, (Reg::RegTensor<int32_t>&)lowReg, mask);

    Reg::RegTensor<float> combined;
    Reg::Muls<float>(combined, highFloat, HIFLOAT_MULS, mask);
    Reg::Add<float>(combined, combined, lowFloat, mask);

    Reg::RegTensor<float> randReg;
    Reg::Muls<float>(randReg, combined, RAND_2POW32_INV, mask);
    Reg::Adds<float>(randReg, randReg, RAND_2POW32_INV_HALF, mask);

    Reg::StoreAlign<float>(randomFloatPtr, randReg, mask);

    Reg::MaskReg dropoutMask;
    Reg::CompareScalar<float, CMPMODE::LT>(dropoutMask, randReg, prob, mask);

    Reg::RegTensor<float> inputFloatReg;
    if constexpr (IsSameType<T, float>::value) {
        Reg::LoadAlign<float>(inputFloatReg, inputPtr);
    } else {
        Reg::RegTensor<T> inputReg;
        Reg::LoadAlign<T, Reg::LoadDist::DIST_UNPACK_B16>(inputReg, inputPtr);
        Reg::Cast<float, T, castTraitB16ToB32>(inputFloatReg, inputReg, mask);
    }

    Reg::RegTensor<float> scaledInput;
    Reg::Muls<float>(scaledInput, inputFloatReg, scale, mask);
    Reg::RegTensor<float> outputFloatReg;
    Reg::Select<float>(outputFloatReg, scaledInput, zeroFloatReg, dropoutMask);

    if constexpr (IsSameType<T, float>::value) {
        Reg::StoreAlign<float>(outputPtr, outputFloatReg, mask);
    } else {
        Reg::RegTensor<T> outputReg;
        Reg::Cast<T, float, castTraitB32ToB16>(outputReg, outputFloatReg, mask);
        Reg::StoreAlign<T, Reg::StoreDist::DIST_PACK_B32>(outputPtr, outputReg, mask);
    }
}

template <int32_t VEC>
__simd_callee__ inline void VectorThreadMappingAndSkip(int64_t baseIndex, uint32_t randNumVal, uint32_t magic32,
                                                       uint32_t shift32, uint32_t totalThreads32,
                                                       const uint32_t* initCounter, Reg::RegTensor<uint32_t>& ctr0,
                                                       Reg::RegTensor<uint32_t>& ctr1, Reg::RegTensor<uint32_t>& ctr2,
                                                       Reg::RegTensor<uint32_t>& ctr3, Reg::MaskReg& pg)
{
    Reg::RegTensor<int32_t> idxVecInt;
    Reg::Arange(idxVecInt, 0);
    Reg::Muls(idxVecInt, idxVecInt, static_cast<int32_t>(randNumVal), pg);
    Reg::RegTensor<int32_t> baseReg;
    Reg::Duplicate(baseReg, static_cast<int32_t>(baseIndex));
    Reg::Add(idxVecInt, idxVecInt, baseReg, pg);

    Reg::RegTensor<uint32_t>& idxVec = (Reg::RegTensor<uint32_t>&)idxVecInt;

    constexpr int16_t log2Vec = (VEC == VEC_2) ? 1 : (VEC == VEC_4) ? NUM_2 : (VEC == VEC_8) ? NUM_3 : NUM_4;
    Reg::RegTensor<uint32_t> groupIdxVec;
    Reg::ShiftRights(groupIdxVec, idxVec, log2Vec, pg);

    Reg::RegTensor<uint32_t> magicReg;
    Reg::Duplicate(magicReg, magic32);
    Reg::RegTensor<uint32_t> mullLow, mullHigh, sum, repeat;
    Reg::Mull(mullLow, mullHigh, groupIdxVec, magicReg, pg);
    Reg::Add(sum, groupIdxVec, mullHigh, pg);
    Reg::ShiftRights(repeat, sum, static_cast<int16_t>(shift32), pg);

    Reg::RegTensor<uint32_t> gtiProduct, gtiVec;
    Reg::Muls(gtiProduct, repeat, totalThreads32, pg);
    Reg::Sub(gtiVec, groupIdxVec, gtiProduct, pg);

    if constexpr (VEC == NUM_8 || VEC == NUM_16) {
        constexpr uint32_t vecDiv4 = VEC / NUM_4;
        Reg::RegTensor<uint32_t> idxDiv4, idxMod, maskModReg;
        Reg::ShiftRights(idxDiv4, idxVec, static_cast<int16_t>(NUM_2), pg);
        Reg::Duplicate(maskModReg, vecDiv4 - 1);
        Reg::And(idxMod, idxDiv4, maskModReg, pg);
        Reg::Muls(repeat, repeat, vecDiv4, pg);
        Reg::Add(repeat, repeat, idxMod, pg);
    }

    Reg::RegTensor<uint32_t> gtiHi, repeatHi, zeroReg;
    Reg::Duplicate(gtiHi, 0);
    Reg::Duplicate(repeatHi, 0);
    Reg::Duplicate(zeroReg, 0);

    Reg::Duplicate(ctr0, initCounter[IDX_0]);
    Reg::Duplicate(ctr1, initCounter[IDX_1]);
    Reg::Duplicate(ctr2, initCounter[IDX_2]);
    Reg::Duplicate(ctr3, initCounter[IDX_3]);

    Reg::MaskReg carry;
    Reg::AddCarryOut(carry, ctr2, ctr2, gtiVec, pg);
    Reg::AddCarryOuts(carry, ctr3, ctr3, gtiHi, carry, pg);

    Reg::MaskReg carry2;
    Reg::AddCarryOut(carry2, ctr0, ctr0, repeat, pg);
    Reg::AddCarryOuts(carry2, ctr1, ctr1, repeatHi, carry2, pg);
    Reg::MaskReg carry3;
    Reg::AddCarryOuts(carry3, ctr2, ctr2, zeroReg, carry2, pg);
    Reg::AddCarryOuts(carry3, ctr3, ctr3, zeroReg, carry3, pg);
}

// ==================== Main Implementation ====================

template <typename T, typename U>
class DropOutV3SimdImpl {
public:
    __aicore__ inline DropOutV3SimdImpl(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mask, const DropOutV3TilingDataStruct* tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process(const DropOutV3TilingDataStruct* tilingData);
    __aicore__ inline bool IsProbEqual(float a, float b);

private:
    __aicore__ inline void ProcessContinuous(const DropOutV3TilingDataStruct* tilingData);
    __aicore__ inline void ProcessProbZero(const DropOutV3TilingDataStruct* tilingData);

    __aicore__ inline void CopyInContinuous(int64_t gmOffset, int64_t currElements);
    __aicore__ inline void ComputeContinuous(int64_t baseLinearIndex, int64_t currElements,
                                             const DropOutV3TilingDataStruct* tilingData);
    template <int32_t VEC>
    __aicore__ inline void ComputeContinuousSimd(int64_t baseLinearIndex, int64_t currElements,
                                                 const DropOutV3TilingDataStruct* tilingData, LocalTensor<T>& inputUb,
                                                 LocalTensor<T>& outputUb, LocalTensor<uint8_t>& maskBitUb,
                                                 LocalTensor<float>& randomFloatUb);
    __aicore__ inline void CopyOutContinuous(int64_t gmOffset, int64_t currElements);

    __aicore__ inline int64_t GetCurrElements(bool isTailCore, bool isTailLoop,
                                              const DropOutV3TilingDataStruct* tilingData);

private:
    TPipe* pipe_ = nullptr;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<uint8_t> maskGm_;

    TQue<QuePosition::VECIN, NUM_2> inputQue_;
    TQue<QuePosition::VECOUT, NUM_2> outputQue_;
    TQue<QuePosition::VECOUT, NUM_2> maskBitQue_;
    TBuf<TPosition::VECCALC> randomFloatBuf_;

    float prob_ = 0.0f;
    uint32_t blockIdx_ = 0;

    uint32_t totalThreads_ = 0;
    uint32_t magic32_ = 0;
    uint32_t shift32_ = 0;
    uint64_t magic64_ = 0;
    uint64_t shift64_ = 0;
    uint32_t vec_ = 0;
};

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR mask,
                                                     const DropOutV3TilingDataStruct* tilingData, TPipe* pipe)
{
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    maskGm_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    pipe_ = pipe;

    prob_ = tilingData->prob;
    vec_ = tilingData->vec;
    totalThreads_ = tilingData->totalThreads;

    GetUintDivMagicAndShift<uint32_t>(magic32_, shift32_, totalThreads_);
    GetUintDivMagicAndShift<uint64_t>(magic64_, shift64_, static_cast<uint64_t>(totalThreads_));
    blockIdx_ = GetBlockIdx();

    int64_t sizeofT = static_cast<int64_t>(sizeof(T));
    int64_t ubFactor = tilingData->ubFactorElements;
    int64_t inputBufSize = Ops::Base::CeilAlign(ubFactor * sizeofT, ALIGNMENT_32);
    int64_t outputBufSize = Ops::Base::CeilAlign(ubFactor * sizeofT, ALIGNMENT_32);
    int64_t maskBitBufSize = Ops::Base::CeilAlign(ubFactor / NUM_8, ALIGNMENT_32);
    int64_t randomFloatBufSize = Ops::Base::CeilAlign(ubFactor * static_cast<int64_t>(sizeof(float)), ALIGNMENT_32);
    pipe_->InitBuffer(inputQue_, NUM_2, inputBufSize);
    pipe_->InitBuffer(outputQue_, NUM_2, outputBufSize);
    pipe_->InitBuffer(maskBitQue_, NUM_2, maskBitBufSize);
    pipe_->InitBuffer(randomFloatBuf_, randomFloatBufSize);
}

template <typename T, typename U>
__aicore__ inline int64_t DropOutV3SimdImpl<T, U>::GetCurrElements(bool isTailCore, bool isTailLoop,
                                                                   const DropOutV3TilingDataStruct* tilingData)
{
    if (isTailCore) {
        return isTailLoop ? tilingData->tailCoreTailUbFactorElements : tilingData->ubFactorElements;
    }
    return isTailLoop ? tilingData->tailUbFactorElements : tilingData->ubFactorElements;
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::CopyInContinuous(int64_t gmOffset, int64_t currElements)
{
    LocalTensor<T> inputUb = inputQue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(currElements * sizeof(T)),
                                 static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams{
        true, static_cast<uint8_t>(0),
        static_cast<uint8_t>(Ops::Base::CeilAlign(currElements, static_cast<int64_t>(NUM_8)) - currElements),
        static_cast<uint8_t>(0)};
    DataCopyPad(inputUb, xGm_[gmOffset], copyParams, padParams);
    inputQue_.EnQue(inputUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::ComputeContinuous(int64_t baseLinearIndex, int64_t currElements,
                                                                  const DropOutV3TilingDataStruct* tilingData)
{
    LocalTensor<T> inputUb = inputQue_.template DeQue<T>();
    LocalTensor<T> outputUb = outputQue_.template AllocTensor<T>();
    LocalTensor<uint8_t> maskBitUb = maskBitQue_.template AllocTensor<uint8_t>();
    LocalTensor<float> randomFloatUb = randomFloatBuf_.Get<float>();

    if (tilingData->outputSize >= 524288) {
        __ubuf__ volatile float* randomFloatPtr = (__ubuf__ volatile float*)randomFloatUb.GetPhyAddr();
        switch (vec_) {
            case VEC_16:
                asc_vf_call<SimtDropOutComputeContinuous<T, static_cast<int32_t>(VEC_16)>>(
                    dim3(CORE_THREAD_NUMBER), (__ubuf__ volatile T*)inputUb.GetPhyAddr(),
                    (__ubuf__ volatile T*)outputUb.GetPhyAddr(), randomFloatPtr, totalThreads_, magic64_, shift64_,
                    currElements, baseLinearIndex, tilingData->seed, tilingData->offset, prob_);
                break;
            case VEC_8:
                asc_vf_call<SimtDropOutComputeContinuous<T, static_cast<int32_t>(VEC_8)>>(
                    dim3(CORE_THREAD_NUMBER), (__ubuf__ volatile T*)inputUb.GetPhyAddr(),
                    (__ubuf__ volatile T*)outputUb.GetPhyAddr(), randomFloatPtr, totalThreads_, magic64_, shift64_,
                    currElements, baseLinearIndex, tilingData->seed, tilingData->offset, prob_);
                break;
            case VEC_4:
                asc_vf_call<SimtDropOutComputeContinuous<T, static_cast<int32_t>(VEC_4)>>(
                    dim3(CORE_THREAD_NUMBER), (__ubuf__ volatile T*)inputUb.GetPhyAddr(),
                    (__ubuf__ volatile T*)outputUb.GetPhyAddr(), randomFloatPtr, totalThreads_, magic64_, shift64_,
                    currElements, baseLinearIndex, tilingData->seed, tilingData->offset, prob_);
                break;
            case VEC_2:
                asc_vf_call<SimtDropOutComputeContinuous<T, static_cast<int32_t>(VEC_2)>>(
                    dim3(CORE_THREAD_NUMBER), (__ubuf__ volatile T*)inputUb.GetPhyAddr(),
                    (__ubuf__ volatile T*)outputUb.GetPhyAddr(), randomFloatPtr, totalThreads_, magic64_, shift64_,
                    currElements, baseLinearIndex, tilingData->seed, tilingData->offset, prob_);
                break;
            default:
                break;
        }
        uint32_t countAlign256 = Ops::Base::CeilAlign(static_cast<uint32_t>(currElements),
                                                      static_cast<uint32_t>(NUM_256));
        AscendC::CompareScalar<float, uint8_t>(maskBitUb, randomFloatUb, prob_, CMPMODE::LT, countAlign256);
    } else {
        switch (vec_) {
            case VEC_16:
                ComputeContinuousSimd<static_cast<int32_t>(VEC_16)>(baseLinearIndex, currElements, tilingData, inputUb,
                                                                    outputUb, maskBitUb, randomFloatUb);
                break;
            case VEC_8:
                ComputeContinuousSimd<static_cast<int32_t>(VEC_8)>(baseLinearIndex, currElements, tilingData, inputUb,
                                                                   outputUb, maskBitUb, randomFloatUb);
                break;
            case VEC_4:
                ComputeContinuousSimd<static_cast<int32_t>(VEC_4)>(baseLinearIndex, currElements, tilingData, inputUb,
                                                                   outputUb, maskBitUb, randomFloatUb);
                break;
            case VEC_2:
                ComputeContinuousSimd<static_cast<int32_t>(VEC_2)>(baseLinearIndex, currElements, tilingData, inputUb,
                                                                   outputUb, maskBitUb, randomFloatUb);
                break;
            default:
                break;
        }
    }
    outputQue_.EnQue(outputUb);
    maskBitQue_.EnQue(maskBitUb);
    inputQue_.FreeTensor(inputUb);
}

template <typename T, typename U>
template <int32_t VEC>
__aicore__ inline void DropOutV3SimdImpl<T, U>::ComputeContinuousSimd(int64_t baseLinearIndex, int64_t currElements,
                                                                      const DropOutV3TilingDataStruct* tilingData,
                                                                      LocalTensor<T>& inputUb, LocalTensor<T>& outputUb,
                                                                      LocalTensor<uint8_t>& maskBitUb,
                                                                      LocalTensor<float>& randomFloatUb)
{
    constexpr uint32_t randNum = NUM_4 - (VEC % NUM_4);
    constexpr uint32_t counterPerBatch = static_cast<uint32_t>(PhiloxInternal::ELE_CNT_B32_ONCE);
    constexpr uint32_t elemPerBatch = counterPerBatch * randNum;

    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    key[0] = static_cast<uint32_t>(tilingData->seed);
    key[1] = static_cast<uint32_t>(static_cast<uint64_t>(tilingData->seed) >> RIGHT_SHIFT);
    uint64_t skipOffset = Ops::Base::CeilDiv(tilingData->offset, static_cast<int64_t>(VEC_4));
    counter[0] = static_cast<uint32_t>(skipOffset);
    counter[1] = static_cast<uint32_t>(skipOffset >> RIGHT_SHIFT);

    float scale = 1.0f / prob_;

    uint32_t totalBatches = Ops::Base::CeilDiv(static_cast<uint32_t>(currElements), elemPerBatch);

    __ubuf__ T* inputPtr = (__ubuf__ T*)inputUb.GetPhyAddr();
    __ubuf__ T* outputPtr = (__ubuf__ T*)outputUb.GetPhyAddr();
    __ubuf__ float* randomFloatPtr = (__ubuf__ float*)randomFloatUb.GetPhyAddr();

    __VEC_SCOPE__
    {
        Reg::MaskReg pg = Reg::CreateMask<uint32_t>();

        Reg::RegTensor<uint32_t> cMul0, cMul1;
        Reg::Duplicate(cMul0, PhiloxInternal::CONST_MUL_0);
        Reg::Duplicate(cMul1, PhiloxInternal::CONST_MUL_1);

        Reg::RegTensor<float> zeroFloatReg;
        Reg::Duplicate(zeroFloatReg, 0.0f);

        Reg::RegTensor<uint32_t> mask16Reg;
        Reg::Duplicate(mask16Reg, 0xFFFF);

        for (uint16_t batch = 0; batch < static_cast<uint16_t>(totalBatches); batch++) {
            // 映射counter
            uint32_t batchStart = batch * elemPerBatch;
            int64_t baseIndex = baseLinearIndex + static_cast<int64_t>(batchStart);

            Reg::RegTensor<uint32_t> key0, key1;
            Reg::Duplicate(key0, key[0]);
            Reg::Duplicate(key1, key[1]);
            Reg::RegTensor<uint32_t> ctr0, ctr1, ctr2, ctr3;
            VectorThreadMappingAndSkip<VEC>(baseIndex, randNum, magic32_, shift32_, totalThreads_, counter, ctr0, ctr1,
                                            ctr2, ctr3, pg);

            // 生成随机数
            Reg::RegTensor<uint32_t> tmpL0, tmpH0, tmpL1, tmpH1;
            SpNetworkKernel<10>(tmpL0, tmpH0, tmpL1, tmpH1, ctr0, ctr1, ctr2, ctr3, key0, key1, cMul0, cMul1, pg);

            if constexpr (randNum == NUM_4) {
                Interleave(ctr0, ctr2, ctr0, ctr2);
                Interleave(ctr1, ctr3, ctr1, ctr3);
                Interleave(ctr0, ctr1, ctr0, ctr1);
                Interleave(ctr2, ctr3, ctr2, ctr3);
            } else {
                Interleave(ctr0, ctr1, ctr0, ctr1);
            }

            // drop out
            __ubuf__ T* batchInputPtr = inputPtr + batchStart;
            __ubuf__ T* batchOutputPtr = outputPtr + batchStart;
            __ubuf__ float* batchRandomPtr = randomFloatPtr + batchStart;

            __ubuf__ T* iPtr0 = batchInputPtr;
            __ubuf__ T* iPtr1 = batchInputPtr + counterPerBatch;
            __ubuf__ T* oPtr0 = batchOutputPtr;
            __ubuf__ T* oPtr1 = batchOutputPtr + counterPerBatch;
            __ubuf__ float* rPtr0 = batchRandomPtr;
            __ubuf__ float* rPtr1 = batchRandomPtr + counterPerBatch;

            uint32_t remaining = (batchStart + elemPerBatch <= static_cast<uint32_t>(currElements)) ?
                                     elemPerBatch :
                                     static_cast<uint32_t>(currElements) - batchStart;

            uint32_t currCount0 = (remaining < counterPerBatch) ? remaining : counterPerBatch;
            remaining -= currCount0;
            uint32_t currCount1 = (remaining < counterPerBatch) ? remaining : counterPerBatch;
            remaining -= currCount1;

            PhiloxCtrConvertAndDropout<T>(ctr0, mask16Reg, zeroFloatReg, currCount0, scale, prob_, rPtr0, iPtr0, oPtr0);
            PhiloxCtrConvertAndDropout<T>(ctr1, mask16Reg, zeroFloatReg, currCount1, scale, prob_, rPtr1, iPtr1, oPtr1);

            if constexpr (randNum == NUM_4) {
                __ubuf__ T* iPtr2 = batchInputPtr + NUM_2 * counterPerBatch;
                __ubuf__ T* iPtr3 = batchInputPtr + NUM_3 * counterPerBatch;
                __ubuf__ T* oPtr2 = batchOutputPtr + NUM_2 * counterPerBatch;
                __ubuf__ T* oPtr3 = batchOutputPtr + NUM_3 * counterPerBatch;
                __ubuf__ float* rPtr2 = batchRandomPtr + NUM_2 * counterPerBatch;
                __ubuf__ float* rPtr3 = batchRandomPtr + NUM_3 * counterPerBatch;

                uint32_t currCount2 = (remaining < counterPerBatch) ? remaining : counterPerBatch;
                remaining -= currCount2;
                uint32_t currCount3 = (remaining < counterPerBatch) ? remaining : counterPerBatch;

                PhiloxCtrConvertAndDropout<T>(ctr2, mask16Reg, zeroFloatReg, currCount2, scale, prob_, rPtr2, iPtr2,
                                              oPtr2);
                PhiloxCtrConvertAndDropout<T>(ctr3, mask16Reg, zeroFloatReg, currCount3, scale, prob_, rPtr3, iPtr3,
                                              oPtr3);
            }
        }
    }

    CompareScalar<float, uint8_t>(maskBitUb, randomFloatUb, prob_, CMPMODE::LT, currElements);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::CopyOutContinuous(int64_t gmOffset, int64_t currElements)
{
    LocalTensor<T> outputUb = outputQue_.template DeQue<T>();
    LocalTensor<uint8_t> maskBitUb = maskBitQue_.template DeQue<uint8_t>();

    DataCopyExtParams copyParamsY{static_cast<uint16_t>(1), static_cast<uint32_t>(currElements * sizeof(T)),
                                  static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(yGm_[gmOffset], outputUb, copyParamsY);

    int64_t maskBytes = Ops::Base::CeilDiv(currElements, static_cast<int64_t>(NUM_8));
    DataCopyExtParams copyParamsMask{static_cast<uint16_t>(1), static_cast<uint32_t>(maskBytes * sizeof(uint8_t)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPad(maskGm_[gmOffset / NUM_8], maskBitUb, copyParamsMask);

    outputQue_.FreeTensor(outputUb);
    maskBitQue_.FreeTensor(maskBitUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::ProcessContinuous(const DropOutV3TilingDataStruct* tilingData)
{
    int64_t gmOffset = static_cast<int64_t>(blockIdx_) * tilingData->perCoreElements;
    bool isTailCore = (blockIdx_ == tilingData->usedCoreNum - 1);
    int64_t loopCount = isTailCore ? tilingData->tailUbLoopCount : tilingData->ubLoopCount;

    for (int64_t loop = 0; loop < loopCount; loop++) {
        bool isTailLoop = (loop == loopCount - 1);
        int64_t currElements = GetCurrElements(isTailCore, isTailLoop, tilingData);

        CopyInContinuous(gmOffset, currElements);
        ComputeContinuous(gmOffset, currElements, tilingData);
        CopyOutContinuous(gmOffset, currElements);
        gmOffset += currElements;
    }
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::ProcessProbZero(const DropOutV3TilingDataStruct* tilingData)
{
    int64_t gmOffset = static_cast<int64_t>(blockIdx_) * tilingData->perCoreElements;
    bool isTailCore = (blockIdx_ == tilingData->usedCoreNum - 1);
    int64_t loopCount = isTailCore ? tilingData->tailUbLoopCount : tilingData->ubLoopCount;

    for (int64_t loop = 0; loop < loopCount; loop++) {
        bool isTailLoop = (loop == loopCount - 1);
        int64_t currElements = GetCurrElements(isTailCore, isTailLoop, tilingData);

        LocalTensor<T> outputUb = outputQue_.template AllocTensor<T>();
        LocalTensor<uint8_t> maskBitUb = maskBitQue_.template AllocTensor<uint8_t>();
        Duplicate(outputUb, static_cast<T>(0), currElements);
        Duplicate(maskBitUb, static_cast<uint8_t>(0), Ops::Base::CeilDiv(currElements, static_cast<int64_t>(NUM_8)));

        outputQue_.EnQue(outputUb);
        maskBitQue_.EnQue(maskBitUb);

        CopyOutContinuous(gmOffset, currElements);

        gmOffset += currElements;
    }
}

template <typename T, typename U>
__aicore__ inline bool DropOutV3SimdImpl<T, U>::IsProbEqual(float a, float b)
{
    return std::abs(a - b) <= double_epsilon;
}

template <typename T, typename U>
__aicore__ inline void DropOutV3SimdImpl<T, U>::Process(const DropOutV3TilingDataStruct* tilingData)
{
    if (blockIdx_ >= tilingData->usedCoreNum) {
        return;
    }

    if (blockIdx_ == 0) {
        constexpr int64_t BIT_NUMBER = 128;
        constexpr int64_t UINT8_BIT_NUMBER = 8;
        int64_t maskWrittenBytes = Ops::Base::CeilDiv(tilingData->outputSize, UINT8_BIT_NUMBER);
        int64_t maskTotalBytes = Ops::Base::CeilAlign(tilingData->outputSize, BIT_NUMBER) / UINT8_BIT_NUMBER;
        int64_t tailOffset = Ops::Base::FloorAlign(maskWrittenBytes, (int64_t)NUM_2);
        int64_t tailBytes = maskTotalBytes - tailOffset;
        if (tailBytes > 0) {
            GlobalTensor<uint16_t> maskGmU16;
            maskGmU16.SetGlobalBuffer((__gm__ uint16_t*)maskGm_.GetPhyAddr());
            GlobalTensor<uint16_t> maskGmTail = maskGmU16[tailOffset / NUM_2];
            Fill<uint16_t>(maskGmTail, tailBytes / NUM_2, 0);
        }
    }
    SyncAll();

    if (IsProbEqual(prob_, 0.0f)) {
        ProcessProbZero(tilingData);
        return;
    }

    ProcessContinuous(tilingData);
}
} // namespace DropOutV3
#endif // DROP_OUT_V3_SIMD_IMPL_H
