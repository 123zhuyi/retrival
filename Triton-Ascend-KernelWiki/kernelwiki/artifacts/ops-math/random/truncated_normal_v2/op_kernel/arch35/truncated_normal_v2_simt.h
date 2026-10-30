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
 * \file truncated_normal_v2_simt.h
 * \brief truncated_normal_v2_simt
 */

#ifndef RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_
#define RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace TruncatedNormalV2 {
using namespace AscendC;

constexpr uint64_t RESERVED_SAMPLES_PER_OUTPUT = 256;
constexpr uint64_t GROUP_SIZE = 4;
constexpr float RANDOM_THREAD_R = 2.0f;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr uint32_t SHIFT_BITS = 32;
constexpr int IDX_0 = 0;
constexpr int IDX_1 = 1;
constexpr int IDX_2 = 2;
constexpr int IDX_3 = 3;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 128;
constexpr uint32_t THREAD_LAUNCH = 512;
#else
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t THREAD_LAUNCH = 512;
#endif

__simt_callee__ __aicore__ inline float Uint32ToFloat(const uint32_t x)
{
    constexpr uint32_t MANTISSA_BIT = 23;
    const uint32_t man = x & 0x7fffffu;
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << MANTISSA_BIT) | man;
    float result = *reinterpret_cast<const float*>(&val);
    return result - 1.0f;
}

__simt_callee__ __aicore__ inline void FilterSample(float* results, int& index, float f0)
{
    if (fabsf(f0) < RANDOM_THREAD_R) {
        results[index] = f0;
        index = index + 1;
    }
}

__simt_callee__ __aicore__ inline void GenSamples(float* results, const uint32_t* key, const uint32_t* counter)
{
    int index = 0;
    uint32_t counterSkip[ALG_COUNTER_SIZE];
    RandomKernelBase::CopyArray<ALG_COUNTER_SIZE>(counterSkip, counter);
    while (index < static_cast<int>(GROUP_SIZE)) {
        uint32_t counterRst[ALG_COUNTER_SIZE];
        RandomKernelBase::PhiloxRandomSimt(key, counterSkip, counterRst);
        RandomKernelBase::SkipOne(counterSkip);

        // Repeatedly take samples from the normal distribution, until we have
        // the desired number of elements that fall within the pre-defined cutoff
        // threshold.
        float f[2];
        RandomKernelBase::BoxMullerFloatSafe(Uint32ToFloat(counterRst[IDX_0]), Uint32ToFloat(counterRst[IDX_1]), &f[0], &f[1]);
        FilterSample(results, index, f[0]);
        if (index >= static_cast<int>(GROUP_SIZE)) {
            return;
        }
        FilterSample(results, index, f[1]);
        if (index >= static_cast<int>(GROUP_SIZE)) {
            return;
        }

        RandomKernelBase::BoxMullerFloatSafe(Uint32ToFloat(counterRst[IDX_2]), Uint32ToFloat(counterRst[IDX_3]), &f[0], &f[1]);
        FilterSample(results, index, f[0]);
        if (index >= static_cast<int>(GROUP_SIZE)) {
            return;
        }
        FilterSample(results, index, f[1]);
        if (index >= static_cast<int>(GROUP_SIZE)) {
            return;
        }
    }
}

template <typename Y_T, typename OFFSET_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void SimtCompute(__gm__ Y_T* yGm,
    int64_t offsetValue, int64_t outputNum, uint32_t key0, uint32_t key1, uint32_t counter2, uint32_t counter3)
{
    uint32_t key[ALG_KEY_SIZE] = {key0, key1};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, counter2, counter3};

    int64_t groupIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t totalThreadCount = gridDim.x * blockDim.x;
    int64_t offset = groupIndex * static_cast<int64_t>(GROUP_SIZE);

    // the op is stateful, offsetGm saved the previous offset of counter
    if (offsetValue > 0) {
        RandomKernelBase::SkipLo(counter, static_cast<uint64_t>(offsetValue));
    }
    while (offset < outputNum) {
        uint32_t counterThread[ALG_COUNTER_SIZE];
        RandomKernelBase::CopyArray<ALG_COUNTER_SIZE>(counterThread, counter);
        RandomKernelBase::SkipLo(counterThread, static_cast<uint64_t>(groupIndex) * RESERVED_SAMPLES_PER_OUTPUT);

        float results[ALG_COUNTER_SIZE];
        GenSamples(results, key, counterThread);

        for (int32_t i = 0; i < static_cast<int32_t>(GROUP_SIZE); ++i) {
            if (offset >= outputNum) {
                return;
            }
            if constexpr (IsSameType<Y_T, bfloat16_t>::value || IsSameType<Y_T, half>::value) {
                yGm[offset] = static_cast<Y_T>(results[i]);
            } else {
                yGm[offset] = results[i];
            }
            ++offset;
        }
        offset += (totalThreadCount - 1) * static_cast<int64_t>(GROUP_SIZE);
        groupIndex += totalThreadCount;
    }
}

template <typename Y_T, typename OFFSET_T>
class TruncatedNormalV2Simt {
public:
    __aicore__ inline TruncatedNormalV2Simt(const RandomUnifiedSimtTilingDataStruct* tiling) : tilingData_(tiling){};
    __aicore__ inline void Process(GM_ADDR y, GM_ADDR offset);

private:
    const RandomUnifiedSimtTilingDataStruct* tilingData_;
};

template <typename Y_T, typename OFFSET_T>
__aicore__ inline void TruncatedNormalV2Simt<Y_T, OFFSET_T>::Process(GM_ADDR y, GM_ADDR offset)
{
    GlobalTensor<OFFSET_T> offsetGm;
    offsetGm.SetGlobalBuffer((__gm__ OFFSET_T*)offset);
    int64_t stateOffset = offsetGm(0);
    int64_t outputNum = tilingData_->outputSize;
    const uint32_t key0 = static_cast<uint32_t>(tilingData_->seed);
    const uint32_t key1 = static_cast<uint32_t>(tilingData_->seed >> SHIFT_BITS);
    const uint32_t counter2 = static_cast<uint32_t>(tilingData_->offset);
    const uint32_t counter3 = static_cast<uint32_t>(tilingData_->offset >> SHIFT_BITS);

    SyncAll();
    asc_vf_call<SimtCompute<Y_T, OFFSET_T>>(
        dim3{USED_THREAD}, (__gm__ Y_T*)y, stateOffset, outputNum, key0, key1, counter2,
        counter3);
    if (GetBlockIdx() == 0) {
        // update output offset
        offsetGm(0) = stateOffset + 
            static_cast<OFFSET_T>(outputNum * static_cast<int64_t>(RESERVED_SAMPLES_PER_OUTPUT));
        AscendC::DataCacheCleanAndInvalid<
            OFFSET_T, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(offsetGm[0]);
    }
}
} // namespace TruncatedNormalV2
#endif // RANDOM_TRUNCATED_NORMAL_V2_SIMT_H_
