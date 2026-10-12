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
 * \file stateless_bernoulli_simt.h
 * \brief
 */

#ifndef STATELESS_BERNOULLI_H
#define STATELESS_BERNOULLI_H

#include "kernel_operator.h"
#include "../../random_common/arch35/random_kernel_base.h"
#include "simt_api/asc_simt.h"

namespace StatelessBernoulli {

constexpr uint64_t PROB_SCALAR = 1;
constexpr uint64_t PROB_TENSOR = 0;
constexpr uint32_t PHILOX_THREAD_LAUNCH = 512;
constexpr uint32_t MAX_THREADS_PER_PROCESSOR = 2048;
constexpr uint32_t PHILOX_BLOCK_THREAD = 512;
constexpr uint32_t GPU_GRID_SIZE = 2147483647;
constexpr int32_t STEP = 4;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr int32_t CONTINUOUS_USE = 0;

template <typename Tp, typename To, uint64_t PROB_MODE, typename TIdx>
__simt_vf__ __aicore__ LAUNCH_BOUND(PHILOX_THREAD_LAUNCH) inline void PhiloxBernoulliSample(
    __gm__ To* outGm, __gm__ Tp* probGm, int64_t offset, int64_t seed, TIdx outputSize,
    uint64_t magic, uint64_t shift, uint64_t totalThreads)
{
    uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    RandomKernelBase::PhiloxAlgParsInit(key, counterTmp, seed, offset);

    for (TIdx i = (blockIdx.x * blockDim.x + threadIdx.x) * STEP;
         i < outputSize;
         i += gridDim.x * blockDim.x * STEP) {
        uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        RandomKernelBase::CopyArray<ALG_COUNTER_SIZE>(counter, counterTmp);
        float results[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
        RandomKernelBase::ThreadMappingAndSkip<STEP, CONTINUOUS_USE>(i, counter, magic, shift, totalThreads);
        RandomKernelBase::PhiloxRandomSimt(key, counter, results);
        for (uint8_t j = 0; j < 4; ++j) {
            if (i + j >= outputSize) {
                break;
            }
            float probFp32 = 0.0f;
            if constexpr (PROB_MODE == PROB_SCALAR) {
                probFp32 = AscendC::IsSameType<Tp, float>::value ?
                    probGm[0] : static_cast<float>(probGm[0]);
            } else {
                probFp32 = AscendC::IsSameType<Tp, float>::value ?
                    probGm[i + j] : static_cast<float>(probGm[i + j]);
            }
            outGm[i + j] = results[j] <= probFp32 ? 1 : 0;
        }
    }
}

template <typename Tp, typename To>
class StatelessBernoulliKernel {
public:
    __aicore__ inline StatelessBernoulliKernel(){};
    __aicore__ inline void Process(
        GM_ADDR prob, GM_ADDR out, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData);
};

template <typename Tp, typename To>
__aicore__ inline void StatelessBernoulliKernel<Tp, To>::Process(
    GM_ADDR prob, GM_ADDR out, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }

    uint64_t minBlockNums =
        (tilingData->outputSize + MAX_THREADS_PER_PROCESSOR - 1) / MAX_THREADS_PER_PROCESSOR;
    minBlockNums = minBlockNums < GPU_GRID_SIZE ? minBlockNums : GPU_GRID_SIZE;
    uint64_t totalThreads = minBlockNums * PHILOX_BLOCK_THREAD;

    uint64_t magic = 0;
    uint64_t shift = 0;
    RandomKernelBase::GetUintDivMagicAndShift(magic, shift, totalThreads);

    uint64_t probMode = static_cast<uint64_t>(tilingData->extraInt64Param1);
    if (tilingData->outputSize <= GPU_GRID_SIZE) {
        if (PROB_SCALAR == probMode) {
            asc_vf_call<PhiloxBernoulliSample<Tp, To, PROB_SCALAR, uint32_t>>(
                dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob,
                tilingData->offset, tilingData->seed, tilingData->outputSize, magic, shift, totalThreads);
        } else {
            asc_vf_call<PhiloxBernoulliSample<Tp, To, PROB_TENSOR, uint32_t>>(
                dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob,
                tilingData->offset, tilingData->seed, tilingData->outputSize, magic, shift, totalThreads);
        }
    } else {
        if (PROB_SCALAR == probMode) {
            asc_vf_call<PhiloxBernoulliSample<Tp, To, PROB_SCALAR, uint64_t>>(
                dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob,
                tilingData->offset, tilingData->seed, tilingData->outputSize, magic, shift, totalThreads);
        } else {
            asc_vf_call<PhiloxBernoulliSample<Tp, To, PROB_TENSOR, uint64_t>>(
                dim3(PHILOX_THREAD_LAUNCH), (__gm__ To*)out, (__gm__ Tp*)prob,
                tilingData->offset, tilingData->seed, tilingData->outputSize, magic, shift, totalThreads);
        }
    }
}
} // namespace StatelessBernoulli
#endif // STATELESS_BERNOULLI_H
