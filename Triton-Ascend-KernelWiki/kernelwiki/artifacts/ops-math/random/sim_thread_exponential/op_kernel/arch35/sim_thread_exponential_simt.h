/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_THREAD_EXPONENTIAL_SIMT_H
#define SIM_THREAD_EXPONENTIAL_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../random_common/arch35/random_unified_tiling_data_arch35.h"
#include "sim_thread_exponential_tiling_key.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace SimThreadExponential {
using namespace AscendC;
using namespace RandomKernelBase;

template <typename T>
struct ExponentialTransform {
    float lambda_;
    float halfEpsilon_;

    __aicore__ ExponentialTransform(float lambda) : lambda_(lambda), halfEpsilon_(1.1920929e-07f / 2.0f)
    {}

    __simt_callee__ __aicore__ inline void operator()(
        __gm__ volatile T* outputGm, uint64_t li, const uint32_t* results, uint32_t iStep,
        [[maybe_unused]] uint32_t unroll = 1)
    {
        float u = results[iStep] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
        float logVal = (u >= 1.0f - halfEpsilon_) ? -halfEpsilon_ : AscendC::Simt::Log(u);
        float x = -1.0f / lambda_ * logVal;
        outputGm[li] = static_cast<T>(x);
    }
};

template <typename T>
struct ExponentialLauncher {
    int64_t seed_;
    float lambda_;
    GM_ADDR baseAddr_;

    __aicore__ ExponentialLauncher(int64_t seed, float lambda, GM_ADDR baseAddr)
        : seed_(seed), lambda_(lambda), baseAddr_(baseAddr)
    {}

    __aicore__ inline void operator()(
        const ExecutionPolicyKernel& policy, int64_t gmOffset, int64_t kernelOffset, int64_t numel, int64_t grid,
        int64_t totalThreads)
    {
        __gm__ volatile T* gmPtr = (__gm__ volatile T*)baseAddr_ + gmOffset;
        ExponentialTransform<T> transform(lambda_);
        AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, ExponentialTransform<T>>>(
            AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM), gmPtr, kernelOffset, seed_, numel, policy.magic, policy.shift,
            totalThreads, transform);
    }
};

template <typename T>
__aicore__ inline void Process(GM_ADDR self, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum))
        return;

    ExponentialLauncher<T> launcher(tilingData->seed, tilingData->prob, self);
    ProcessWithSplitBlocks(tilingData, launcher);
}

} // namespace SimThreadExponential
#endif