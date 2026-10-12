/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_exponential.cpp
 * \brief Kernel entry point for StatelessExponential.
 *
 * The per-element computation (Philox4x32-10 -> uniform -> -log/lambda -> cast) is identical
 * to SimThreadExponential, so this file reuses SimThreadExponential::ExponentialTransform<T>.
 *
 * Difference from SimThreadExponential: seed/offset are tensor INPUTS (not attrs). Callers
 * such as aclnnMultinomialTensor pass an offset that is a device-computed intermediate
 * (l0op::Add output) with no host value at tiling time, so the tiling layer cannot read
 * their values (host-gathering an unmapped device page faults). Following the stateless_normal
 * reference, the kernel reads the real seed/offset directly from GM at runtime and applies
 * them here. The tiling split-block kernelOffset carries only the per-block counter increments
 * (base offset 0), so the real offset from GM is added on top of kernelOffset (identical scheme
 * to stateless_normal). Values are the same as the original tiling-fed design intended; only
 * their source moves from a crashing host-gather to a valid device read.
 */
#include "../sim_thread_exponential/arch35/sim_thread_exponential_simt.h"

namespace StatelessExponential {
using namespace AscendC;
using namespace RandomKernelBase;

template <typename T>
struct StatelessExponentialLauncher {
    int64_t seed_;
    int64_t realOffset_;
    float lambda_;
    GM_ADDR baseAddr_;

    __aicore__ StatelessExponentialLauncher(int64_t seed, int64_t realOffset, float lambda, GM_ADDR baseAddr)
        : seed_(seed), realOffset_(realOffset), lambda_(lambda), baseAddr_(baseAddr)
    {}

    __aicore__ inline void operator()(const ExecutionPolicyKernel& policy, int64_t gmOffset, int64_t kernelOffset,
                                      int64_t numel, [[maybe_unused]] int64_t grid, int64_t totalThreads)
    {
        __gm__ volatile T* gmPtr = (__gm__ volatile T*)baseAddr_ + gmOffset;
        SimThreadExponential::ExponentialTransform<T> transform(lambda_);
        AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, SimThreadExponential::ExponentialTransform<T>>>(
            AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM), gmPtr, realOffset_ + kernelOffset, seed_, numel, policy.magic,
            policy.shift, totalThreads, transform);
    }
};

// self is both input and output (in-place). seed/offset are scalar INT64 GM tensors.
template <typename T>
__aicore__ inline void Process(GM_ADDR self, GM_ADDR seed, GM_ADDR offset,
                               const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum))
        return;

    // Read real seed/offset from GM (tiling filled 0 placeholders; the Add producing offset
    // has already executed by kernel launch time).
    int64_t realSeed = *(reinterpret_cast<__gm__ int64_t*>(seed));
    int64_t realOffset = *(reinterpret_cast<__gm__ int64_t*>(offset));

    StatelessExponentialLauncher<T> launcher(realSeed, realOffset, tilingData->prob, self);
    ProcessWithSplitBlocks(tilingData, launcher);
}
} // namespace StatelessExponential

extern "C" __global__ __aicore__ void stateless_exponential(GM_ADDR self, GM_ADDR seed, GM_ADDR offset, GM_ADDR selfOut,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(RandomUnifiedSimtTilingDataStruct, tilingData, tiling);

    if (TILING_KEY_IS(3)) {
        StatelessExponential::Process<float>(self, seed, offset, &tilingData);
    } else if (TILING_KEY_IS(1)) {
        StatelessExponential::Process<half>(self, seed, offset, &tilingData);
    } else if (TILING_KEY_IS(2)) {
        StatelessExponential::Process<bfloat16_t>(self, seed, offset, &tilingData);
    }
}
