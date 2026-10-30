/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATELESS_UNIFORM_SIMT_H
#define STATELESS_UNIFORM_SIMT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "../../random_common/arch35/random_kernel_base.h"

namespace StatelessUniformSimt {
using namespace AscendC;
using namespace RandomKernelBase;

// UniformTransform: maps rand in (0,1] to [from, to) with boundary flip
// Matches CUDA: value = rand * (to - from) + from; if (value == to) value = from;
template <typename T>
struct UniformTransform {
    float from_;
    float to_;

    __aicore__ UniformTransform(float from, float to)
        : from_(from), to_(to) {}

    __simt_callee__ __aicore__ inline void operator()(
        __gm__ volatile T* outputGm, uint64_t li, const uint32_t* results, uint32_t iStep,
        [[maybe_unused]] uint32_t unroll = 1)
    {
        // 公共模板传入原始 uint32 Philox 输出，需自行归一化
        // 对标 CUDA _curand_uniform4: x * 2^-32 + 2^-33 → (0, 1]
        float rand = results[iStep] * RAND_2POW32_INV + RAND_2POW32_INV_HALF;
        if constexpr (IsSameType<T, float>::value) {
            float range = to_ - from_;
            float value = rand * range + from_;
            if (value == to_) {
                value = from_;
            }
            outputGm[li] = value;
        } else if constexpr (IsSameType<T, half>::value) {
            half fromH = static_cast<half>(from_);
            half toH = static_cast<half>(to_);
            half rangeH = toH - fromH;
            float rangeF = static_cast<float>(rangeH);
            float fromHF = static_cast<float>(fromH);
            float valueF = rand * rangeF + fromHF;
            half valueH = static_cast<half>(valueF);
            if (valueH == toH) {
                valueH = fromH;
            }
            outputGm[li] = valueH;
        } else { // bfloat16_t
            bfloat16_t fromB = static_cast<bfloat16_t>(from_);
            bfloat16_t toB = static_cast<bfloat16_t>(to_);
            bfloat16_t rangeB = toB - fromB;
            float rangeF = static_cast<float>(rangeB);
            float fromBF = static_cast<float>(fromB);
            float valueF = rand * rangeF + fromBF;
            bfloat16_t valueB = static_cast<bfloat16_t>(valueF);
            if (valueB == toB) {
                valueB = fromB;
            }
            outputGm[li] = valueB;
        }
    }
};

template <typename T>
struct UniformLauncher {
    int64_t seed_;
    int64_t realOffset_;
    float from_;
    float to_;
    GM_ADDR baseAddr_;

    __aicore__ UniformLauncher(int64_t seed, int64_t realOffset, float from, float to, GM_ADDR baseAddr)
        : seed_(seed), realOffset_(realOffset), from_(from), to_(to), baseAddr_(baseAddr) {}

    __aicore__ inline void operator()(
        const ExecutionPolicyKernel& policy,
        int64_t gmOffset,
        int64_t kernelOffset,
        int64_t numel,
        int64_t grid,
        int64_t totalThreads)
    {
        __gm__ volatile T* gmPtr = (__gm__ volatile T*)baseAddr_ + gmOffset;
        UniformTransform<T> transform(from_, to_);
        AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, UniformTransform<T>>>(
            AscendC::Simt::Dim3(DEFAULT_SIMT_THREAD_NUM),
            gmPtr, realOffset_ + kernelOffset, seed_, numel,
            policy.magic, policy.shift, totalThreads, transform);
    }
};

template <typename T>
__aicore__ inline void Process(GM_ADDR seed, GM_ADDR offset, GM_ADDR y, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) return;

    // 读出真实seedoffset
    int64_t realSeed = *(reinterpret_cast<__gm__ int64_t*>(seed));
    int64_t realOffset = *(reinterpret_cast<__gm__ int64_t*>(offset));

    UniformLauncher<T> launcher(realSeed, realOffset, tilingData->fromFp32, tilingData->toFp32, y);
    ProcessWithSplitBlocks(tilingData, launcher);
}

} // namespace StatelessUniformSimt
#endif
