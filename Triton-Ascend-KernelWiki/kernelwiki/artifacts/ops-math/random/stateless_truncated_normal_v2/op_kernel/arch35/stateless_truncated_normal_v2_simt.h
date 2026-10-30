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
 * \file stateless_truncated_normal_v2_simt.h
 * \brief StatelessTruncatedNormalV2 SIMT kernel implementation
 *        Uses common PhiloxRandomSimt + BoxMullerFloat from random_kernel_base.h
 *        Cannot use PhiloxSimtKernelDiscontinuous because truncated normal requires
 *        rejection sampling (variable Philox calls per output element).
 *        Uses ProcessWithSplitBlocks for large tensor split handling.
 *
 *        Algorithm (matching TF StatelessTruncatedNormalV2):
 *        1. Philox4x32(key, counter) -> 4 x uint32
 *        2. uint32 -> float in (0,1) via IEEE754 mantissa trick
 *        3. Box-Muller: (u1, u2) -> (z0, z1) standard normal
 *        4. Rejection: keep |z| < 2.0, else re-sample
 *        Determinism: each thread gets counter offset = groupIndex * RESERVED_SAMPLES_PER_OUTPUT
 *
 *        TF V2 interface Philox state mapping:
 *        key_data[0]      -> key_[0], key_[1]
 *        counter_data[0]  -> counter_[0], counter_[1]  (low 64 bits)
 *        counter_data[1]  -> counter_[2], counter_[3]  (high 64 bits)
 */

#ifndef RANDOM_STATELESS_TRUNCATED_NORMAL_V2_SIMT_H_
#define RANDOM_STATELESS_TRUNCATED_NORMAL_V2_SIMT_H_

#include "kernel_operator.h"
#include "../../random_common/arch35/random_kernel_base.h"
#include "../../random_common/arch35/random_unified_tiling_data_arch35.h"

namespace StatelessTruncatedNormalV2 {

using namespace AscendC;
using namespace RandomKernelBase;

// ============================================================================
// Constants
// ============================================================================
constexpr uint64_t RESERVED_SAMPLES_PER_OUTPUT = 256;
constexpr uint64_t GROUP_SIZE = 4;
constexpr float TRUNCATE_VALUE = 2.0f;
constexpr uint32_t SHIFT_BITS = 32;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD = 128;
#else
constexpr uint32_t USED_THREAD = DEFAULT_SIMT_THREAD_NUM;
#endif

// ============================================================================
// Scalar Uint32ToFloat (IEEE754 mantissa trick, same as TF Uint32ToFloat)
// Result in [0, 1), uses low 23 bits as mantissa
// ============================================================================
__simt_callee__ __aicore__ inline float Uint32ToFloat(const uint32_t x)
{
    constexpr uint32_t MANTISSA_BIT = 23;
    const uint32_t man = x & 0x7fffffu;
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << MANTISSA_BIT) | man;
    float result = *reinterpret_cast<const float*>(&val);
    return result - 1.0f;
}

// ============================================================================
// TruncatedNormal VF Kernel (rejection sampling)
// ============================================================================

/// @brief SIMT VF kernel for truncated normal generation with rejection sampling.
///        Each thread generates GROUP_SIZE=4 valid samples per iteration.
///        Determinism guaranteed by per-thread counter offset (groupIndex * 256).
///        key/counter are passed as raw uint32 arrays (TF V2 Philox state layout).
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(DEFAULT_SIMT_THREAD_NUM) inline void TruncatedNormalSimtKernel(
    __gm__ T* outputGm, uint64_t outputSize,
    uint32_t key0, uint32_t key1,
    uint32_t counter0, uint32_t counter1, uint32_t counter2, uint32_t counter3,
    int64_t kernelOffset)
{
    uint32_t key[ALG_KEY_SIZE] = {key0, key1};
    uint32_t counter[ALG_COUNTER_SIZE] = {counter0, counter1, counter2, counter3};

    // Apply kernelOffset to counter low bits (for split blocks inter-block spacing)
    if (kernelOffset > 0) {
        SkipLo(counter, static_cast<uint64_t>(kernelOffset));
    }

    // Use ACTUAL runtime thread count as stride (gridDim.x * blockDim.x), NOT the
    // tiling-computed virtual totalThreads. The tiling value (grid*256, capped at 624*256)
    // does not match the real launched thread count (usedCoreNum * USED_THREAD), and using
    // it as stride leaves large unwritten gaps in the output. This matches TruncatedNormalV2.
    uint64_t totalThreadCount = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    uint64_t groupIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t outOffset = groupIndex * GROUP_SIZE;

    while (outOffset < outputSize) {
        // Each thread gets a unique counter region to ensure determinism
        uint32_t counterThread[ALG_COUNTER_SIZE];
        CopyArray<ALG_COUNTER_SIZE>(counterThread, counter);
        SkipLo(counterThread, groupIndex * RESERVED_SAMPLES_PER_OUTPUT);

        // Generate GROUP_SIZE valid truncated normal samples via rejection sampling
        float results[GROUP_SIZE];
        uint32_t validCount = 0;

        while (validCount < GROUP_SIZE) {
            uint32_t philoxOut[SIMT_STEP];
            PhiloxRandomSimt(key, counterThread, philoxOut);
            SkipOne(counterThread);

            // Box-Muller pair 1: philoxOut[0], philoxOut[1] -> z0, z1
            // Use Uint32ToFloat (IEEE754 mantissa trick) + BoxMullerFloatSafe (eps-protected)
            // to match TF BoxMullerFloat implementation exactly
            float z0, z1;
            BoxMullerFloatSafe(Uint32ToFloat(philoxOut[0]), Uint32ToFloat(philoxOut[1]), &z0, &z1);

            if (fabsf(z0) < TRUNCATE_VALUE) {
                results[validCount++] = z0;
                if (validCount >= GROUP_SIZE) break;
            }
            if (fabsf(z1) < TRUNCATE_VALUE) {
                results[validCount++] = z1;
                if (validCount >= GROUP_SIZE) break;
            }

            // Box-Muller pair 2: philoxOut[2], philoxOut[3] -> z2, z3
            BoxMullerFloatSafe(Uint32ToFloat(philoxOut[2]), Uint32ToFloat(philoxOut[3]), &z0, &z1);

            if (fabsf(z0) < TRUNCATE_VALUE) {
                results[validCount++] = z0;
                if (validCount >= GROUP_SIZE) break;
            }
            if (fabsf(z1) < TRUNCATE_VALUE) {
                results[validCount++] = z1;
                if (validCount >= GROUP_SIZE) break;
            }
        }

        // Write valid samples to GM
#pragma unroll
        for (uint32_t i = 0; i < GROUP_SIZE; ++i) {
            if (outOffset >= outputSize) return;
            if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
                outputGm[outOffset] = static_cast<T>(results[i]);
            } else {
                outputGm[outOffset] = results[i];
            }
            ++outOffset;
        }

        // Advance to next group (strided across all ACTUAL runtime threads)
        outOffset += (totalThreadCount - 1) * GROUP_SIZE;
        groupIndex += totalThreadCount;
    }
}

// ============================================================================
// Launcher: wraps VF_CALL for ProcessWithSplitBlocks integration
// ============================================================================

template <typename T>
struct TruncatedNormalLauncher {
    uint32_t key0_, key1_;
    uint32_t counter0_, counter1_, counter2_, counter3_;
    GM_ADDR yAddr_;

    __aicore__ TruncatedNormalLauncher(
        uint32_t key0, uint32_t key1,
        uint32_t counter0, uint32_t counter1, uint32_t counter2, uint32_t counter3,
        GM_ADDR y)
        : key0_(key0), key1_(key1),
          counter0_(counter0), counter1_(counter1), counter2_(counter2), counter3_(counter3),
          yAddr_(y) {}

    __aicore__ inline void operator()(
        const ExecutionPolicyKernel& policy,
        int64_t gmOffset,
        int64_t kernelOffset,
        int64_t numel,
        [[maybe_unused]] int64_t grid,
        [[maybe_unused]] int64_t totalThreads)
    {
        __gm__ T* gmPtr = reinterpret_cast<__gm__ T*>(yAddr_) + gmOffset;
        // kernelOffset is relative increment from 0 (tiling.offset=0)
        // VF Kernel applies it to counter low bits via SkipLo.
        // NOTE: stride inside the kernel uses the ACTUAL runtime thread count
        // (gridDim.x * blockDim.x), NOT the tiling totalThreads, to avoid output gaps.
        Simt::VF_CALL<TruncatedNormalSimtKernel<T>>(
            Simt::Dim3(USED_THREAD),
            gmPtr, static_cast<uint64_t>(numel),
            key0_, key1_, counter0_, counter1_, counter2_, counter3_,
            kernelOffset);
    }
};

// ============================================================================
// Entry point
// ============================================================================

/// @brief Main entry point called from stateless_truncated_normal_v2.cpp
///        Reads key/counter from GM inputs (TF V2 interface), manually initializes
///        full 128-bit Philox state, launches SIMT kernel via ProcessWithSplitBlocks.
template <typename T>
__aicore__ inline void Process(
    GM_ADDR key, GM_ADDR counter,
    GM_ADDR y,
    const RandomUnifiedSimtTilingDataStruct* __restrict tilingData)
{
    if (GetBlockIdx() >= static_cast<uint32_t>(tilingData->usedCoreNum)) {
        return;
    }

    // Read key (uint64[1]) from GM → split into Philox key[0,1]
    uint64_t keyVal = *(reinterpret_cast<__gm__ uint64_t*>(key));
    uint32_t key0 = static_cast<uint32_t>(keyVal);
    uint32_t key1 = static_cast<uint32_t>(keyVal >> SHIFT_BITS);

    // Read counter (uint64[2]) from GM → split into Philox counter[0,1,2,3]
    // TF layout: counter_data[0] → counter_[0,1] (low 64-bit)
    //            counter_data[1] → counter_[2,3] (high 64-bit)
    uint64_t counterLo = *(reinterpret_cast<__gm__ uint64_t*>(counter));
    uint64_t counterHi = *(reinterpret_cast<__gm__ uint64_t*>(
        reinterpret_cast<__gm__ uint8_t*>(counter) + sizeof(uint64_t)));
    uint32_t c0 = static_cast<uint32_t>(counterLo);
    uint32_t c1 = static_cast<uint32_t>(counterLo >> SHIFT_BITS);
    uint32_t c2 = static_cast<uint32_t>(counterHi);
    uint32_t c3 = static_cast<uint32_t>(counterHi >> SHIFT_BITS);

    TruncatedNormalLauncher<T> launcher(key0, key1, c0, c1, c2, c3, y);
    ProcessWithSplitBlocks(tilingData, launcher);
}

} // namespace StatelessTruncatedNormalV2
#endif // RANDOM_STATELESS_TRUNCATED_NORMAL_V2_SIMT_H_
