/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATELESS_RANDOM_IMPL_H
#define STATELESS_RANDOM_IMPL_H

#include "../../random_common/arch35/random_kernel_base.h"
#include "simt_api/asc_simt.h"

namespace StatelessRandom {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr static uint32_t NUM_2 = 2;
constexpr static uint32_t NUM_4 = 4;
constexpr static int64_t INT64_THRESHOLD = 268435456LL;
constexpr uint16_t CORE_THREAD_NUM = 512;

constexpr int64_t RANGE_MODE_INT64 = 0;
constexpr int64_t RANGE_MODE_INT32 = 1;

template <typename T, int32_t UNROLL, int64_t RangeMode>
struct StatelessRandomTransform {
    int64_t from_;
    uint64_t range_;
    
    __aicore__ StatelessRandomTransform(int64_t from, uint64_t range) 
        : from_(from), range_(range) {}
    
    __simt_callee__ __aicore__ inline void operator()(
        __gm__ volatile T* outputGm, uint64_t li, const uint32_t* results, uint32_t iStep,
        [[maybe_unused]] uint32_t randPerOutput = 1)
    {
        if constexpr (RangeMode == RANGE_MODE_INT64) {
            uint32_t high = results[iStep * NUM_2];
            uint32_t low = results[iStep * NUM_2 + 1];
            uint64_t randU64 = (static_cast<uint64_t>(high) << 32) | low;
            int64_t mappedValue = static_cast<int64_t>(randU64 % static_cast<uint64_t>(range_)) + from_;
            outputGm[li] = static_cast<T>(mappedValue);
        } else {
            uint32_t randU32 = results[iStep];
            int64_t mappedValue = static_cast<int64_t>(randU32 % static_cast<uint32_t>(range_)) + from_;
            outputGm[li] = static_cast<T>(mappedValue);
        }
    }
};

template <typename T, int32_t UNROLL, int64_t RangeMode>
struct StatelessRandomLauncher {
    __gm__ volatile T* outputGm_;
    int64_t seed_;
    int64_t realOffset_;
    int64_t from_;
    int64_t range_;
    
    __aicore__ StatelessRandomLauncher(__gm__ volatile T* outputGm, int64_t seed, int64_t realOffset,
                                       int64_t from, uint64_t range)
        : outputGm_(outputGm), seed_(seed), realOffset_(realOffset), from_(from), range_(range) {}
    
    __aicore__ inline void operator()(
        const ExecutionPolicyKernel& policy, int64_t gmOffset, int64_t kernelOffset,
        int64_t numel, int64_t grid, int64_t totalThreads)
    {
        __gm__ volatile T* gmPtr = outputGm_ + gmOffset;
        StatelessRandomTransform<T, UNROLL, RangeMode> transform(from_, range_);
        AscendC::Simt::VF_CALL<PhiloxSimtKernelDiscontinuous<T, StatelessRandomTransform<T, UNROLL, RangeMode>, CORE_THREAD_NUM, UNROLL>>(
            AscendC::Simt::Dim3(CORE_THREAD_NUM),
            gmPtr, realOffset_ + kernelOffset, seed_, static_cast<uint64_t>(numel),
            policy.magic, policy.shift, static_cast<uint64_t>(totalThreads), transform);
    }
};

template <typename T>
class StatelessRandomImpl {
public:
    __aicore__ inline StatelessRandomImpl(TPipe* pipe, const RandomUnifiedSimtTilingDataStruct* tilingData)
        : pipe_(pipe), tiling_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR y);
    __aicore__ inline void Process(GM_ADDR seed, GM_ADDR offset);

private:
    TPipe* pipe_;
    const RandomUnifiedSimtTilingDataStruct* tiling_;

    GlobalTensor<T> outputGm_;
    int64_t from_ = 0;
    uint64_t range_ = 0;
    int64_t useInt64Mode_ = 0;
};

template <typename T>
__aicore__ inline void StatelessRandomImpl<T>::Init(GM_ADDR y)
{
    outputGm_.SetGlobalBuffer((__gm__ T*)y);

    from_ = tiling_->from;
    range_ = tiling_->range;
    useInt64Mode_ = tiling_->extraInt64Param1;
}

template <typename T>
__aicore__ inline void StatelessRandomImpl<T>::Process(GM_ADDR seed, GM_ADDR offset)
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= tiling_->usedCoreNum) {
        return;
    }

    int64_t realSeed = *(reinterpret_cast<__gm__ int64_t*>(seed));
    int64_t realOffset = *(reinterpret_cast<__gm__ int64_t*>(offset));
    if (useInt64Mode_ == NUM_2) {
        StatelessRandomLauncher<T, NUM_2, RANGE_MODE_INT64> launcher(
            (__gm__ volatile T*)outputGm_.GetPhyAddr(), realSeed, realOffset, from_, range_);
        ProcessWithSplitBlocks(tiling_, launcher);
    } else {
        StatelessRandomLauncher<T, NUM_4, RANGE_MODE_INT32> launcher(
            (__gm__ volatile T*)outputGm_.GetPhyAddr(), realSeed, realOffset, from_, range_);
        ProcessWithSplitBlocks(tiling_, launcher);
    }
}

} // namespace StatelessRandom
#endif // STATELESS_RANDOM_IMPL_H