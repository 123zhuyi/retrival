/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATELESS_SAMPLE_MULTINOMIAL_IMPL_H
#define STATELESS_SAMPLE_MULTINOMIAL_IMPL_H

#include "../../random_common/arch35/random_kernel_base.h"
#include "simt_api/asc_simt.h"

namespace StatelessSampleMultinomial {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr static uint32_t NUM_4 = 4;
constexpr uint16_t CORE_THREAD_NUM_U32 = 256;
constexpr uint16_t CORE_THREAD_NUM_U64 = 256;
constexpr static int64_t SAMPLES_ALIGNMENT = 128;
constexpr static uint32_t UNROLL_FACTOR = 4;

template <typename XT, typename IndexT, uint16_t THREAD_LAUNCH_BOUND>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH_BOUND) inline void SimtUniformRandomBinarySearch(
    __gm__ volatile int64_t* outputGM,
    __gm__ volatile XT* xGM,
    IndexT elementNum, int64_t seed,
    IndexT numsamples, uint32_t numCat,
    uint64_t nsMagic, uint64_t nsShift, IndexT samplesAligned,
    uint32_t baseOffsetLo, uint32_t baseOffsetHi)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);

    IndexT idx = blockIdx.x * blockDim.x + threadIdx.x;
    IndexT stride = blockDim.x * gridDim.x;

    IndexT elementNumAligned = elementNum & ~static_cast<IndexT>(UNROLL_FACTOR - 1);

    for (IndexT baseIndex = idx * UNROLL_FACTOR; baseIndex < elementNum;
         baseIndex += stride * UNROLL_FACTOR) {

        uint32_t count = (baseIndex < elementNumAligned) ?
            UNROLL_FACTOR : static_cast<uint32_t>(elementNum - baseIndex);

        IndexT d = static_cast<IndexT>(Simt::UintDiv(static_cast<uint64_t>(baseIndex), nsMagic, nsShift));
        IndexT s = baseIndex - d * numsamples;
        __gm__ volatile XT* x = xGM + d * static_cast<IndexT>(numCat);
        uint64_t subsequence = static_cast<uint64_t>(d) * samplesAligned + s;

        for (uint32_t k = 0; k < count; k++) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {
                baseOffsetLo, baseOffsetHi,
                static_cast<uint32_t>(subsequence),
                static_cast<uint32_t>(subsequence >> 32)
            };
            PhiloxRandomSimt(key, counterTmp, counterTmp);

            XT u = static_cast<XT>(counterTmp[0] * RAND_2POW32_INV + RAND_2POW32_INV_HALF);

            IndexT start = 0;
            IndexT end = numCat;

            while (end > start) {
                IndexT mid = start + ((end - start) >> 1);
                if (x[mid] < u) {
                    start = mid + 1;
                } else {
                    end = mid;
                }
            }

            if (start >= numCat) {
                start = numCat - 1;
            }
            while (start >= 1 && x[start] == x[start - 1]) {
                start--;
            }

            outputGM[baseIndex + k] = static_cast<int64_t>(start);

            ++subsequence;
            if (++s >= numsamples) {
                s = 0;
                x += numCat;
                subsequence += samplesAligned - numsamples;
            }
        }
    }
}

template <typename XT>
class StatelessSampleMultinomialOp {
public:
    __aicore__ inline StatelessSampleMultinomialOp() {};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR x, GM_ADDR workspace,
                                const RandomUnifiedSimtTilingDataStruct* __restrict tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    const RandomUnifiedSimtTilingDataStruct* tilingData_;
    GlobalTensor<int64_t> outputGM_;
    GM_ADDR xGM_;
    uint32_t blockIdx_;
};

template <typename XT>
__aicore__ inline void StatelessSampleMultinomialOp<XT>::Init(
    GM_ADDR y, GM_ADDR x, GM_ADDR workspace,
    const RandomUnifiedSimtTilingDataStruct* __restrict tilingData,
    TPipe* pipe)
{
    tilingData_ = tilingData;
    outputGM_.SetGlobalBuffer((__gm__ int64_t*)y);
    xGM_ = x;
    blockIdx_ = GetBlockIdx();
}

template <typename XT>
__aicore__ inline void StatelessSampleMultinomialOp<XT>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    uint64_t numsamples = static_cast<uint64_t>(tilingData_->from);
    uint32_t numCat = static_cast<uint32_t>(tilingData_->range);
    uint64_t elementNum = static_cast<uint64_t>(tilingData_->extraInt64Param1);
    bool useUint64Index = (tilingData_->splitBlockCount != 0);

    uint64_t nsMagic, nsShift;
    GetUintDivMagicAndShift(nsMagic, nsShift, numsamples);

    uint64_t baseOffset = (static_cast<uint64_t>(tilingData_->offset) + VEC_4 - 1) / VEC_4;
    uint32_t baseOffsetLo = static_cast<uint32_t>(baseOffset);
    uint32_t baseOffsetHi = static_cast<uint32_t>(baseOffset >> 32);

    uint64_t samplesAligned = ((numsamples + SAMPLES_ALIGNMENT - 1)
                               / SAMPLES_ALIGNMENT) * SAMPLES_ALIGNMENT;

    if (useUint64Index) {
        asc_vf_call<SimtUniformRandomBinarySearch<XT, uint64_t, CORE_THREAD_NUM_U64>>(dim3(CORE_THREAD_NUM_U64),
            (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()),
            (__gm__ volatile XT*)(xGM_),
            elementNum, tilingData_->seed,
            numsamples, numCat, nsMagic, nsShift, samplesAligned,
            baseOffsetLo, baseOffsetHi);
    } else {
        asc_vf_call<SimtUniformRandomBinarySearch<XT, uint32_t, CORE_THREAD_NUM_U32>>(dim3(CORE_THREAD_NUM_U32),
            (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()),
            (__gm__ volatile XT*)(xGM_),
            static_cast<uint32_t>(elementNum), tilingData_->seed,
            static_cast<uint32_t>(numsamples), numCat, nsMagic, nsShift,
            static_cast<uint32_t>(samplesAligned), baseOffsetLo, baseOffsetHi);
    }
}
} // namespace StatelessSampleMultinomial
#endif // STATELESS_SAMPLE_MULTINOMIAL_IMPL_H
