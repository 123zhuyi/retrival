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
#include "utils/debug/asc_assert.h"

namespace StatelessSampleMultinomial {
using namespace AscendC;
using namespace RandomKernelBase;

constexpr uint16_t CORE_THREAD_NUM_U32 = 256;
constexpr uint16_t CORE_THREAD_NUM_U64 = 256;
constexpr static int64_t SAMPLES_ALIGNMENT = 128;
constexpr static uint32_t UNROLL_FACTOR = 4;

template <typename XT, typename NormProbsT, bool HAS_NORM_PROBS, typename IndexT, uint16_t THREAD_LAUNCH_BOUND>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH_BOUND) inline void SimtUniformRandomBinarySearch(
    __gm__ volatile int64_t* outputGM, __gm__ XT* xGM, __gm__ NormProbsT* normProbsGM, IndexT elementNum, int64_t seed,
    IndexT numsamples, uint32_t numCat, uint64_t nsMagic, uint64_t nsShift, IndexT samplesAligned,
    uint32_t baseOffsetLo, uint32_t baseOffsetHi)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    key[0] = static_cast<uint32_t>(seed);
    key[1] = static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32);

    IndexT idx = blockIdx.x * blockDim.x + threadIdx.x;
    IndexT stride = blockDim.x * gridDim.x;

    IndexT elementNumAligned = elementNum & ~static_cast<IndexT>(UNROLL_FACTOR - 1);

    for (IndexT baseIndex = idx * UNROLL_FACTOR; baseIndex < elementNum; baseIndex += stride * UNROLL_FACTOR) {
        uint32_t count = (baseIndex < elementNumAligned) ? UNROLL_FACTOR :
                                                           static_cast<uint32_t>(elementNum - baseIndex);

        IndexT d = static_cast<IndexT>(Simt::UintDiv(static_cast<uint64_t>(baseIndex), nsMagic, nsShift));
        IndexT s = baseIndex - d * numsamples;
        __gm__ XT* x = xGM + d * static_cast<IndexT>(numCat);
        __gm__ NormProbsT* normProbs = normProbsGM;
        if constexpr (HAS_NORM_PROBS) {
            normProbs += d * static_cast<IndexT>(numCat);
        }
        uint64_t subsequence = static_cast<uint64_t>(d) * samplesAligned + s;

        for (uint32_t k = 0; k < count; k++) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {baseOffsetLo, baseOffsetHi, static_cast<uint32_t>(subsequence),
                                                     static_cast<uint32_t>(subsequence >> 32)};
            PhiloxRandomSimt(key, counterTmp, counterTmp);

            XT u = static_cast<XT>(counterTmp[0] * RAND_2POW32_INV + RAND_2POW32_INV_HALF);

            IndexT start = 0;
            IndexT end = numCat;
            assert(x[numCat - 1] > static_cast<XT>(0));

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
            if constexpr (HAS_NORM_PROBS) {
                while (start >= 1 && normProbs[start] == static_cast<NormProbsT>(0)) {
                    start--;
                }
            } else {
                while (start >= 1 && x[start] == x[start - 1]) {
                    start--;
                }
            }

            outputGM[baseIndex + k] = static_cast<int64_t>(start);

            ++subsequence;
            if (++s >= numsamples) {
                s = 0;
                x += numCat;
                if constexpr (HAS_NORM_PROBS) {
                    normProbs += numCat;
                }
                subsequence += samplesAligned - numsamples;
            }
        }
    }
}

template <typename XT, typename NormProbsT>
class StatelessSampleMultinomialOp {
public:
    __aicore__ inline StatelessSampleMultinomialOp(){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR x, GM_ADDR normProbs, GM_ADDR seed, GM_ADDR offset,
                                GM_ADDR workspace, const RandomUnifiedSimtTilingDataStruct* __restrict tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    const RandomUnifiedSimtTilingDataStruct* tilingData_;
    GlobalTensor<int64_t> outputGM_;
    GM_ADDR xGM_;
    GM_ADDR normProbsGM_;
    GM_ADDR seedGM_;
    GM_ADDR offsetGM_;
    uint32_t blockIdx_;
};

template <typename XT, typename NormProbsT>
__aicore__ inline void StatelessSampleMultinomialOp<XT, NormProbsT>::Init(
    GM_ADDR y, GM_ADDR x, GM_ADDR normProbs, GM_ADDR seed, GM_ADDR offset, GM_ADDR workspace,
    const RandomUnifiedSimtTilingDataStruct* __restrict tilingData, TPipe* pipe)
{
    tilingData_ = tilingData;
    outputGM_.SetGlobalBuffer((__gm__ int64_t*)y);
    xGM_ = x;
    normProbsGM_ = normProbs;
    seedGM_ = seed;
    offsetGM_ = offset;
    blockIdx_ = GetBlockIdx();
}

template <typename XT, typename NormProbsT>
__aicore__ inline void StatelessSampleMultinomialOp<XT, NormProbsT>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    uint64_t numsamples = static_cast<uint64_t>(tilingData_->from);
    uint32_t numCat = static_cast<uint32_t>(tilingData_->range);
    uint64_t elementNum = static_cast<uint64_t>(tilingData_->extraInt64Param1);
    bool useUint64Index = (tilingData_->splitBlockCount != 0);

    // Read real seed/offset from GM (tiling filled 0 placeholders because the offset is a
    // device-computed intermediate with no host value at tiling time; see the tiling impl).
    int64_t realSeed = *(reinterpret_cast<__gm__ int64_t*>(seedGM_));
    int64_t realOffset = *(reinterpret_cast<__gm__ int64_t*>(offsetGM_));

    uint64_t nsMagic, nsShift;
    GetUintDivMagicAndShift(nsMagic, nsShift, numsamples);

    uint64_t baseOffset = (static_cast<uint64_t>(realOffset) + VEC_4 - 1) / VEC_4;
    uint32_t baseOffsetLo = static_cast<uint32_t>(baseOffset);
    uint32_t baseOffsetHi = static_cast<uint32_t>(baseOffset >> 32);

    uint64_t samplesAligned = ((numsamples + SAMPLES_ALIGNMENT - 1) / SAMPLES_ALIGNMENT) * SAMPLES_ALIGNMENT;
    if (useUint64Index) {
        if (normProbsGM_ != nullptr) {
            asc_vf_call<SimtUniformRandomBinarySearch<XT, NormProbsT, true, uint64_t, CORE_THREAD_NUM_U64>>(
                dim3(CORE_THREAD_NUM_U64), (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()), (__gm__ XT*)(xGM_),
                (__gm__ NormProbsT*)(normProbsGM_), elementNum, realSeed, numsamples, numCat, nsMagic, nsShift,
                samplesAligned, baseOffsetLo, baseOffsetHi);
        } else {
            asc_vf_call<SimtUniformRandomBinarySearch<XT, NormProbsT, false, uint64_t, CORE_THREAD_NUM_U64>>(
                dim3(CORE_THREAD_NUM_U64), (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()), (__gm__ XT*)(xGM_),
                (__gm__ NormProbsT*)(normProbsGM_), elementNum, realSeed, numsamples, numCat, nsMagic, nsShift,
                samplesAligned, baseOffsetLo, baseOffsetHi);
        }
    } else {
        if (normProbsGM_ != nullptr) {
            asc_vf_call<SimtUniformRandomBinarySearch<XT, NormProbsT, true, uint32_t, CORE_THREAD_NUM_U32>>(
                dim3(CORE_THREAD_NUM_U32), (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()), (__gm__ XT*)(xGM_),
                (__gm__ NormProbsT*)(normProbsGM_), static_cast<uint32_t>(elementNum), realSeed,
                static_cast<uint32_t>(numsamples), numCat, nsMagic, nsShift, static_cast<uint32_t>(samplesAligned),
                baseOffsetLo, baseOffsetHi);
        } else {
            asc_vf_call<SimtUniformRandomBinarySearch<XT, NormProbsT, false, uint32_t, CORE_THREAD_NUM_U32>>(
                dim3(CORE_THREAD_NUM_U32), (__gm__ volatile int64_t*)(outputGM_.GetPhyAddr()), (__gm__ XT*)(xGM_),
                (__gm__ NormProbsT*)(normProbsGM_), static_cast<uint32_t>(elementNum), realSeed,
                static_cast<uint32_t>(numsamples), numCat, nsMagic, nsShift, static_cast<uint32_t>(samplesAligned),
                baseOffsetLo, baseOffsetHi);
        }
    }
}
} // namespace StatelessSampleMultinomial
#endif // STATELESS_SAMPLE_MULTINOMIAL_IMPL_H
