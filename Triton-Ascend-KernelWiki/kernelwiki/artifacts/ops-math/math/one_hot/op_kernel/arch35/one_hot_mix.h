/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file one_hot_mix.h
 * \brief
 */

#ifndef ONE_HOT_MIX_IMPL_H
#define ONE_HOT_MIX_IMPL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

#ifdef __DAV_FPGA__
#define BOUND_THREAD_NUM (512)
#else
#define BOUND_THREAD_NUM (1024)
#endif

namespace OneHot {
template <typename T1, typename T2, typename T3>
class OneHotMix {
public:
    __aicore__ inline OneHotMix(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR depth, GM_ADDR on_value, GM_ADDR off_value, GM_ADDR y, GM_ADDR workspace, TPipe* pipe,
        const OneHotTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyOut(LocalTensor<T3>& srcTensor, int64_t offset, int32_t copyLength);

private:
    __aicore__ inline void InitOffValueAndCopyOut(void);

private:
    const int64_t MIN_UB_INIT_SIZE = 64;
    const int32_t BIT32 = 32;
    const int32_t BIT64 = 64;
    const uint64_t ONE = 1;
#ifdef __DAV_FPGA__
    const int32_t THREAD_NUM = 128;
#else
    const int32_t THREAD_NUM = 1024;
#endif
    const int32_t MIN_FACTOR = 1024;
    const OneHotTilingData* tiling;
    TPipe* pipe_ = nullptr;
    int32_t blockIdx_;
    int32_t usedCoreNum_;
    int64_t blockFactor_;
    uint64_t factor1;
    uint64_t factor2;
    int64_t depth_;
    int64_t offValueInitLoop_;
    int64_t singleOffValueCalNum_;
    int32_t singleCalNum_;
    int32_t tailOffValueCalNum_;
    int64_t totalInputDim_;
    T3 onValue_;
    T3 offValue_;
    uint64_t factorOut1;
    uint64_t factorOut2;
    GlobalTensor<T1> inputGm;
    GlobalTensor<T2> depthGm;
    GlobalTensor<T3> outputGm;
    GlobalTensor<T3> onValueGm;
    GlobalTensor<T3> offValueGm;
    TBuf<> offValueUb;
    uint32_t shift_;
    uint32_t m_;
};
template <typename T1, typename T2, typename T3>
__aicore__ inline void OneHotMix<T1, T2, T3>::Init(
    GM_ADDR x, GM_ADDR depth, GM_ADDR on_value, GM_ADDR off_value, GM_ADDR y, GM_ADDR workspace, TPipe* pipe,
    const OneHotTilingData* tilingData)
{
    pipe_ = pipe;
    tiling = tilingData;
    blockIdx_ = GetBlockIdx();
    totalInputDim_ = tiling->prefixDim * tiling->suffixDim;
    inputGm.SetGlobalBuffer((__gm__ T1*)x, totalInputDim_);
    // output reuse input
    depthGm.SetGlobalBuffer((__gm__ T2*)depth, 1);
    depth_ = static_cast<int64_t>(depthGm.GetValue(0));
    outputGm.SetGlobalBuffer((__gm__ T3*)y, totalInputDim_ * depth_);
    onValueGm.SetGlobalBuffer((__gm__ T3*)on_value, 1);
    offValueGm.SetGlobalBuffer((__gm__ T3*)off_value, 1);
    singleOffValueCalNum_ = tiling->offValuePerCoreCalSize;
    int64_t actualOffValueCalNum = singleOffValueCalNum_;
    int64_t totalCalSize = totalInputDim_ * depth_;
    if (blockIdx_ == (tiling->offValueUsedCoreNum - 1) && tiling->offValueHasTail) {
        if (tiling->offValueUsedCoreNum >= tiling->coreNum) {
            actualOffValueCalNum = singleOffValueCalNum_ + totalCalSize % tiling->coreNum;
        } else {
            actualOffValueCalNum = totalCalSize % MIN_UB_INIT_SIZE;
        }
    }
    int32_t offValueMaxNum = tiling->ubSize / sizeof(T3);
    if (offValueMaxNum >= actualOffValueCalNum) {
        offValueInitLoop_ = 1;
        singleCalNum_ = actualOffValueCalNum;
        tailOffValueCalNum_ = actualOffValueCalNum;
    } else {
        int64_t actualLastCoreTotalSize = actualOffValueCalNum;
        offValueInitLoop_ = (actualLastCoreTotalSize + offValueMaxNum - 1) / offValueMaxNum;
        singleCalNum_ = offValueMaxNum;
        tailOffValueCalNum_ = actualLastCoreTotalSize % offValueMaxNum;
    }
    pipe_->InitBuffer(offValueUb, tiling->ubSize);
    onValue_ = static_cast<T3>(onValueGm.GetValue(0));
    offValue_ = static_cast<T3>(offValueGm.GetValue(0));
    usedCoreNum_ = tiling->usedCoreNum;
    blockFactor_ = tiling->perCoreCalSize;
    factor1 = tiling->suffixDim;

    factorOut1 = tiling->suffixDim * depth_;
    factorOut2 = tiling->suffixDim;
    // fast division
    auto lz = BIT64 - clz(factor1);
    auto bc = bcnt1(factor1);
    shift_ = (bc != 1) ? lz : lz - 1;
    uint64_t magic = ((ONE << BIT32) * ((ONE << shift_) - factor1)) / factor1 + 1;
    m_ = static_cast<uint32_t>(magic);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void OneHotMix<T1, T2, T3>::CopyOut(LocalTensor<T3>& srcTensor, int64_t offset, int32_t copyLength)
{
    DataCopyPad(
        outputGm[offset], srcTensor,
        {static_cast<uint16_t>(1), static_cast<uint32_t>(copyLength * sizeof(T3)), 0, 0, 0});
}

template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(BOUND_THREAD_NUM) inline void SimtComputeFor64Mix(
    int64_t startOffset, int64_t coreFactor, int64_t depth, T3 offValue, T3 onValue, uint64_t factor1,
    uint64_t factorOut1, uint64_t factorOut2, __gm__ T1* inputData, __gm__ T3* outputData)
{
    for (int64_t idx = static_cast<int64_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<int64_t>(blockDim.x)) {
        int64_t gmIdx = startOffset + idx;
        int64_t i = gmIdx / factor1;
        int64_t j = gmIdx % factor1;
        int64_t dstOffset;
        if (inputData[gmIdx] < depth && inputData[gmIdx] >= 0) {
            dstOffset = i * factorOut1 + inputData[gmIdx] * factorOut2 + j;
            outputData[dstOffset] = onValue;
        }
    }
}

template <typename T1, typename T2, typename T3>
__simt_vf__ __aicore__ LAUNCH_BOUND(BOUND_THREAD_NUM) inline void SimtComputeMix(
    int32_t startOffset, int32_t coreFactor, int64_t depth, T3 offValue, T3 onValue, uint64_t factor1,
    uint64_t factorOut1, uint64_t factorOut2, uint32_t m, uint32_t shift, __gm__ T1* inputData, __gm__ T3* outputData)
{
    for (uint32_t idx = static_cast<int32_t>(threadIdx.x); idx < coreFactor;
         idx += static_cast<int32_t>(blockDim.x)) {
        int32_t gmIdx = startOffset + idx;
        uint32_t t = __umulhi(gmIdx, m);
        t = t + gmIdx;
        int32_t i = t >> shift;
        int32_t j = gmIdx - i * factor1;
        int64_t dstOffset;
        if (inputData[gmIdx] < depth && inputData[gmIdx] >= 0) {
            dstOffset = i * factorOut1 + inputData[gmIdx] * factorOut2 + j;
            outputData[dstOffset] = onValue;
        }
    }
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void OneHotMix<T1, T2, T3>::InitOffValueAndCopyOut(void)
{
    int64_t offValueOffset = blockIdx_ * tiling->offValuePerCoreCalSize;
    int32_t copyNum = singleCalNum_;
    LocalTensor<T3> offValueTensor = offValueUb.Get<T3>();
    int32_t initSize = singleCalNum_ > tailOffValueCalNum_ ? singleCalNum_ : tailOffValueCalNum_;
    Duplicate(offValueTensor, offValue_, initSize);
    event_t eventIdV2Mte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    SetFlag<HardEvent::V_MTE3>(eventIdV2Mte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV2Mte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdV2Mte3);
    for (int64_t idx = 0; idx < offValueInitLoop_ - 1; idx++) {
        CopyOut(offValueTensor, offValueOffset, copyNum);
        offValueOffset += copyNum;
    }
    // last loop
    copyNum = tailOffValueCalNum_;
    CopyOut(offValueTensor, offValueOffset, copyNum);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void OneHotMix<T1, T2, T3>::Process()
{
    if (blockIdx_ < tiling->offValueUsedCoreNum) {
        InitOffValueAndCopyOut();
    }
    SyncAll<true>();
    if (blockIdx_ >= tiling->usedCoreNum) {
        return;
    }
    int64_t coreFactor = blockFactor_;
    if (tiling->hasTail && blockIdx_ > 0 && blockIdx_ == usedCoreNum_ - 1) {
        // 尾块核
        if (totalInputDim_ <= tiling->coreNum * MIN_FACTOR) {
            coreFactor = totalInputDim_ % MIN_FACTOR;
        } else {
            coreFactor = blockFactor_ + totalInputDim_ % tiling->coreNum;
        }
    }
    int64_t startOffset = blockFactor_ * blockIdx_;
    int64_t endOffset = startOffset + coreFactor;
    auto threadsPerBlock = cce::dim3{static_cast<unsigned int>(THREAD_NUM)};
    __gm__ T1* inputData = inputGm.GetPhyAddr(0);
    __gm__ T3* outputData = outputGm.GetPhyAddr(0);
    if (unlikely(endOffset > INT32_MAX)) {
        asc_vf_call<SimtComputeFor64Mix<T1, T2, T3>>(
            threadsPerBlock, startOffset, coreFactor, depth_, offValue_, onValue_, factor1, factorOut1, factorOut2,
            inputData, outputData);
    } else {
        asc_vf_call<SimtComputeMix<T1, T2, T3>>(
            threadsPerBlock, static_cast<int32_t>(startOffset), static_cast<int32_t>(coreFactor), depth_, offValue_,
            onValue_, factor1, factorOut1, factorOut2, m_, shift_, inputData, outputData);
    }
}
} // namespace OneHot
#endif