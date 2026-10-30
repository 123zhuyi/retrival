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
 * \file gather_v2_simt_two_dim.h
 * \brief
 */
#ifndef MOE_GATHER_V2_SIMT_TWO_DIM_H
#define MOE_GATHER_V2_SIMT_TWO_DIM_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_TWO_DIM = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_TWO_DIM = 2048;
#endif

namespace gatherv2 {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
class Gatherv2SimtTwoDim {
public:
    __aicore__ inline Gatherv2SimtTwoDim(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                const MoeTokenPermuteWithRoutingMapTilingData *tilingData, bool isprob);
    __aicore__ inline void Process();
    __aicore__ inline void ProbProcess();

private:
    template <const bool NIS>
    static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void GatherSimt(
        const INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0,
        INDEX_SIZE_T innerSize, __gm__ X_T *x, __gm__ INDICES_T *indices, __gm__ volatile X_T *y);
    template <const bool NIS>
    static __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void GatherSimtProb(
        const INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0,
        INDEX_SIZE_T tokenNum, __gm__ X_T *x, __gm__ INDICES_T *indices, __gm__ volatile X_T *y);

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> yGm_;
    const IndexCopyComputeRMTilingData *tilingData_ = nullptr;
    int32_t blockIdx = 0;
    int32_t needCoreNum = 0;
    uint32_t threadNum = 0;
    INDEX_SIZE_T innerSize = 0;
    INDEX_SIZE_T currentCoreElements = 0;
    INDEX_SIZE_T CoreElements = 0;
    INDEX_SIZE_T capacity = 0;
    INDEX_SIZE_T m0 = 0;
    INDEX_SIZE_T shift0 = 0;
    INDEX_SIZE_T tokenNum = 0;
};

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
template <const bool NIS>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::GatherSimt(
    const INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T m0, INDEX_SIZE_T shift0,
    INDEX_SIZE_T innerSize, __gm__ X_T *x, __gm__ INDICES_T *indices, __gm__ volatile X_T *y)
{
    for (INDEX_SIZE_T index = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()); index < currentCoreElements;
         index += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum())) {
        INDEX_SIZE_T yIndex = yIndexBase + index;
        INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
        INDEX_SIZE_T innerI = yIndex - gatherI * innerSize;

        INDICES_T indicesValue = indices[gatherI];

        INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
        INDEX_SIZE_T xIndex = indicesValueI * innerSize + innerI;
        y[yIndex] = x[xIndex];
    }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
template <const bool NIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_TWO_DIM) inline void Gatherv2SimtTwoDim<
    X_T, INDICES_T, INDEX_SIZE_T>::GatherSimtProb(const INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements,
                                                  INDEX_SIZE_T m0, INDEX_SIZE_T shift0, INDEX_SIZE_T tokenNum,
                                                  __gm__ X_T *x, __gm__ INDICES_T *indices, __gm__ volatile X_T *y)
{
    for (INDEX_SIZE_T index = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()); index < currentCoreElements;
         index += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum())) {
        INDEX_SIZE_T yIndex = yIndexBase + index;
        INDEX_SIZE_T expertId = Simt::UintDiv(yIndex, m0, shift0);
        INDEX_SIZE_T expertStartI = expertId * tokenNum;

        INDICES_T indicesValue = indices[yIndex];

        INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
        INDEX_SIZE_T xIndex = indicesValueI + expertStartI;
        y[yIndex] = x[xIndex];
    }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, const MoeTokenPermuteWithRoutingMapTilingData *tilingData, bool isprob)
{
    this->tilingData_ = &(tilingData->indexCopyComputeParamsOp);

    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T *)indices);
    yGm_.SetGlobalBuffer((__gm__ X_T *)y);
    blockIdx = static_cast<int32_t>(GetBlockIdx());
    if (isprob) {
        needCoreNum = static_cast<int32_t>(tilingData_->frontCoreNum);
        threadNum = static_cast<uint32_t>(tilingData_->onceIndices);
        innerSize = 1;
        currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->frontCoreLoop);
        CoreElements = currentCoreElements;
        if (blockIdx == needCoreNum - 1) {
            currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->tailCoreLoop);
        }
        capacity = static_cast<INDEX_SIZE_T>(tilingData->capacity);
        tokenNum = static_cast<INDEX_SIZE_T>(tilingData->n);

        // fast division
        GetUintDivMagicAndShift(m0, shift0, capacity);
    } else {
        needCoreNum = static_cast<int32_t>(tilingData_->needCoreNum);
        threadNum = static_cast<uint32_t>(tilingData_->onceIndicesTokenNums);
        innerSize = static_cast<INDEX_SIZE_T>(tilingData_->onceUbTokenNums);
        currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->coreCalcNum);
        CoreElements = currentCoreElements;
        if (blockIdx == needCoreNum - 1) {
            currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->tailCoreNum);
        }
        // fast division
        GetUintDivMagicAndShift(m0, shift0, innerSize);
    }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::Process()
{
    if (blockIdx < needCoreNum) {
        INDEX_SIZE_T yIndexBase = blockIdx * CoreElements;

        AscendC::Simt::VF_CALL<GatherSimt<false>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0, shift0,
                                                  innerSize, (__gm__ X_T *)(xGm_.GetPhyAddr()),
                                                  (__gm__ INDICES_T *)(indicesGm_.GetPhyAddr()),
                                                  (__gm__ volatile X_T *)(yGm_.GetPhyAddr()));
    }
}

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
__aicore__ inline void Gatherv2SimtTwoDim<X_T, INDICES_T, INDEX_SIZE_T>::ProbProcess()
{
    if (blockIdx < needCoreNum) {
        INDEX_SIZE_T yIndexBase = blockIdx * CoreElements;

        AscendC::Simt::VF_CALL<GatherSimtProb<false>>(Simt::Dim3(threadNum), yIndexBase, currentCoreElements, m0,
                                                      shift0, tokenNum, (__gm__ X_T *)(xGm_.GetPhyAddr()),
                                                      (__gm__ INDICES_T *)(indicesGm_.GetPhyAddr()),
                                                      (__gm__ volatile X_T *)(yGm_.GetPhyAddr()));
    }
}
} // namespace gatherv2
#endif // MOE_GATHER_V2_SIMT_TWO_DIM_H
