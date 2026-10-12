/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file diag_part_simt.h
 * \brief diag_part_simt
 */

#ifndef DIAG_PART_SIMT_H
#define DIAG_PART_SIMT_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

constexpr int32_t DOUBLE_THREAD_DIM = 2048;
constexpr int32_t THREAD_DIM = 1024;
constexpr int32_t HALF_THREAD_DIM = 512;
constexpr int32_t QUARTER_THREAD_DIM = 256;
constexpr int32_t AN_EIGHTH_THREAD_DIM = 128;

namespace DiagPart {
using namespace AscendC;

template <typename T> // T原始数据类型  U做sizeSplits的数据类型
class DiagPartSimt {  // 每个核处理的数据小于128K，才会有这个模版，所以数据类型都可以降级
public:
    __aicore__ inline DiagPartSimt(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline __gm__ T* GetTensorAddr(uint32_t index);

private:
    const DiagPartTilingData* tilingData_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int32_t blockIdx_ = 0;
    int32_t blockNums_ = 0;
    int64_t curBlockLoopNumStart_ = 0;
    int64_t xGmOffset_ = 0;
    int64_t yGmOffset_ = 0;
    int64_t sideLen_ = 0;
};

template <typename T>
__simt_vf__ LAUNCH_BOUND(DOUBLE_THREAD_DIM) __aicore__ void ProcessDoubleSingleBlock(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, int64_t sideLen, int64_t processNum, int64_t startBlock,
    int64_t blockLen)
{
    int64_t startSideLen = startBlock * blockLen;
    for (int64_t idx = startSideLen + threadIdx.x; idx < startSideLen + processNum;
         idx += blockDim.x) {
        int64_t inputIndex = idx * sideLen + idx;
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM) __aicore__ void ProcessSingleBlock(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, int64_t sideLen, int64_t processNum, int64_t startBlock,
    int64_t blockLen)
{
    int64_t startSideLen = startBlock * blockLen;
    for (int64_t idx = startSideLen + threadIdx.x; idx < startSideLen + processNum;
         idx += blockDim.x) {
        int64_t inputIndex = idx * sideLen + idx;
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(HALF_THREAD_DIM) __aicore__ void ProcessSingleBlockHalf(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, int64_t sideLen, int64_t processNum, int64_t startBlock,
    int64_t blockLen)
{
    int64_t startSideLen = startBlock * blockLen;
    for (int64_t idx = startSideLen + threadIdx.x; idx < startSideLen + processNum;
         idx += blockDim.x) {
        int64_t inputIndex = idx * sideLen + idx;
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(QUARTER_THREAD_DIM) __aicore__ void ProcessSingleBlockQuarter(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, int64_t sideLen, int64_t processNum, int64_t startBlock,
    int64_t blockLen)
{
    int64_t startSideLen = startBlock * blockLen;
    for (int64_t idx = startSideLen + threadIdx.x; idx < startSideLen + processNum;
         idx += blockDim.x) {
        int64_t inputIndex = idx * sideLen + idx;
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T>
__simt_vf__ LAUNCH_BOUND(AN_EIGHTH_THREAD_DIM) __aicore__ void ProcessSingleBlockEighth(
    __gm__ T* inputGM, __gm__ volatile T* outputGM, int64_t sideLen, int64_t processNum, int64_t startBlock,
    int64_t blockLen)
{
    int64_t startSideLen = startBlock * blockLen;
    for (int64_t idx = startSideLen + threadIdx.x; idx < startSideLen + processNum;
         idx += blockDim.x) {
        int64_t inputIndex = idx * sideLen + idx;
        outputGM[idx] = inputGM[inputIndex];
    }
}

template <typename T>
__aicore__ inline void DiagPartSimt<T>::Init(GM_ADDR x, GM_ADDR y, const DiagPartTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    sideLen_ = tilingData_->sideLength;
    blockNums_ = (sideLen_ + tilingData_->numPerCore - 1) / tilingData_->numPerCore;
}

template <typename T>
__aicore__ inline void DiagPartSimt<T>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }
    int64_t processNum = 0;

    int64_t loopNum = blockNums_ / tilingData_->realCoreNum;
    int64_t mainCoreNum = blockNums_ % tilingData_->realCoreNum;
    int64_t blockLen = tilingData_->numPerCore;

    curBlockLoopNumStart_ = blockIdx_ < mainCoreNum ? blockIdx_ * (loopNum + 1) :
                                                      mainCoreNum * (loopNum + 1) + (blockIdx_ - mainCoreNum) * loopNum;
    if (blockIdx_ == tilingData_->realCoreNum - 1 && tilingData_->tailNum != 0) {
        processNum = blockIdx_ < mainCoreNum ? loopNum * tilingData_->numPerCore + tilingData_->tailNum :
                                               (loopNum - 1) * tilingData_->numPerCore + tilingData_->tailNum;
    } else {
        processNum =
            blockIdx_ < mainCoreNum ? (loopNum + 1) * tilingData_->numPerCore : loopNum * tilingData_->numPerCore;
    }
    if (processNum <= AN_EIGHTH_THREAD_DIM) {
        asc_vf_call<ProcessSingleBlockEighth<T>>(
            dim3(AN_EIGHTH_THREAD_DIM), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), sideLen_,
            processNum, curBlockLoopNumStart_, blockLen);
    } else if (processNum <= QUARTER_THREAD_DIM) {
        asc_vf_call<ProcessSingleBlockQuarter<T>>(
            dim3(QUARTER_THREAD_DIM), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), sideLen_,
            processNum, curBlockLoopNumStart_, blockLen);
    } else if (processNum <= HALF_THREAD_DIM) {
        asc_vf_call<ProcessSingleBlockHalf<T>>(
            dim3(HALF_THREAD_DIM), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), sideLen_,
            processNum, curBlockLoopNumStart_, blockLen);
    } else if (processNum <= THREAD_DIM) {
        asc_vf_call<ProcessSingleBlock<T>>(
            dim3(THREAD_DIM), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), sideLen_,
            processNum, curBlockLoopNumStart_, blockLen);
    } else {
        asc_vf_call<ProcessDoubleSingleBlock<T>>(
            dim3(DOUBLE_THREAD_DIM), (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), sideLen_,
            processNum, curBlockLoopNumStart_, blockLen);
    }
}

} // namespace DiagPart
#endif // DIAG_PART_SIMT_H