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
 * \file diag_flat_common.h
 * \brief Common functions shared by diag_flat_nd_to_2d kernel variants
 */
#ifndef DIAG_FLAT_COMMON_H
#define DIAG_FLAT_COMMON_H

#include "kernel_operator.h"

using namespace AscendC;

namespace DiagFlat {

/*
 * Common MemSetZero function shared across diag_flat_nd_to_2d variants
 */
template <typename U>
__aicore__ inline void DiagFlatMemSetZero(TPipe* pipe, GlobalTensor<U> gmTensor, int64_t size)
{
    if (g_coreType == AIC) {
        return;
    }
    int64_t int16Size = (size * sizeof(U) + sizeof(int16_t) - 1) / sizeof(int16_t);
    LocalTensor<int16_t> popBuffer;
    bool ret = PopStackBuffer<int16_t, TPosition::LCM>(popBuffer);
    uint32_t maxBurstSize = (MAX_REPEAT_TIMES * ONE_BLK_SIZE) / sizeof(int16_t);
    uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    uint32_t round = int16Size / popSize;
    uint32_t tail = int16Size % popSize;
    uint32_t roundSize = round != 0 ? popSize : 0;
    AscendC::Duplicate<int16_t>(popBuffer, static_cast<int16_t>(0), popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    uint32_t comOffset = 0;
    // compute the main block
    for (int index = 0; index < round; ++index) {
        DataCopyUB2GMImpl(
            (__gm__ int16_t*)gmTensor.GetPhyAddr() + comOffset, (__ubuf__ int16_t*)popBuffer.GetPhyAddr(),
            {1, static_cast<uint16_t>((roundSize * sizeof(int16_t) + ONE_BLK_SIZE - 1) / (ONE_BLK_SIZE)), 0, 0});
        comOffset += roundSize;
    }
    // compute the tail block
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyUB2GMImpl((__gm__ int16_t*)gmTensor.GetPhyAddr() + comOffset, (__ubuf__ int16_t*)popBuffer.GetPhyAddr(),
                          {1, static_cast<uint16_t>((tail * sizeof(int16_t) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE), 0, 0});
    }
}

/*
 * Common Init logic shared by diag_flat b16 variants (b16_less and b16_more64)
 * Includes: workspace setup, inter-core sync setup, output clearing, SyncAll, param init
 */
template <typename T, bool IsDataCopyPadSupport>
__aicore__ inline void DiagFlatInitB16Common(TPipe* pipe, GlobalTensor<int32_t>& syncGlobal,
                                             TQue<QuePosition::VECIN, 1>& workQueue, GlobalTensor<T>& gmOutput,
                                             GlobalTensor<T>& gmInput, TQue<QuePosition::VECIN, 1>& inputQueue,
                                             TQue<QuePosition::VECOUT, 1>& outputQueue, GM_ADDR input, GM_ADDR output,
                                             GM_ADDR workspace, int64_t totalCoreNum, int64_t inputNum, int64_t offset,
                                             int64_t inputIdx)
{
    // set workspace as 0, each core handle workspace 32bytes
    constexpr int32_t EACH_CORE_HANDLE_NUM = 32 / sizeof(int32_t);
    syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, totalCoreNum * 32 / sizeof(int32_t));
    DiagFlatMemSetZero<int32_t>(pipe, syncGlobal, totalCoreNum * EACH_CORE_HANDLE_NUM);

    // 核间同步
    syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace, 1024 * sizeof(int32_t));
    pipe->InitBuffer(workQueue, 1, totalCoreNum * 8 * sizeof(int32_t));

    // gm输出进行清零
    int32_t coreNum = totalCoreNum;
    int64_t inputSumNum = (inputNum + abs(offset)) * (inputNum + abs(offset));
    int64_t cleanNum = (inputSumNum + coreNum - 1) / coreNum;

    if (GetBlockIdx() == coreNum - 1) {
        int64_t lastCleanNum = inputSumNum % cleanNum;
        if (lastCleanNum == 0) {
            lastCleanNum = cleanNum;
        }
        gmOutput.SetGlobalBuffer((__gm__ T*)output + (GetBlockIdx() * cleanNum));
        DiagFlatMemSetZero<T>(pipe, gmOutput, lastCleanNum);
    } else {
        gmOutput.SetGlobalBuffer((__gm__ T*)output + (GetBlockIdx() * cleanNum));
        DiagFlatMemSetZero<T>(pipe, gmOutput, cleanNum);
    }

    // 区分芯片使用syncAll同步
    if constexpr (IsDataCopyPadSupport) {
        SyncAll();
    } else {
        LocalTensor<int32_t> workLocal = workQueue.AllocTensor<int32_t>();
        SyncAll(syncGlobal, workLocal);
        workQueue.FreeTensor(workLocal);
    }

    // 参数初始化
    gmInput.SetGlobalBuffer((__gm__ T*)input + inputIdx * 2, inputNum);

    // 首地址偏移量
    gmOutput.SetGlobalBuffer(
        (__gm__ T*)output + (inputIdx + (offset > 0 ? 0 : abs(offset))) * (inputNum + abs(offset)) * 2,
        (inputNum + abs(offset)) * (inputNum + abs(offset)) * 2);

    // 申请内存空间
    pipe->InitBuffer(inputQueue, 1, 128 * sizeof(int64_t));
    pipe->InitBuffer(outputQueue, 1, 64 * 128 * sizeof(int64_t));
}

} // namespace DiagFlat

#endif // DIAG_FLAT_COMMON_H
