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
 * \file all_clear.h
 * \brief
 */

#ifndef ALL_CLEAR_H
#define ALL_CLEAR_H

#include "kernel_operator.h"

namespace StridedSliceGrad
{
using namespace AscendC;

template <typename T>
class AllClear
{
public:
    __aicore__ inline AllClear(){};
    __aicore__ inline void Init(GM_ADDR output, const StridedSliceGradTilingData& tilingData, TPipe& pipeIn);
    __aicore__ inline void Process();
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void SyncALLCoresSimt();

    TPipe pipe_;
    GlobalTensor<T> outputGm_;

    uint32_t usedCoreNum_;
    uint32_t normalCoreProcessNum_;
    uint32_t tailCoreProcessNum_;
    uint32_t blockIdx_;
};

template <typename T>
__aicore__ inline void AllClear<T>::Init(GM_ADDR output, const StridedSliceGradTilingData& tilingData, TPipe& pipeIn)
{
    // tiling_data
    usedCoreNum_ = tilingData.usedCoreNumForClear;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNumForClear;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNumForClear;

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_;

    // shield normal core and tail core
    if (blockIdx_ + 1 == usedCoreNum_) {
        normalCoreProcessNum_ = tailCoreProcessNum_;
    }

    outputGm_.SetGlobalBuffer((__gm__ T*)output + intraCoreOffset);
    pipe_ = pipeIn;
}

template <typename T>
__aicore__ inline void AllClear<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    InitOutput<T>(outputGm_, normalCoreProcessNum_, 0);
}

template <typename T>
__aicore__ inline void AllClear<T>::SyncALLCores()
{
    SyncAll();
}

template <typename T>
__aicore__ inline void AllClear<T>::SyncALLCoresSimt()
{
    PipeBarrier<PIPE_ALL>();
    SyncAll();
}
}  // namespace StridedSliceGrad

#endif  // ALL_CLEAR_H