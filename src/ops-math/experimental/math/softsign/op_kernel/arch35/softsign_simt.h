/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softsign_simt.h
 * \brief softsign SIMT kernel: softsign(x) = x / (1 + |x|)
 */
#ifndef SOFTSIGN_SIMT_H
#define SOFTSIGN_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softsign_tiling_data.h"
#include "softsign_tiling_key.h"

namespace NsSoftsign {

using namespace AscendC;

constexpr uint32_t THREAD_NUM = 512;

template <typename T>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void OpSoftsignSimt(int64_t perCoreElements, __gm__ T* input, __gm__ T* output)
{
    for (uint64_t index = static_cast<uint64_t>(AscendC::Simt::GetThreadIdx());
         index < static_cast<uint64_t>(perCoreElements);
         index += static_cast<uint32_t>(AscendC::Simt::GetThreadNum())) {
        uint64_t globalIndex = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx() * perCoreElements) + index;
        T val = input[globalIndex];
        T absVal = (val < static_cast<T>(0)) ? -val : val;
        T result = val / (static_cast<T>(1) + absVal);
        output[globalIndex] = result;
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR input, GM_ADDR output, const SoftsignTilingData* tilingData)
{
    int64_t perCoreElements = tilingData->perCoreElements;
    __gm__ T* inputGm = (__gm__ T*)input;
    __gm__ T* outputGm = (__gm__ T*)output;
    AscendC::Simt::VF_CALL<OpSoftsignSimt<T>>(AscendC::Simt::Dim3(THREAD_NUM), perCoreElements, inputGm, outputGm);
}

} // namespace NsSoftsign
#endif // SOFTSIGN_SIMT_H
