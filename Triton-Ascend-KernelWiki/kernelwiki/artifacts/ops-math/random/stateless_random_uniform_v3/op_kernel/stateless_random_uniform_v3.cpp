/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/stateless_random_uniform_v3.h"

using namespace StatelessRandomUniformV3Simd;

namespace {
#define STATELESSRANDOMUNIFORMV3_TILING_KEY 100
} // namespace

extern "C" __global__ __aicore__ void stateless_random_uniform_v3(
    GM_ADDR shape, GM_ADDR key, GM_ADDR counter, GM_ADDR from, GM_ADDR to, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedTilingDataStruct);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)

    AscendC::TPipe pipe;
    if (TILING_KEY_IS(STATELESSRANDOMUNIFORMV3_TILING_KEY)) {
        StatelessRandomUniformV3<DTYPE_Y> op(&pipe, &tilingData);
        op.Init(y, key, counter, from, to);
        op.Process();
    }
}