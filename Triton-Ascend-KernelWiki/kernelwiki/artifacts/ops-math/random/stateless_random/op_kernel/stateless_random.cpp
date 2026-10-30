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
 * \file stateless_random.cpp
 * \brief Stateless random number generation kernel with SIMT template
 */

#include "arch35/stateless_random_impl.h"

using namespace StatelessRandom;

namespace {
#define STATELESS_RANDOM_TILING_KEY 100
} // namespace

extern "C" __global__ __aicore__ void stateless_random(
    GM_ADDR shape, GM_ADDR seed, GM_ADDR offset, GM_ADDR from, GM_ADDR to, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(STATELESS_RANDOM_TILING_KEY)) {
        StatelessRandomImpl<DTYPE_Y> op(&pipe, &tilingData);
        op.Init(y);
        op.Process(seed, offset);
    }
}