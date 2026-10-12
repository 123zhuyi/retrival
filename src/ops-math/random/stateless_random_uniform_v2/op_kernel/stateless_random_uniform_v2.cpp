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
 * \file stateless_random_uniform_v2.cpp
 * \brief
 */

#include "arch35/stateless_random_uniform_v2.h"

using namespace StatelessRandom;

namespace {
#define FLOAT_TILING_KEY 101
#define FLOAT16_TILING_KEY 102
#define BFLOAT16_TILING_KEY 103
} // namespace

extern "C" __global__ __aicore__ void stateless_random_uniform_v2(
    GM_ADDR shape, GM_ADDR key, GM_ADDR counter, GM_ADDR alg, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)
    TPipe pipe;
    if (TILING_KEY_IS(FLOAT_TILING_KEY)) {
        StatelessRandomUniformV2<float> op;
        op.Init(y, key, counter, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(FLOAT16_TILING_KEY)) {
        StatelessRandomUniformV2<half> op;
        op.Init(y, key, counter, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(BFLOAT16_TILING_KEY)) {
        StatelessRandomUniformV2<bfloat16_t> op;
        op.Init(y, key, counter, &tilingData, &pipe);
        op.Process();
    }
}