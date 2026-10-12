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
 * \file pows.cpp
 * \brief
 */

#include "pows_fp16.h"
#include "pows_fp32.h"
#include "pows_bf16.h"

using namespace Pows;

extern "C" __global__ __aicore__ void pows(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
#endif

    GET_TILING_DATA(tilingData, tiling);

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 100
    if (TILING_KEY_IS(101)) {
        Pows::PowsFp16<half> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(301)) {
        Pows::PowsFp32<float> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
#else
    if (TILING_KEY_IS(101)) {
        Pows::PowsFp16<half> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(201)) {
        Pows::PowsBfp16<bfloat16_t> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(301)) {
        Pows::PowsFp32<float> op;
        op.Init(x1, x2, y, workspace, &tilingData);
        op.Process();
    }
#endif
}
