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
 * \file sim_thread_exponential.cpp
 * \brief kernel entry for sim_thread_exponential operator
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// 950 chip: SIMT implementation
#include "arch35/sim_thread_exponential_simt.h"

extern "C" __global__ __aicore__ void sim_thread_exponential(
    GM_ADDR self, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(RandomUnifiedSimtTilingDataStruct, tilingData, tiling);

    if (TILING_KEY_IS(3)) {
        SimThreadExponential::Process<float>(self, &tilingData);
    } else if (TILING_KEY_IS(1)) {
        SimThreadExponential::Process<half>(self, &tilingData);
    } else if (TILING_KEY_IS(2)) {
        SimThreadExponential::Process<bfloat16_t>(self, &tilingData);
    }
}

#else
// 910b and other chips: SIMD implementation (original)
#include "sim_thread_exponential.h"

extern "C" __global__ __aicore__ void sim_thread_exponential(
    GM_ADDR self, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(3)) {
        SimThreadExponential<float> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        SimThreadExponential<half> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        SimThreadExponential<bfloat16_t> op(&pipe);
        op.Init(self, self_ref, usrWorkspace, &tiling_data);
        op.Process();
    }
}

#endif