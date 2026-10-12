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
 * \file amp_update_scale.cpp
 * \brief
 */
#include "amp_update_scale.h"

extern "C" __global__ __aicore__ void amp_update_scale(GM_ADDR current_scale, GM_ADDR growth_tracker, GM_ADDR found_inf, GM_ADDR updated_scale, GM_ADDR updated_growth_tracker, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) {
        AmpUpdateScale<float> op;
        op.Init(current_scale, growth_tracker, found_inf, updated_scale, updated_growth_tracker, tiling_data.growthFactor, tiling_data.backoffFactor, tiling_data.growthInterval, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AmpUpdateScale<half> op;
        op.Init(current_scale, growth_tracker, found_inf, updated_scale, updated_growth_tracker, tiling_data.growthFactor, tiling_data.backoffFactor, tiling_data.growthInterval, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AmpUpdateScale<bfloat16_t> op;
        op.Init(current_scale, growth_tracker, found_inf, updated_scale, updated_growth_tracker, tiling_data.growthFactor, tiling_data.backoffFactor, tiling_data.growthInterval, &pipe);
        op.Process();
    }
}