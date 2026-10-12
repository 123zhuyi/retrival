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
 * \file assign_sub.cpp
 * \brief
 */

#include "assign_sub.h"

enum class AssignSubTilingKey : uint32_t {
    TILING_KEY_ASSIGNSUB_MODE_0 = 0,
    TILING_KEY_ASSIGNSUB_MODE_1 = 1,
    TILING_KEY_ASSIGNSUB_MODE_2 = 2,
    TILING_KEY_ASSIGNSUB_MODE_3 = 3,
    TILING_KEY_ASSIGNSUB_MODE_4 = 4,
    TILING_KEY_ASSIGNSUB_MODE_5 = 5,
    TILING_KEY_ASSIGNSUB_MODE_6 = 6,
};

template <uint32_t schMode>
__global__ __aicore__ void assign_sub(GM_ADDR var, GM_ADDR value, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AssignSubTilingData);
    GET_TILING_DATA_WITH_STRUCT(AssignSubTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_0)) {
        NsAssignSub::AssignSub<half> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_1)) {
        NsAssignSub::AssignSub<int8_t> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_2)) {
        NsAssignSub::AssignSub<float> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_3)) {
        NsAssignSub::AssignSub<int32_t> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_4)) {
        NsAssignSub::AssignSub<uint8_t> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_5)) {
        NsAssignSub::AssignSub<bfloat16_t> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(AssignSubTilingKey::TILING_KEY_ASSIGNSUB_MODE_6)) {
        NsAssignSub::AssignSub<int64_t> op;
        op.Init(var, value, var_out, &tilingData);
        op.Process();
    }
}
