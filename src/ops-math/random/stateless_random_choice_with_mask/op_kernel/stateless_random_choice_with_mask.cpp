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
 * \file stateless_random_choice_with_mask.cpp
 * \brief
 */

#include "arch35/stateless_random_choice_with_mask_struct.h"
#include "arch35/stateless_random_choice_with_mask.h"

enum class StatelessRandomChoiceWithMaskTilingKey : uint32_t
{
    DEFAULT = 0
};

template <uint32_t schMode>
__global__ __aicore__ void stateless_random_choice_with_mask(
    GM_ADDR x, GM_ADDR count, GM_ADDR seed, GM_ADDR offset, GM_ADDR y, GM_ADDR mask, GM_ADDR shape_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StatelessRandomChoiceWithMaskSimtTilingData);
    GET_TILING_DATA_WITH_STRUCT(StatelessRandomChoiceWithMaskSimtTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    if constexpr (schMode == static_cast<uint32_t>(StatelessRandomChoiceWithMaskTilingKey::DEFAULT)) {
        StatelessRandomChoiceWithMask::StatelessRandomChoiceWithMask op;
        op.Init(x, count, y, mask, shape_out, workspace, &tilingData, &pipe);
        op.Process();
    }
}
