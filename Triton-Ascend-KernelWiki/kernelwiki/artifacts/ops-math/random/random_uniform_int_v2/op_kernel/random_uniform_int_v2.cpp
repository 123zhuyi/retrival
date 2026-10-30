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
* \file random_uniform_int_v2.cpp
* \brief
*/

#include "arch35/random_uniform_int_v2_struct.h"
#include "arch35/random_uniform_int_v2.h"
using namespace AscendC;
using namespace RandomUniformIntV2;

enum class RandomUniformIntV2TilingKey : uint32_t
{
    RANDOM_UNIFORM_INT = 0,
};

template <uint32_t opType>
__global__ __aicore__ void random_uniform_int_v2(GM_ADDR shape, GM_ADDR min, GM_ADDR max, GM_ADDR inOffset, GM_ADDR y, GM_ADDR outOffset, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUniformIntV2TilingData4RegBase);
    GET_TILING_DATA_WITH_STRUCT(RandomUniformIntV2TilingData4RegBase, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    if constexpr(opType == static_cast<uint32_t>(RandomUniformIntV2TilingKey::RANDOM_UNIFORM_INT)) {
        RandomUniformIntV2::RandomUniformIntV2Op<DTYPE_Y> op(&pipe, &tilingData);
        op.Init(y, outOffset);
        op.Process();
    }
}