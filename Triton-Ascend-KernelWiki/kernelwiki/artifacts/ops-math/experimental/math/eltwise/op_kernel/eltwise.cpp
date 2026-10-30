/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file eltwise.cpp
 * \brief Eltwise kernel entry point (arch35 / Ascend950)
 *
 * DYNAMIC_INPUT: The framework passes a single GM_ADDR for the "inputs"
 * dynamic input group. This address points to a GM pointer table containing
 * the addresses of each individual input tensor. The actual number of inputs
 * is determined by TilingData.inputNum.
 */

#include "eltwise.h"

template <typename D_T, int MODE>
__global__ __aicore__ void eltwise(GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EltwiseTilingData);
    GET_TILING_DATA_WITH_STRUCT(EltwiseTilingData, tilingData, tiling);

    NsEltwise::Eltwise<D_T, MODE> op;
    op.Init(inputs, output, &tilingData);
    op.Process();
}
