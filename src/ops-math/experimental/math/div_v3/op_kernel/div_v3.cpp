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
 * \file div_v3.cpp
 * \brief DivV3 kernel entry point
 */

#include "div_v3.h"

template <uint32_t schMode>
__global__ __aicore__ void div_v3(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DivV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(DivV3TilingData, tilingData, tiling);
    NsDivV3::DivV3<DTYPE_X1> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
