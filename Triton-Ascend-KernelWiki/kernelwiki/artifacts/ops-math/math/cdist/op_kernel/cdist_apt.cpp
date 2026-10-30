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
 * \file cdist.cpp
 * \brief
 */

#include "arch35/cdist_simt.h"
#include "arch35/cdist.h"
#include "cdist_tiling_data.h"
#include "cdist_tiling_key.h"

template <uint32_t isSmallM>
__global__ __aicore__ void cdist(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CdistTilingData);
    GET_TILING_DATA_WITH_STRUCT(CdistTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    if constexpr (isSmallM) {   // SIMT 场景
        NsCdist::CdistSimt<DTYPE_X1> op(&pipe, &tilingData);
        op.Init(x1, x2, y);
        op.Process();
    } else {    // Normal 场景
        NsCdist::Cdist<DTYPE_X1> op;
        op.Init(x1, x2, y, &tilingData, &pipe); 
        op.Process();          
    }
}