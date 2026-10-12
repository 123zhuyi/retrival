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
 * \file equal.cpp
 * \brief
 */

#include "equal.h"

using namespace NsEqual;

template <uint32_t schMode>
__global__ __aicore__ void equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(EqualTilingData, tiling_data, tiling);

    KernelEqual<DTYPE_X1> op;

    op.Init(
        x1, x2, y, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
        tiling_data.finalSmallTileNum, tiling_data.tileDataNum, tiling_data.smallTailDataNum,
        tiling_data.bigTailDataNum, tiling_data.tailBlockNum, tiling_data.bigprocessDataNum_computes,
        tiling_data.smallprocessDataNum_computes, tiling_data.tailbigprocessDataNum_computes,
        tiling_data.tailsmallprocessDataNum_computes);
    op.Process();
}
