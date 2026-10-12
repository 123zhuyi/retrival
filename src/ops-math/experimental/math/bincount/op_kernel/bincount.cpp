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
 * \file bincount.cpp
 * \brief bincount kernel entry
 */

#include "bincount.h"

template <uint32_t schMode>
__global__ __aicore__ void bincount(GM_ADDR array, GM_ADDR size, GM_ADDR weights, GM_ADDR bins, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BincountTilingData);
    GET_TILING_DATA_WITH_STRUCT(BincountTilingData, tilingData, tiling);
    (void)size;
    NsBincount::KernelBincount<DTYPE_ARRAY, DTYPE_BINS> op;
    op.Init(array, weights, bins, workspace, &tilingData);
    op.Process();
}
