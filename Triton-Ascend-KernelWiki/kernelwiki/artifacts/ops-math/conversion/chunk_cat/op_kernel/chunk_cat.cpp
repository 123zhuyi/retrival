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
 * \file chunk_cat.cpp
 * \brief
 */
#if __CCE_AICORE__ == 310
#include "chunk_cat_arch35.h"
#else
#include "chunk_cat.h"
#endif

extern "C" __global__ __aicore__ void chunk_cat(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    REGISTER_TILING_DEFAULT(ChunkCatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkCatTilingData, tilingData, tiling);
#if __CCE_AICORE__ == 310
    ChunkCatArch35<DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, y, tilingData);
    op.Process();
#else
    #if (ORIG_DTYPE_X == ORIG_DTYPE_Y)
        ChunkCat<DTYPE_X, DTYPE_Y> op(&pipe);
        op.Init(x, y, tilingData);
        op.Process();
    #elif (ORIG_DTYPE_X != ORIG_DTYPE_Y)
        ChunkCat<DTYPE_X, DTYPE_Y, true> op(&pipe);
        op.Init(x, y, tilingData);
        op.Process();
    #endif
#endif
}