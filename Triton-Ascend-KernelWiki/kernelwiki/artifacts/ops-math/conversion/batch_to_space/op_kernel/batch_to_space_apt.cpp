/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/batch_to_space.h"
#include "arch35/batch_to_space_tiling_data.h"
#include "arch35/batch_to_space_tiling_key.h"

using namespace NsBatchToSpace;

// template 版本供 precompile 阶段解析 kernel entry 名称
template <uint8_t UbAxis>
__global__ __aicore__ void batch_to_space(GM_ADDR x, GM_ADDR crops, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(BatchToSpaceTilingData);
    GET_TILING_DATA_WITH_STRUCT(BatchToSpaceTilingData, tilingData, tiling);
    BatchToSpace<DTYPE_X, UbAxis> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
