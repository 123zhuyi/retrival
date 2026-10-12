/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "is_close.h"
#include "is_close_tiling_key.h"

template <uint32_t IS_CLOSE_TPL_KEY>
__global__ __aicore__ void is_close(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(IsCloseTilingData);
    GET_TILING_DATA_WITH_STRUCT(IsCloseTilingData, tilingData, tiling);
    constexpr uint32_t BROADCAST_MODE = IS_CLOSE_TPL_KEY / IS_CLOSE_TPL_DTYPE_COUNT;
    constexpr uint32_t DTYPE_MODE = IS_CLOSE_TPL_KEY % IS_CLOSE_TPL_DTYPE_COUNT + 1;
    NsIsClose::IsCloseKernelImpl<BROADCAST_MODE, DTYPE_MODE>(x1, x2, y, workspace, &tilingData);
}
