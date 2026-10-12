/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/c_types.h"
#include "../../../op_kernel/im2col.cpp"

extern "C" __global__ __aicore__ void im2col_fp16_contiguous(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    im2col<half, IM2COL_PATH_CONTIGUOUS_W>(x, y, workspace, tiling);
}

extern "C" __global__ __aicore__ void im2col_fp32_channel_template(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                   GM_ADDR tiling)
{
    im2col<float, IM2COL_PATH_CHANNEL_TEMPLATE>(x, y, workspace, tiling);
}
