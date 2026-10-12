/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef I_SOFTSHRINK_GRAD_TILING_H
#define I_SOFTSHRINK_GRAD_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/softshrink_grad_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"
#include "graph/c_types.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tikicpulib.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

inline void InitTilingData(uint8_t* tiling, SoftShrinkGradTilingData* constData)
{
    std::memcpy(constData, tiling, sizeof(SoftShrinkGradTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    SoftShrinkGradTilingData tilingData;       \
    InitTilingData(tilingArg, &tilingData)

#endif
