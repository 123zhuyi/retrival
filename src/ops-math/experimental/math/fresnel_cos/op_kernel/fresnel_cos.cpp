/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 *
 * FresnelCos kernel entry (arch35)
 *
 * Template parameter:
 *   schMode: 0 = FP32, 1 = FP16, 2 = BF16
 *            (maps to TilingKey defined in fresnel_cos_tiling_key.h)
 */

#include "arch35/fresnel_cos.h"

template <uint32_t schMode>
__global__ __aicore__ void fresnel_cos(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (schMode == FRESNEL_COS_KEY_FP32) {
        NsFresnelCos::KernelFresnelCos<float, FRESNEL_COS_KEY_FP32> op;
        op.Init(x, y, workspace, tiling);
        op.Process();
    } else if constexpr (schMode == FRESNEL_COS_KEY_FP16) {
        NsFresnelCos::KernelFresnelCos<half, FRESNEL_COS_KEY_FP16> op;
        op.Init(x, y, workspace, tiling);
        op.Process();
    } else if constexpr (schMode == FRESNEL_COS_KEY_BF16) {
        NsFresnelCos::KernelFresnelCos<bfloat16_t, FRESNEL_COS_KEY_BF16> op;
        op.Init(x, y, workspace, tiling);
        op.Process();
    }
}
