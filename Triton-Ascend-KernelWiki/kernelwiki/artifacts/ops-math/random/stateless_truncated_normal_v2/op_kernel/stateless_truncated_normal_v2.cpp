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
 * \file stateless_truncated_normal_v2.cpp
 * \brief stateless_truncated_normal_v2 kernel entry
 */

#include "arch35/stateless_truncated_normal_v2_simt.h"

extern "C" __global__ __aicore__ void stateless_truncated_normal_v2(
    GM_ADDR shape, GM_ADDR key, GM_ADDR counter, GM_ADDR alg, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100)) {
        StatelessTruncatedNormalV2::Process<DTYPE_Y>(key, counter, y, &tilingData);
    }
}
