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
 * \file transform_bias_rescale_qkv.cpp
 * \brief transform_bias_rescale_qkv kernel
 */

#include "transform_bias_rescale_qkv.h"

using namespace AscendC;

using namespace TransformBiasRescaleQkv;

extern "C" __global__ __aicore__ void transform_bias_rescale_qkv(
    GM_ADDR qkv, GM_ADDR qkv_bias, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWs = nullptr;

#if __CCE_AICORE__ >= 220
    if (TILING_KEY_IS(1)) {
        TransformBiasRescaleQkvND<DTYPE_QKV> op;
        op.Init(qkv, qkv_bias, q, k, v, userWs, &tilingData);
        op.Process();
    }
#else
#endif
}