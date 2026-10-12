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
 * \file fused_mul_add_n.cpp
 * \brief A2 (DAV_2201 / ascend910b) flat default kernel entry for FusedMulAddN.
 *        y = x1 * x3[0] + x2 (elementwise). Dispatch by TilingKey (set by op_host tiling):
 *        0=fp32(direct) / 1=fp16(cast->fp32) / 2=int32(direct) / 3=int16(direct) / 4=bf16(cast->fp32).
 */

#include "kernel_operator.h"
#include "fused_mul_add_n_tiling_data.h"
#include "fused_mul_add_n_align.h"
#include "fused_mul_add_n_align_half.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void fused_mul_add_n(
    GM_ADDR inputX1, GM_ADDR inputX2, GM_ADDR inputX3, GM_ADDR outputY, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FusedMulAddNTilingData);
    GET_TILING_DATA_WITH_STRUCT(FusedMulAddNTilingData, tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        FusedMulAddNNs::FusedMulAddNAlign<float> op;
        op.Init(inputX1, inputX2, inputX3, outputY, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        FusedMulAddNNs::FusedMulAddNAlignHalf<half> op;
        op.Init(inputX1, inputX2, inputX3, outputY, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        TPipe pipe;
        FusedMulAddNNs::FusedMulAddNAlign<int32_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        TPipe pipe;
        FusedMulAddNNs::FusedMulAddNAlign<int16_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        TPipe pipe;
        FusedMulAddNNs::FusedMulAddNAlignHalf<bfloat16_t> op;
        op.Init(inputX1, inputX2, inputX3, outputY, &tilingData, &pipe);
        op.Process();
    }
}
