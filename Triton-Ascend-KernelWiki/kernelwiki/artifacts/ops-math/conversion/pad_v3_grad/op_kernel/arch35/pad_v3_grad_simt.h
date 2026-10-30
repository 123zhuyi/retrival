/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file pad_v3_grad_simt.h
 * \brief pad_v3_grad_simt
 */
#ifndef ASCENDC_PAD_V3_GRAD_SIMT_H
#define ASCENDC_PAD_V3_GRAD_SIMT_H

#include "pad_v3_grad_constant_simt.h"
#include "pad_v3_grad_circular_simt.h"
#include "pad_v3_grad_edge_simt.h"
#include "pad_v3_grad_mirror_simt.h"

namespace PadV3Grad {
template <typename T, bool isBigShape>
__aicore__ inline void LaunchConstantKernelSIMT(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    if constexpr (isBigShape) {
        PadV3GradConstantSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<uint64_t, int64_t>();
    } else {
        PadV3GradConstantSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<uint32_t, int32_t>();
    }
}

template <typename T, bool isBigShape>
__aicore__ inline void LaunchCircularKernelSIMT(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    if constexpr (isBigShape) {
        PadV3GradCircularSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<int64_t>();
    } else {
        PadV3GradCircularSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<int32_t>();
    }
}

template <typename T, bool isBigShape>
__aicore__ inline void LaunchEdgeKernelSIMT(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    if constexpr (isBigShape) {
        PadV3GradEdgeSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<int64_t>();
    } else {
        PadV3GradEdgeSimt<T> op;
        op.Init(x, y, &tilingData);
        op.template Process<int32_t>();
    }
}

template <typename T, bool isBigShape, uint8_t modeName>
__aicore__ inline void LaunchMirrorKernelSIMT(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    if constexpr (isBigShape) {
        PadV3GradMirrorSimt<T, modeName> op;
        op.Init(x, y, &tilingData);
        op.template Process<int64_t>();
    } else {
        PadV3GradMirrorSimt<T, modeName> op;
        op.Init(x, y, &tilingData);
        op.template Process<int32_t>();
    }
}

} // namespace PadV3Grad

#endif
