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
 * \file pad_v3_grad_simd.h
 * \brief pad_v3_grad_simd
 */
#ifndef ASCENDC_PAD_V3_GRAD_SIMD_H
#define ASCENDC_PAD_V3_GRAD_SIMD_H

#include "pad_v3_grad_mirror_normal.h"
#include "pad_v3_grad_mirror_gather.h"
#include "pad_v3_grad_mirror_huge_width.h"
#include "pad_v3_grad_circular_huge_width.h"

namespace PadV3Grad {

template <typename T, bool isBigShape, uint8_t modeName>
__aicore__ inline void LaunchMirrorKernelGather(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    PadV3GradGather<T, modeName> op;
    op.Init(x, y, &tilingData);
    op.Process();
}

template <typename T, uint8_t modeName>
__aicore__ inline void LaunchMirrorKernelNormal(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    KernelPadV3GradMirrWithNormalWidth<T, modeName> op(&pipe, &tilingData);
    op.Init(x, y);
    op.Process();
}

template <typename T, uint8_t modeName>
__aicore__ inline void LaunchKernelPadV3GradMirrorHugeWidth(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    KernelPadV3GradMirrorHugeWidth<T, modeName> op(&pipe, &tilingData);
    op.Init(x, y);
    op.Process();
}

template <typename T>
__aicore__ inline void LaunchKernelPadV3GradCircularHugeWidth(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadV3GradACTilingData, tilingData, tiling);
    KernelPadV3GradCircularHugeWidth<T> op(&pipe, &tilingData);
    op.Init(x, y);
    op.Process();
}

} // namespace PadV3Grad

#endif
