/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file circular_pad_grad.cpp
 * \brief circular_pad_grad kernel
 */

#include "../pad_v3_grad/arch35/pad_v3_grad_struct.h"
#include "../pad_v3_grad/arch35/pad_v3_grad_tilingkey.h"
#include "../pad_v3_grad/arch35/pad_v3_grad_simt.h"
#include "../pad_v3_grad/arch35/pad_v3_grad_simd.h"

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

using namespace AscendC;

template <uint8_t modeName, bool isBigShape>
__aicore__ inline void PadV3GradSimt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (modeName == TPL_MODE_CONSTANT) {
        PadV3Grad::LaunchConstantKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_EDGE) {
        PadV3Grad::LaunchEdgeKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_REFLECT || modeName == TPL_MODE_SYMMETRIC) {
        PadV3Grad::LaunchMirrorKernelSIMT<DTYPE_X, isBigShape, modeName>(x, y, tiling);
    } else if constexpr (modeName == TPL_MODE_CIRCULAR) {
        PadV3Grad::LaunchCircularKernelSIMT<DTYPE_X, isBigShape>(x, y, tiling);
    }
}

template <uint8_t modeName, uint8_t cutMode, bool isBigShape>
__aicore__ inline void PadV3GradMirror(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (cutMode == TPL_SIMD_BIG) {
        PadV3Grad::LaunchKernelPadV3GradMirrorHugeWidth<DTYPE_X, modeName>(x, y, tiling);
    } else if constexpr (cutMode == TPL_SIMD_NORMAL) {
        // PadV3Grad::LaunchMirrorKernelNormal<DTYPE_X, modeName>(x, y, tiling);
        PadV3Grad::LaunchMirrorKernelSIMT<DTYPE_X, isBigShape, modeName>(x, y, tiling);
    } else if constexpr (cutMode == TPL_SIMD_SMALL) {
        PadV3Grad::LaunchMirrorKernelGather<DTYPE_X, true, modeName>(x, y, tiling);
    }
}

template <uint8_t modeName, uint8_t cutMode>
__aicore__ inline void PadV3GradCircular(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (cutMode == TPL_SIMD_BIG) {
        PadV3Grad::LaunchKernelPadV3GradCircularHugeWidth<DTYPE_X>(x, y, tiling);
    }
}

template <uint8_t modeName, bool isBigShape, bool isSimt, uint8_t cutMode>
__global__ __aicore__ void circular_pad_grad(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(PadV3GradACTilingData);
    if constexpr (isSimt) {
        PadV3GradSimt<modeName, isBigShape>(x, y, workspace, tiling);
    } else if constexpr (modeName == TPL_MODE_REFLECT || modeName == TPL_MODE_SYMMETRIC) {
        PadV3GradMirror<modeName, cutMode, isBigShape>(x, y, workspace, tiling);
    } else if constexpr (modeName == TPL_MODE_CIRCULAR) {
        PadV3GradCircular<modeName, cutMode>(x, y, workspace, tiling);
    }
}
