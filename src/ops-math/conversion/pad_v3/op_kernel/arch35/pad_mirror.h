/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file pad_reflect.h
 * \brief pad_reflect
 */
#ifndef ASCENDC_PAD_MIRROR_H
#define ASCENDC_PAD_MIRROR_H

#include "pad_mirror_simt.h"
#include "pad_mirror_simt_huge.h"
#include "pad_mirror_normal_w.h"
#include "pad_mirror_gather.h"
#include "pad_mirror_huge_width.h"

namespace PadV3 {
template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadMirrorSimt(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadReflectSimt<int32_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadReflectSimt<int64_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadReflectSimt<int8_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadReflectSimt<T, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadMirrorSimtHuge(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadReflectSimtHuge<int32_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadReflectSimtHuge<int64_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadReflectSimtHuge<int8_t, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadReflectSimtHuge<T, KEY> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadMirrorWithNormalWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadMirrWithNormalWidth<uint8_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadMirrWithNormalWidth<uint16_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadMirrWithNormalWidth<uint32_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadMirrWithNormalWidth<uint64_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadMirrorWithHugeWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadMirrorWithHugeWidth<uint8_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadMirrorWithHugeWidth<uint16_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadMirrorWithHugeWidth<uint32_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadMirrorWithHugeWidth<uint64_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadMirrorGather(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadMirrorGather<uint8_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        PadMirrorGather<uint16_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadMirrorGather<uint32_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadMirrorGather<uint64_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    }
}

} // namespace PadV3

#endif
