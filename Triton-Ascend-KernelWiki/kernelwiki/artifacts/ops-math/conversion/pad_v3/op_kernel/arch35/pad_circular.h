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
 * \file pad_circular.h
 * \brief pad_circular
 */
#ifndef ASCENDC_PAD_CIRCULAR_H
#define ASCENDC_PAD_CIRCULAR_H

#include "pad_circular_simt.h"
#include "pad_circular_simt_huge.h"
#include "pad_circular_normal_w.h"
#include "pad_circular_gather.h"
#include "pad_circular_huge_width.h"

namespace PadV3 {
template <typename T>
__aicore__ inline void LaunchKernelPadCircularSimt(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadCircularSimt<int32_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadCircularSimt<int64_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadCircularSimt<int8_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadCircularSimt<T> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadCircularSimtHuge(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadCircularSimtHuge<int32_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadCircularSimtHuge<int64_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadCircularSimtHuge<int8_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadCircularSimtHuge<T> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadCircularWithHugeWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadCircularWithHugeWidth<uint8_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadCircularWithHugeWidth<uint16_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadCircularWithHugeWidth<uint32_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadCircularWithHugeWidth<uint64_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadCircularWithNormalWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadCircularWithNormalWidth<uint8_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadCircularWithNormalWidth<uint16_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadCircularWithNormalWidth<uint32_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadCircularWithNormalWidth<uint64_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadCircularGather(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadCircularGather<uint8_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        PadCircularGather<uint16_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadCircularGather<uint32_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadCircularGather<uint64_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    }
}

} // namespace PadV3

#endif
