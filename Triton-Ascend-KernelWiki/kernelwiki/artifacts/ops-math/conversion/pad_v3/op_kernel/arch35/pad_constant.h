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
 * \file pad_common.h
 * \brief pad_common
 */

#ifndef ASCENDC_PAD_CONSTANT_H
#define ASCENDC_PAD_CONSTANT_H
#include "pad_huge_width.h"
#include "pad_normal_width.h"
#include "pad_gather.h"
#include "pad_scatter.h"
#include "pad_simt_huge.h"
#include "pad_simt.h"

namespace PadV3 {
template <typename T>
__aicore__ inline void LaunchKernelPadWithHugeWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                                    GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadWithHugeWidth<uint8_t, T> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadWithHugeWidth<uint16_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadWithHugeWidth<uint32_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadWithHugeWidth<uint64_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadWithNormalWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                                      GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadWithNormalWidth<uint8_t, KEY, T> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadWithNormalWidth<uint16_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadWithNormalWidth<uint32_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadWithNormalWidth<uint64_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y, constValue);
        op.Process();
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadGather(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                             GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadGather<uint8_t, T> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        PadGather<uint16_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadGather<uint32_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadGather<uint64_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadScatter(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                              GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadScatter<uint8_t, T> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        PadScatter<uint16_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadScatter<uint32_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadScatter<uint64_t> op(&pipe);
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process();
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadSimt(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                           GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadSimt<int32_t> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadSimt<int64_t> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadSimt<int8_t, T> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else {
        PadV3::PadSimt<T> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadSimtHuge(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling,
                                               GM_ADDR constValue = nullptr)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadSimtHuge<int32_t> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadSimtHuge<int64_t> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadSimtHuge<int8_t, T> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    } else {
        PadV3::PadSimtHuge<T> op;
        op.Init(x, paddings, y, &tilingData, constValue);
        op.Process(tiling);
    }
}
} // namespace PadV3

#endif
