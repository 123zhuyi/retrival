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
 * \file pad_edge.h
 * \brief pad_edge
 */
#ifndef ASCENDC_PAD_EDGE_H
#define ASCENDC_PAD_EDGE_H

#include "pad_edge_huge_width.h"
#include "pad_edge_simt.h"
#include "pad_edge_simt_huge.h"
#include "pad_edge_gather.h"
#include "pad_edge_normal_w.h"

namespace PadV3 {
template <typename T>
__aicore__ inline void LaunchKernelPadEdgeSimt(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadEdgeSimt<int32_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadEdgeSimt<int64_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadEdgeSimt<int8_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadEdgeSimt<T> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadEdgeSimtHuge(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadV3::PadEdgeSimtHuge<int32_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadV3::PadEdgeSimtHuge<int64_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadV3::PadEdgeSimtHuge<int8_t> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    } else {
        PadV3::PadEdgeSimtHuge<T> op;
        op.Init(x, paddings, y, &tilingData);
        op.Process(tiling);
    }
}

template <typename T>
__aicore__ inline void LaunchKernelPadEdgeWithHugeWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadEdgeWithHugeWidth<uint8_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadEdgeWithHugeWidth<uint16_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadEdgeWithHugeWidth<uint32_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadEdgeWithHugeWidth<uint64_t> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadEdgeWithNormalWidth(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        KernelPadReplWithNormalWidth<uint8_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        KernelPadReplWithNormalWidth<uint16_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        KernelPadReplWithNormalWidth<uint32_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        KernelPadReplWithNormalWidth<uint64_t, KEY> op(&pipe, &tilingData);
        op.Init(x, paddings, y);
        op.Process();
    }
}

template <typename T, int32_t KEY>
__aicore__ inline void LaunchKernelPadEdgeGather(GM_ADDR x, GM_ADDR paddings, GM_ADDR y, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(PadACTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        PadEdgeGather<uint8_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        PadEdgeGather<uint16_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        PadEdgeGather<uint32_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        PadEdgeGather<uint64_t, KEY> op(&pipe);
        op.Init(x, paddings, y, &tilingData);
        op.Process();
    }
}

} // namespace PadV3

#endif
