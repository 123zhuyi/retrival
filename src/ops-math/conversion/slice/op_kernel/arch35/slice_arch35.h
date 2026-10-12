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
 * \file slice_arch35.h
 * \brief
 */

#ifndef SLICE_ARCH35_H
#define SLICE_ARCH35_H

#include "slice_nddma.h"
#include "slice_nddma_last_dim.h"
#include "slice_move_align_two_dim.h"
#include "slice_move_align_last_dim.h"
#include "slice_move_align.h"
#include "slice_move_align_gather.h"
#include "slice_move_align_unalign_datcopy.h"
#include "slice_two_dim_small_shape.h"


using namespace Slice;

extern "C" __aicore__ inline void SliceMoveAlignProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceMoveAlignTilingData* tilingData, TPipe* pipe)
{
    // ub不切最后一根轴
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceMoveAlign<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceMoveAlign<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceMoveAlign<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceMoveAlign<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceMoveAlignLastDimProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceMoveAlignLastDimTilingData* tilingData, TPipe* pipe)
{
    // ub切最后一根轴 or 纯搬运模板
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceMoveAlignLastDim<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceMoveAlignLastDim<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceMoveAlignLastDim<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceMoveAlignLastDim<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceMoveAlignTwoDimProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceMoveAlignLast2DimTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceMoveAlignTwoDim<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Slice::SliceMoveAlignTwoDim<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceMoveAlignTwoDim<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceMoveAlignTwoDim<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceNDDMAProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceNDDMATilingData* tilingData, TPipe* pipe)
{
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceNDDMA<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceNDDMA<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceNDDMA<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceNDDMA<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceNDDMALastDimProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceNDDMALastDimTilingData* tilingData, TPipe* pipe)
{
    // ub切最后一根轴
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceNDDMALastDim<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceNDDMALastDim<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceNDDMALastDim<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceNDDMALastDim<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, size, nullptr, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceMoveAlignGatherProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceMoveAlignGatherTilingData* tilingData, TPipe* pipe)
{
    // ub切最后一根轴
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceMoveAlignGather<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceMoveAlignGather<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceMoveAlignGather<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceMoveAlignGather<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceMoveAlignDataCopyUnalignProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceMoveAlignGatherTilingData* tilingData, TPipe* pipe)
{
    // ub切最后一根轴
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceMoveAlignDataCopyUnalign<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceMoveAlignDataCopyUnalign<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceMoveAlignDataCopyUnalign<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceMoveAlignDataCopyUnalign<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    }
}

extern "C" __aicore__ inline void SliceTwoDimSmallShapeProcess(
    GM_ADDR x, GM_ADDR offsets, GM_ADDR size, GM_ADDR y, const SliceTwoDimSmallSapeTilingData* tilingData, TPipe* pipe)
{
    // complex32/uint32/int32/float32
    if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Slice::SliceTwoDimSmallShape<int32_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        // complex64/uint64/int64
        Slice::SliceTwoDimSmallShape<int64_t, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Slice::SliceTwoDimSmallShape<int8_t, DTYPE_OFFSETS, DTYPE_X> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    } else {
        Slice::SliceTwoDimSmallShape<DTYPE_X, DTYPE_OFFSETS> op;
        op.Init(x, offsets, y, tilingData, pipe);
        op.Process();
    }
}


#endif // SLICE_ARCH35_H