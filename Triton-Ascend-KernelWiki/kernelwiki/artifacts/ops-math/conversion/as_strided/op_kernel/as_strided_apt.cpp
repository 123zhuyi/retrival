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
 * \file as_strided.cpp
 * \brief as_strided
 */

#include <cstdint>
#include "./arch35/as_strided_move_align.h"
#include "./arch35/as_strided_dual_cut.h"
#include "./arch35/as_strided.h"
#include "./arch35/as_strided_struct.h"
#include "./arch35/as_strided_zero_stride.h"
#include "./arch35/as_strided_gather.h"
#include "./arch35/as_strided_simt.h"

using namespace AsStrided;

#define AS_STRIDED_B8 1
#define AS_STRIDED_B16 2
#define AS_STRIDED_B32 4
#define AS_STRIDED_B64 8
#define AS_STRIDED_MOVE_ALIGN_B8 101
#define AS_STRIDED_MOVE_ALIGN_B16 102
#define AS_STRIDED_MOVE_ALIGN_B32 104
#define AS_STRIDED_MOVE_ALIGN_B64 108
#define AS_STRIDED_DUAL_CUT 200
#define ALL_STRIDEDS_ZERO_KEY 300
#define SIMT_KEY 400
#define AS_STRIDED_GATHER 500
#define EMPTY_TENSOR_KEY 1000

extern "C" __global__ __aicore__ void as_strided(GM_ADDR input, GM_ADDR outShape, GM_ADDR outStride,
    GM_ADDR storageOffset, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;
    if (TILING_KEY_IS(AS_STRIDED_B8)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStrided<int8_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B16)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStrided<half> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B32)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStrided<float> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_B64)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStrided<int64_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B8)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStridedMoveAlign<int8_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B16)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStridedMoveAlign<half> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B32)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStridedMoveAlign<float> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_MOVE_ALIGN_B64)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        KernelAsStridedMoveAlign<int64_t> op;
        op.Init(input, outShape, outStride, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(AS_STRIDED_DUAL_CUT)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            KernelAsStridedDualCut<int8_t, AS_STRIDED_DUAL_CUT> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();    
        } else {
            KernelAsStridedDualCut<DTYPE_X, AS_STRIDED_DUAL_CUT> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();    
        }
    } else if (TILING_KEY_IS(ALL_STRIDEDS_ZERO_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedZeroStrideTilingData, tilingData, tiling);
        if(sizeof(DTYPE_X) == sizeof(int8_t)) {
            StridedIsZero<int8_t> op;
            op.Init(input, output, &tilingData);
            op.Process();
        } else if(sizeof(DTYPE_X) == sizeof(half)) {
            StridedIsZero<half> op;
            op.Init(input, output, &tilingData);
            op.Process();
        } else if(sizeof(DTYPE_X) == sizeof(float)) {
            StridedIsZero<float> op;
            op.Init(input, output, &tilingData);
            op.Process();
        } else if(sizeof(DTYPE_X) == sizeof(int64_t)) {
            StridedIsZero<int64_t> op;
            op.Init(input, output, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(SIMT_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedSimtTilingData, tilingData, tiling);
        if(sizeof(DTYPE_X) == sizeof(int8_t)) {
            AsStridedSimt<int8_t> op;
            op.Init(input, output, &tilingData);
            op.Process(tiling);
        } else if(sizeof(DTYPE_X) == sizeof(half)) {
            AsStridedSimt<half> op;
            op.Init(input, output, &tilingData);
            op.Process(tiling);
        } else if(sizeof(DTYPE_X) == sizeof(float)) {
            AsStridedSimt<float> op;
            op.Init(input, output, &tilingData);
            op.Process(tiling);
        } else if(sizeof(DTYPE_X) == sizeof(int64_t)) {
            AsStridedSimt<int64_t> op;
            op.Init(input, output, &tilingData);
            op.Process(tiling);
        }
    } else if (TILING_KEY_IS(AS_STRIDED_GATHER)) {
        GET_TILING_DATA_WITH_STRUCT(AsStridedWithGatherTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            KernelAsStridedGather<uint8_t> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            KernelAsStridedGather<uint16_t> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            KernelAsStridedGather<uint32_t> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            KernelAsStridedGather<uint64_t> op;
            op.Init(input, outShape, outStride, output, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(EMPTY_TENSOR_KEY)){
        GET_TILING_DATA_WITH_STRUCT(AsStridedEmptyTilingData, tilingData, tiling);
    }
}