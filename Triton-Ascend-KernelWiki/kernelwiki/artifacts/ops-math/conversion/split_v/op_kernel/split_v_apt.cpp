/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/split_v_ub_split.h"
#include "arch35/split_v_pure_copy.h"
#include "arch35/split_v_pure_copy_special.h"
#include "arch35/split_v_pure_copy_same_len.h"
#include "arch35/split_v_ub_split_same_len.h"
#include "arch35/split_v_ub_split_same_len_small_g.h"
#include "arch35/split_v_ub_split_same_len_deinterleave.h"
#include "arch35/split_v_simt.h"
#include "arch35/split_v_simt_same_len.h"
#include "arch35/split_v_simt_split_in_tensor.h"
// #include "arch35/split_v_simt_chunk.h"
// #include "arch35/split_v_simt_chunk_ub.h"
#include "arch35/split_v_simt_chunk_pre.h"
#include "arch35/split_v_simt_same_len_split_in_tensor.h"
#include "kernel_operator.h"

#define TILING_KEY_PURE_MOVE_SAME_LEN 100
#define TILING_KEY_UB_SPLIT_SAME_LEN 101
#define TILING_KEY_PURE_COPY 102
#define TILING_KEY_UB_SPLIT 103
#define TILING_KEY_PURE_COPY_SPECIAL 104
#define TILING_KEY_UB_SPLIT_SAME_LEN_SMALL_G 111
#define TILING_KEY_UB_SPLIT_SAME_LEN_DEINTERLEAVE 112
#define TILING_KEY_SIMT 200
#define TILING_KEY_SIMT_SAME_LEN 201
#define TILING_KEY_SIMT_SPLIT_IN_TENSOR 202
#define TILING_KEY_SIMT_SAME_LEN_SPLIT_IN_TENSOR 203
#define TILING_KEY_SIMT_CHUNK 204

using namespace AscendC;
using namespace Ops::Base;

extern "C" __global__ __aicore__ void split_v(GM_ADDR x, GM_ADDR sizeSplits, GM_ADDR splitDim, GM_ADDR y,
                                              GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(TILING_KEY_UB_SPLIT)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplit<uint8_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplit<uint16_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplit<uint32_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplit<uint64_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_PURE_COPY)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVPureCopyMode<uint8_t, DTYPE_SIZE_SPLITS> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVPureCopyMode<uint16_t, DTYPE_SIZE_SPLITS> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVPureCopyMode<uint32_t, DTYPE_SIZE_SPLITS> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVPureCopyMode<uint64_t, DTYPE_SIZE_SPLITS> op(pipe);
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_PURE_COPY_SPECIAL)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVPureCopySpecialMode<uint8_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVPureCopySpecialMode<uint16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVPureCopySpecialMode<uint32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVPureCopySpecialMode<uint64_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_PURE_MOVE_SAME_LEN)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint8_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVPureCopyModeSameLen<uint64_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLen<uint8_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLen<uint16_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLen<uint32_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLen<uint64_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN_SMALL_G)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint8_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint16_t, uint16_t, int16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint32_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLenSmallG<uint64_t, uint32_t, int32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_UB_SPLIT_SAME_LEN_DEINTERLEAVE)) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint8_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint16_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint32_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVUbSplitSameLenDeinterleave<uint64_t> op(pipe);
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(SplitVSIMTTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVSIMT<uint8_t, DTYPE_SIZE_SPLITS> op;
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVSIMT<uint16_t, DTYPE_SIZE_SPLITS> op;
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVSIMT<uint32_t, DTYPE_SIZE_SPLITS> op;
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVSIMT<uint64_t, DTYPE_SIZE_SPLITS> op;
            op.Init(x, y, sizeSplits, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_SAME_LEN)) {
        GET_TILING_DATA_WITH_STRUCT(SplitSIMTTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVSIMTSameLen<uint8_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVSIMTSameLen<uint16_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVSIMTSameLen<uint32_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVSIMTSameLen<uint64_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_SPLIT_IN_TENSOR)) {
        GET_TILING_DATA_WITH_STRUCT(SplitVSIMTTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVSIMTInTensor<uint8_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVSIMTInTensor<uint16_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVSIMTInTensor<uint32_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVSIMTInTensor<uint64_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_CHUNK)) {
        GET_TILING_DATA_WITH_STRUCT(SplitVSIMTChunkPreTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVSIMTChunkPre<uint8_t> op;
            op.Init(x, y, workSpace, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVSIMTChunkPre<uint16_t> op;
            op.Init(x, y, workSpace, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVSIMTChunkPre<uint32_t> op;
            op.Init(x, y, workSpace, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVSIMTChunkPre<uint64_t> op;
            op.Init(x, y, workSpace, &tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_SAME_LEN_SPLIT_IN_TENSOR)) {
        GET_TILING_DATA_WITH_STRUCT(SplitSIMTTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
            SplitV::SplitVSIMTSameLenSplitTensor<uint8_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
            SplitV::SplitVSIMTSameLenSplitTensor<uint16_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            SplitV::SplitVSIMTSameLenSplitTensor<uint32_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
        if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            SplitV::SplitVSIMTSameLenSplitTensor<uint64_t> op;
            op.Init(x, y, &tilingData);
            op.Process();
        }
    }
}
