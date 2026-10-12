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
 * \file as_strided.cpp
 * \brief
 */
#include "as_strided.h"
#include "as_strided_tiling_data.h"
#include "as_strided_tiling_key.h"

using namespace NsAsStrided;

namespace {
#if defined(ORIG_DTYPE_X)
#if ORIG_DTYPE_X == DT_BOOL || ORIG_DTYPE_X == DT_INT8 || ORIG_DTYPE_X == DT_UINT8 || ORIG_DTYPE_X == DT_HIFLOAT8 || \
    ORIG_DTYPE_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_X == DT_FLOAT8_E4M3FN
using AsStridedStorageType = int8_t;
#elif ORIG_DTYPE_X == DT_FLOAT16 || ORIG_DTYPE_X == DT_BF16 || ORIG_DTYPE_X == DT_INT16 || ORIG_DTYPE_X == DT_UINT16
using AsStridedStorageType = int16_t;
#elif ORIG_DTYPE_X == DT_FLOAT || ORIG_DTYPE_X == DT_INT32 || ORIG_DTYPE_X == DT_UINT32 || ORIG_DTYPE_X == DT_COMPLEX32
using AsStridedStorageType = int32_t;
#elif ORIG_DTYPE_X == DT_DOUBLE || ORIG_DTYPE_X == DT_INT64 || ORIG_DTYPE_X == DT_UINT64 || ORIG_DTYPE_X == DT_COMPLEX64
using AsStridedStorageType = int64_t;
#else
using AsStridedStorageType = DTYPE_X;
#endif
#else
using AsStridedStorageType = float;
#endif

#if defined(ORIG_DTYPE_SIZE)
#if ORIG_DTYPE_SIZE == DT_INT32
using AsStridedIndexType = int32_t;
#else
using AsStridedIndexType = int64_t;
#endif
#elif defined(DTYPE_SIZE)
#if DTYPE_SIZE == DT_INT32
using AsStridedIndexType = int32_t;
#else
using AsStridedIndexType = int64_t;
#endif
#else
using AsStridedIndexType = int64_t;
#endif
} // namespace

extern "C" __global__ __aicore__ void as_strided(GM_ADDR x, GM_ADDR size, GM_ADDR stride, GM_ADDR storageOffset,
                                                 GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AsStridedTilingData);
    GET_TILING_DATA_WITH_STRUCT(AsStridedTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    AsStridedKernel<AsStridedStorageType, AsStridedIndexType> op;
    op.Init(x, size, stride, storageOffset, y, &tilingData, &pipe);
    op.Process();
}
