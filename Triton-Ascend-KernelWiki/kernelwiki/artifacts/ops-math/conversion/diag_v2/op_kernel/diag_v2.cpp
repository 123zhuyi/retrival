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
 * \file diag_v2.cpp
 * \brief
 */
#include "diag_v2_b8.h"
#include "diag_v2_b16.h"
#include "diag_v2_b32.h"
#include "diag_v2_b64.h"
#include "diag_v2_b128.h"
#include "diag_v2_t4.h"
#ifdef __CCE_UT_TEST__
#include "../../../conversion/diag_flat/op_kernel/diag_flat_nd_to_2d.h"
#include "../../../conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_with_few.h"
#include "../../../conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_b16_more64.h"
#include "../../../conversion/diag_flat/op_kernel/diag_flat_nd_to_2d_b16_less.h"
#else
#include "../diag_flat/diag_flat_nd_to_2d.h"
#include "../diag_flat/diag_flat_nd_to_2d_with_few.h"
#include "../diag_flat/diag_flat_nd_to_2d_b16_more64.h"
#include "../diag_flat/diag_flat_nd_to_2d_b16_less.h"
#endif

using namespace DiagV2;
using namespace DiagFlat;

extern "C" __global__ __aicore__ void diag_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2101)) {
        DiagV2::DiagV2B8<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2102)) {
        DiagV2::DiagV2B16<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2103)) {
        DiagV2::DiagV2B32<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2104)) {
        DiagV2::DiagV2B64<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2105)) {
        DiagV2::DiagV2B128<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2401)) {
        DiagV2::DiagV2T4<int8_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2402)) {
        DiagV2::DiagV2T4<int16_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2403)) {
        DiagV2::DiagV2T4<int32_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2404)) {
        DiagV2::DiagV2T4<int64_t> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        // dtype is complex32/uint32/int32/float32
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int32_t> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is complex64/uint64/int64/float64
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int64_t> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is int16/uin16/float16
        } else {
            DiagFlat::DiagFlatNDTo2DWithFew<DTYPE_X> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
        }
        // dtype is complex128, input number is less than 64
    } else if (TILING_KEY_IS(102)) {
        DiagFlat::DiagFlatND2To2DB16Less64<int64_t> op(&pipe);
        op.Init(x, y, workspace, &tilingData);
        op.Process();
        // input number more than 64
    } else if (TILING_KEY_IS(103)) {
        // dtype is complex32/uint32/int32/float32
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagFlat::DiagFlatNDTo2D<int32_t> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is complex64/uint64/int64/float64
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagFlat::DiagFlatNDTo2D<int64_t> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
            // dtype is int16/uin16/float16
        } else {
            DiagFlat::DiagFlatNDTo2D<DTYPE_X> op(&pipe);
            op.Init(x, y, workspace, &tilingData);
            op.Process();
        }
        // input number more than 64, input type is complex128
    } else if (TILING_KEY_IS(104)) {
        DiagFlat::DiagFlatND2To2DB16More64<int64_t> op(&pipe);
        op.Init(x, y, workspace, &tilingData);
        op.Process();
    }
}
