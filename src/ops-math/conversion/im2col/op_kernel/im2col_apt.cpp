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
 * \file im2col_apt.cpp
 * \brief Im2col implementation
 */

#include <type_traits>
#include "arch35/im2col_tilingdata.h"
#include "arch35/im2col_tilingkey.h"
#include "arch35/im2col_simt_NCHW.h"
#include "arch35/im2col_simt_NHWC.h"
#include "arch35/im2col_gather_cut_hw.h"
#include "arch35/im2col_gather_cut_nc.h"
#include "arch35/im2col_norm_NHWC.h"

using namespace AscendC;
using namespace Im2col;
using namespace Im2ColAsc;

template <uint8_t format, bool isBigShape>
__aicore__ inline void Im2colSimt(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_X);
    using DTYPE_X_ = std::conditional_t<
        tSize != b32,
        std::conditional_t<
            tSize == b8, uint8_t,
            std::conditional_t<tSize == b16, uint16_t, std::conditional_t<tSize == b64, uint64_t, DTYPE_X>>>,
        uint32_t>;

    SetSysWorkspace(workspace);

    TPipe pipe;
    if constexpr (format == TPL_FORMAT_NCHW && isBigShape == false) {
        GET_TILING_DATA_WITH_STRUCT(Im2ColSIMTTilingData, tilingData, tiling);
        Im2ColSIMT_NCHW<DTYPE_X_, uint32_t> op;
        op.Init(x, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (format == TPL_FORMAT_NCHW && isBigShape == true) {
        GET_TILING_DATA_WITH_STRUCT(Im2ColSIMTTilingData, tilingData, tiling);
        Im2ColSIMT_NCHW<DTYPE_X_, uint64_t> op;
        op.Init(x, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (format == TPL_FORMAT_NHWC && isBigShape == false) {
        GET_TILING_DATA_WITH_STRUCT(Im2ColSIMTTilingData, tilingData, tiling);
        Im2ColSIMT_NHWC<DTYPE_X_, uint32_t> op;
        op.Init(x, y, &tilingData);
        op.Process(tiling);
    } else if constexpr (format == TPL_FORMAT_NHWC && isBigShape == true) {
        GET_TILING_DATA_WITH_STRUCT(Im2ColSIMTTilingData, tilingData, tiling);
        Im2ColSIMT_NHWC<DTYPE_X_, uint64_t> op;
        op.Init(x, y, &tilingData);
        op.Process(tiling);
    }
}

template <bool isPadding>
__aicore__ inline void Im2colNCHWCutNC(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(Im2ColNCHWTilingData, tilingData, tiling);
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Im2ColGatherCutNc<uint8_t, isPadding> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Im2ColGatherCutNc<uint16_t, isPadding> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Im2ColGatherCutNc<uint32_t, isPadding> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Im2ColGatherCutNc<uint64_t, isPadding> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

template <bool isPadding>
__aicore__ inline void Im2colNCHWCutHW(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(Im2ColNCHWTilingData, tilingData, tiling);
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        Im2colGatherCutHw<uint8_t, uint16_t, int16_t, isPadding> op(pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        Im2colGatherCutHw<uint16_t, uint16_t, int16_t, isPadding> op(pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        Im2colGatherCutHw<uint32_t, uint32_t, int32_t, isPadding> op(pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        Im2colGatherCutHw<uint64_t, uint64_t, int64_t, isPadding> op(pipe);
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

template <bool isPadding, uint8_t ubAxis>
__aicore__ inline void Im2colNHWC(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;

    GET_TILING_DATA_WITH_STRUCT(Im2ColNHWCTilingData, tilingData, tiling);
    if constexpr (sizeof(DTYPE_X) == sizeof(int8_t)) {
        KernelIm2ColNormNhwc<uint8_t, isPadding, ubAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
        KernelIm2ColNormNhwc<uint16_t, isPadding, ubAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
        KernelIm2ColNormNhwc<uint32_t, isPadding, ubAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
        KernelIm2ColNormNhwc<uint64_t, isPadding, ubAxis> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

template <uint8_t format, uint8_t ubAxis, bool isPadding, bool isSIMT, bool IsBigShape>
__global__ __aicore__ void im2col(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(Im2ColNCHWTilingData);

    if constexpr (isSIMT) {
        Im2colSimt<format, IsBigShape>(x, y, workspace, tiling);
        return;
    }
    if constexpr (format == TPL_FORMAT_NCHW && ubAxis == TPL_UB_AXIS_NCHW_NC) {
        Im2colNCHWCutNC<isPadding>(x, y, workspace, tiling);
        return;
    }
    if constexpr (format == TPL_FORMAT_NCHW && ubAxis == TPL_UB_AXIS_NCHW_HW) {
        Im2colNCHWCutHW<isPadding>(x, y, workspace, tiling);
        return;
    }
    if constexpr (format == TPL_FORMAT_NHWC) {
        Im2colNHWC<isPadding, ubAxis>(x, y, workspace, tiling);
        return;
    }
}
