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
 * \file matrix_set_diag_apt.cpp
 * \brief MatrixSetDiag implementation
 */

#include <type_traits>
#include "arch35/matrix_set_diag_tilingdata.h"
#include "arch35/matrix_set_diag_tilingkey.h"
#include "arch35/matrix_set_diag_simt.h"
#include "arch35/matrix_set_diag_no_cutw.h"
#include "arch35/matrix_set_diag_cutw.h"

using namespace AscendC;
using namespace MSD;

template <typename T>
__aicore__ inline void MSDNoCutW(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatrixSetDiagTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        MSD::MatrixSetDiagNoCutWScatter<uint8_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        MSD::MatrixSetDiagNoCutWScatter<uint16_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        MSD::MatrixSetDiagNoCutWScatter<uint32_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        MSD::MatrixSetDiagNoCutWScatter<uint64_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    }
}

template <typename T>
__aicore__ inline void MSDSimt(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MatrixSetDiagTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        MSD::MatrixSetDiagSimt<uint8_t> op;
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        MSD::MatrixSetDiagSimt<uint16_t> op;
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        MSD::MatrixSetDiagSimt<uint32_t> op;
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        MSD::MatrixSetDiagSimt<uint64_t> op;
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    }
}

template <typename T>
__aicore__ inline void MSDCutW(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatrixSetDiagTilingData, tilingData, tiling);
    if constexpr (sizeof(T) == sizeof(int8_t)) {
        MSD::MatrixSetDiagCutWScatter<uint8_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int16_t)) {
        MSD::MatrixSetDiagCutWScatter<uint16_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int32_t)) {
        MSD::MatrixSetDiagCutWScatter<uint32_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (sizeof(T) == sizeof(int64_t)) {
        MSD::MatrixSetDiagCutWScatter<uint64_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    }
}

template <bool IsCutW, bool IsSIMT>
__global__ __aicore__ void matrix_set_diag(GM_ADDR x, GM_ADDR diagonal, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MatrixSetDiagTilingData);

    if constexpr (IsSIMT) {
        MSDSimt<DTYPE_X>(x, diagonal, y, workspace, tiling);
        return;
    }
    if constexpr (IsCutW) {
        MSDCutW<DTYPE_X>(x, diagonal, y, workspace, tiling);
        return;
    } else {
        MSDNoCutW<DTYPE_X>(x, diagonal, y, workspace, tiling);
        return;
    }
}
