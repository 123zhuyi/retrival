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
#include "matrix_set_diag_v2_tilingdata.h"
#include "matrix_set_diag_v2_tilingkey.h"
#include "matrix_set_diag_v2_no_cut_tail.h"
#include "matrix_set_diag_v2_cut_tail.h"
#include "matrix_set_diag_v1_cut_tail.h"

using namespace AscendC;
using namespace MSD;

static constexpr auto X_DTYPE_SIZE_ = sizeof(DTYPE_INPUT);
using DTYPE_INPUT_ = std::conditional_t<
    X_DTYPE_SIZE_ == sizeof(uint8_t), uint8_t,
    std::conditional_t<X_DTYPE_SIZE_ == sizeof(uint16_t), uint16_t,
                       std::conditional_t<X_DTYPE_SIZE_ == sizeof(uint32_t), uint32_t,
                                          std::conditional_t<X_DTYPE_SIZE_ == sizeof(uint64_t), uint64_t, void>>>>;
static constexpr bool INVALID_DTYPE_ = std::is_same_v<DTYPE_INPUT_, void>;

__aicore__ inline void MSDV1CutTail(GM_ADDR x, GM_ADDR diagonal, GM_ADDR k, GM_ADDR y, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MatrixSetDiagTilingData, tilingData, tiling);

    MSD::MatrixSetDiagCutTailScatter<DTYPE_INPUT_> op(&pipe);
    op.Init(x, diagonal, y, &tilingData);
    op.Process();
}

template <bool IsBigShape>
__aicore__ inline void MSDV2CutTail(GM_ADDR x, GM_ADDR diagonal, GM_ADDR k, GM_ADDR y, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MSDV2CutTailTilingData, tilingData, tiling);
    if constexpr (IsBigShape == false) {
        MSD::MatrixSetDiagCutTail<DTYPE_INPUT_, uint32_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    } else if constexpr (IsBigShape == true) {
        MSD::MatrixSetDiagCutTail<DTYPE_INPUT_, uint64_t> op(&pipe);
        op.Init(x, diagonal, y, &tilingData);
        op.Process();
    }
}

template <uint8_t Way, bool IsVLFullLoad>
__aicore__ inline void MSDNoCutTail(GM_ADDR x, GM_ADDR diagonal, GM_ADDR k, GM_ADDR y, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(MSDV2NoCutTailTilingData, tilingData, tiling);

    MSD::MatrixSetDiagNoCutWV2<DTYPE_INPUT_, Way, IsVLFullLoad> op(&pipe);
    op.Init(x, diagonal, y, &tilingData);
    op.Process();
}

template <uint8_t Way, bool IsVLFullLoad, bool IsBigShape, bool IsCutTail>
__global__ __aicore__ void matrix_set_diag_v2(GM_ADDR x, GM_ADDR diagonal, GM_ADDR k, GM_ADDR y, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if constexpr (INVALID_DTYPE_) {
        return;
    }
    if constexpr (Way == TPL_WAY_V1) {
        MSDV1CutTail(x, diagonal, k, y, workspace, tiling);
        return;
    }
    if constexpr (IsCutTail) {
        MSDV2CutTail<IsBigShape>(x, diagonal, k, y, workspace, tiling);
        return;
    }
    MSDNoCutTail<Way, IsVLFullLoad>(x, diagonal, k, y, workspace, tiling);
    return;
}
