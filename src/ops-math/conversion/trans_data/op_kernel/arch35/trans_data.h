/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_data.h
 * \brief expose trans_data implementation
 */

#ifndef TRANS_DATA_HEAD_FILE__H_
#define TRANS_DATA_HEAD_FILE__H_

#include <type_traits>

#include "trans_data_with_simt.h"
#include "trans_data_nz2nd_with_simt.h"

using namespace AscendC;
using namespace TRSD;

#define TILING_MODE_SIMT 21000
#define TILING_MODE_SIMT_LARGE_SHAPE 21001
#define TILING_MODE_SIMT_NZ_TO_ND 21002

__aicore__ void inline trans_data_impl(GM_ADDR src, GM_ADDR dst, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_SRC);
    using DTYPE_SRC_ = std::conditional_t<
        tSize != b32,
        std::conditional_t<
            tSize == b8, uint8_t,
            std::conditional_t<tSize == b16, uint16_t, std::conditional_t<tSize == b64, uint64_t, DTYPE_SRC>>>,
        DTYPE_SRC>;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(TILING_MODE_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(TransDataASCTilingData, tilingData, tiling);
        TransWithSIMT<DTYPE_SRC_> op;
        op.Init(src, dst, &tilingData);
        op.Process<uint32_t>();
    } else if (TILING_KEY_IS(TILING_MODE_SIMT_LARGE_SHAPE)) {
        GET_TILING_DATA_WITH_STRUCT(TransDataASCTilingData, tilingData, tiling);
        TransWithSIMT<DTYPE_SRC_> op;
        op.Init(src, dst, &tilingData);
        op.Process<uint64_t>();
    } else if (TILING_KEY_IS(TILING_MODE_SIMT_NZ_TO_ND)) {
        GET_TILING_DATA_WITH_STRUCT(TransDataNzToNdTilingData, tilingData, tiling);
        TransNZ2NDWithSIMT<DTYPE_SRC_> op;
        op.Init(src, dst, &tilingData);
        op.Process<uint64_t>();
    }
}

#endif  // TRANS_DATA_HEAD_FILE__H_
