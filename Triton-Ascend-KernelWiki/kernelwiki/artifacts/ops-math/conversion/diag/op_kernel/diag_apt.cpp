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
 * \file diag_apt.cpp
 * \brief Diag operator kernel entry point
 */

#include <type_traits>
#include "arch35/diag_simt.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void diag(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    constexpr auto b8 = sizeof(uint8_t);
    constexpr auto b16 = sizeof(uint16_t);
    constexpr auto b32 = sizeof(uint32_t);
    constexpr auto b64 = sizeof(uint64_t);
    constexpr auto tSize = sizeof(DTYPE_X);
    using DTYPE_X_ =
        std::conditional_t<tSize != b32,
                           std::conditional_t<tSize == b8, uint8_t,
                                              std::conditional_t<tSize == b16, uint16_t,
                                                                 std::conditional_t<tSize == b64, uint64_t, DTYPE_X>>>,
                           uint32_t>;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DiagSimtTilingData);

    SetSysWorkspace(workspace);

    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_SIMT)) {
        GET_TILING_DATA_WITH_STRUCT(DiagSimtTilingData, tilingData, tiling);
        DiagSIMT<DTYPE_X_> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
