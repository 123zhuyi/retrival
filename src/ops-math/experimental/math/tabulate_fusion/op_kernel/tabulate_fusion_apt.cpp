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
 * \file tabulate_fusion_apt.cpp
 * \brief Kernel entry for tabulate_fusion operator
 *
 * Single template parameter:
 *   schMode (uint32_t): scene mode (TABULATE_FUSION_MODE_DEFAULT = 0)
 * Dtype is handled by DTYPE_TABLE macro (float32 / float16 auto-instantiated).
 */

#include "arch35/tabulate_fusion_simt.h"

enum class TabulateFusionTilingKey : uint32_t {
    TILING_KEY_DEFAULT = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void tabulate_fusion(GM_ADDR table, GM_ADDR table_info, GM_ADDR em_x, GM_ADDR em,
                                           GM_ADDR descriptor, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TabulateFusionTilingData);
    GET_TILING_DATA_WITH_STRUCT(TabulateFusionTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(TabulateFusionTilingKey::TILING_KEY_DEFAULT)) {
        // DTYPE_TABLE macro auto-instantiates float / half per def.cpp DataType config
        NsTabulateFusion::Process<DTYPE_TABLE>(table, table_info, em_x, em, descriptor, &tilingData);
    }
}
