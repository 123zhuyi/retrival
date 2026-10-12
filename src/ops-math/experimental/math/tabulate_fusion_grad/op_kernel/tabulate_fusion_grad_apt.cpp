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
 * \file tabulate_fusion_grad_apt.cpp
 * \brief Kernel entry for tabulate_fusion_grad operator
 *
 * 单模板参数:
 *   schMode (uint32_t): 场景模式 (TABULATE_FUSION_GRAD_MODE_DEFAULT = 0)
 * Dtype 由 DTYPE_TABLE 宏自动实例化 (float32).
 * 输入顺序: table, table_info, em_x, em, dy, descriptor
 * 输出顺序: dy_dem_x, dy_dem
 */

#include "arch35/tabulate_fusion_grad_simt.h"

enum class TabulateFusionGradTilingKey : uint32_t {
    TILING_KEY_DEFAULT = 0,
};

template <uint32_t schMode>
__global__ __aicore__ void tabulate_fusion_grad(GM_ADDR table, GM_ADDR table_info, GM_ADDR em_x, GM_ADDR em, GM_ADDR dy,
                                                GM_ADDR descriptor, GM_ADDR dy_dem_x, GM_ADDR dy_dem, GM_ADDR workspace,
                                                GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TabulateFusionGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(TabulateFusionGradTilingData, tilingData, tiling);
    (void)tilingData; // apt.cpp 仅注册，Process 内部自行 GET_TILING_DATA_WITH_STRUCT 解析

    if constexpr (schMode == static_cast<uint32_t>(TabulateFusionGradTilingKey::TILING_KEY_DEFAULT)) {
        // DTYPE_TABLE 宏由 _def.cpp 中 table 输入的 DataType 配置自动实例化为 float
        NsTabulateFusionGrad::Process<DTYPE_TABLE>(table, table_info, em_x, em, dy, descriptor, dy_dem_x, dy_dem,
                                                   workspace, tiling);
    }
}
