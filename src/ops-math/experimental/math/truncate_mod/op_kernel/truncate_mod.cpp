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
 * \file truncate_mod.cpp
 * \brief TruncateMod kernel entry. schMode selects the compute dtype.
 */

#include "truncate_mod.h"

#if defined(ASCENDC_CPU_DEBUG)
#include <cstring>
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(structType, name, tilingArg) \
    structType name;                                             \
    (void)memcpy(&(name), (tilingArg), sizeof(structType))
#endif
#else
#include "truncate_mod_tiling_key.h"
#endif

template <uint32_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateModTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_0) {
        NsTruncateMod::Run<half>(x1, x2, y, usrWorkspace, &tilingData);
    } else if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_1) {
        NsTruncateMod::Run<float>(x1, x2, y, usrWorkspace, &tilingData);
    } else if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_3) {
        NsTruncateMod::Run<int32_t>(x1, x2, y, usrWorkspace, &tilingData);
    } else if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_4) {
        NsTruncateMod::Run<int8_t>(x1, x2, y, usrWorkspace, &tilingData);
    } else if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_5) {
        NsTruncateMod::Run<uint8_t>(x1, x2, y, usrWorkspace, &tilingData);
    }
#if __CCE_AICORE__ >= 220
    else if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_2) {
        NsTruncateMod::Run<bfloat16_t>(x1, x2, y, usrWorkspace, &tilingData);
    }
#endif
}
