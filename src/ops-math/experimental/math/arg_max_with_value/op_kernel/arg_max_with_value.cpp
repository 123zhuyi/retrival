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
 * \file arg_max_with_value.cpp
 * \brief ArgMaxWithValue (A2) kernel entry. The host SetTilingKey picks schMode; this dispatches at
 *        compile time to one pattern (COPY / LAST / NLAST) so each schMode is its own binary with no
 *        runtime branch. IS_MIN fixes the reduction direction for this entry.
 */
#include "arg_max_with_value.h"

template <uint32_t schMode, bool gather>
__global__ __aicore__ void arg_max_with_value(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                              GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArgMaxWithValueTilingData);
    GET_TILING_DATA_PTR_WITH_STRUCT(ArgMaxWithValueTilingData, tilingData, tiling);
    ArgWithValueNs::RunArgWithValue<DTYPE_X, false, schMode, gather>(x, indice, values, workspace, tilingData);
}

// Non-split paths use only fixed UB addresses and direct GM output.  They do not consume the
// framework's system workspace, so avoid the wrapper's redundant global-pointer initialization.
// Keep the three cross-core split keys on the real implementation: those paths publish partials
// through the user workspace and must receive the translated workspace address.
#if defined(TILING_KEY_VAR) && (TILING_KEY_VAR != ARG_SCH_LAST_SPLIT1 && TILING_KEY_VAR != ARG_SCH_LAST_SPLIT2 && \
                                TILING_KEY_VAR != ARG_SCH_NLAST_SPLIT)
namespace AscendC {
__aicore__ inline void SkipUnusedSysWorkspace(GM_ADDR) {}
} // namespace AscendC
#define SetSysWorkspaceForce SkipUnusedSysWorkspace
#endif
