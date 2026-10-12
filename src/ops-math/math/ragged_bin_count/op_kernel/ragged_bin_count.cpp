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
 * \file ragged_bin_count.cpp
 * \brief Kernel entry for RaggedBinCount.
 */

#include "arch35/ragged_bin_count_tiling_key.h"
#include "arch35/ragged_bin_count_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void ragged_bin_count(GM_ADDR splits, GM_ADDR values, GM_ADDR size, GM_ADDR weights,
                                            GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RaggedBinCountTilingData);
    GET_TILING_DATA_WITH_STRUCT(RaggedBinCountTilingData, tilingData, tiling);

    // Decoded with the same constants the host encodes with (ragged_bin_count_tiling_data.h), so the
    // two sides cannot drift apart silently.
    constexpr uint32_t MAPPING_MODE = schMode >> RBC_MAPPING_MODE_SHIFT;
    constexpr bool BINARY_OUTPUT = ((schMode >> RBC_BINARY_OUTPUT_SHIFT) & RBC_HAS_WEIGHTS_MASK) != 0U;
    constexpr bool HAS_WEIGHTS = (schMode & RBC_HAS_WEIGHTS_MASK) != 0U;

    (void)size;
    AscendC::SetSysWorkspace(workspace);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    // Only the privatised path allocates from the pipe; it is constructed unconditionally because a
    // TPipe has to exist before the first InitBuffer and the choice is a runtime tiling value.
    AscendC::TPipe pipe;
    NsRaggedBinCount::Process<DTYPE_VALUES, MAPPING_MODE, BINARY_OUTPUT, HAS_WEIGHTS>(
        splits, values, weights, output, userWorkspace, &tilingData, &pipe);
}
