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
 * \file sort_with_index.cpp
 * \brief SortWithIndex kernel entry (ascend910b, DAV_2201).
 *
 * 3-D TilingKey dispatch:
 *   - VALUE_DT = DTYPE_X     (value x/y dtype, compile-time macro from op_host DataType list)
 *   - INDEX_DT = DTYPE_INDEX (index dtype, compile-time macro)
 *   - SIZE_MODE = schMode    (ASCENDC_TPL key: 0 SINGLE_TILE / 1 MRGSORT / 2 EMPTY)
 *
 * Iteration-2 implements all branches: VALUE_DT in {half,float,bf16,int32} x INDEX_DT in {int32,int64}
 * x SIZE_MODE in {0 SINGLE_TILE, 1 MRGSORT, 2 EMPTY}. The schMode TPL key compiles all three SIZE_MODE
 * variants per dtype pair; the value/index path is selected inside the template via if constexpr.
 */

#include "sort_with_index_kernel.h"

// NOTE: the kernel entry function name MUST equal the op type in snake_case (= sort_with_index),
// which the build passes as --main_func. opFile.value (= sort_with_index) sets the FILE name AND the
// per-op kernel binary config file name (kernel/config/<soc>/sort_with_index.json) that the runtime
// looks up by op interface name at launch; it must therefore match the op snake-case name, otherwise
// ParseDynamicKernelConfig fails with errno 561108 ("does not has any binary").
template <uint32_t schMode>
__global__ __aicore__ void sort_with_index(
    GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sorted_index, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SortWithIndexTilingData);
    GET_TILING_DATA_WITH_STRUCT(SortWithIndexTilingData, tilingData, tiling);
    NsSortWithIndex::SortWithIndex<DTYPE_X, DTYPE_INDEX, schMode> op;
    op.Init(x, index, y, sorted_index, workspace, &tilingData);
    op.Process();
}
