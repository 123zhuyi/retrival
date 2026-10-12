/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file data_compare_apt.cpp
 * \brief DataCompare 算子 kernel 入口分发（APT entry）
 *
 * All Reduce 简化：2 个 bool（templateType / isEmptyTensor），无 isTailR。
 * dtype 走 DTYPE_X1 编译期实例化（6 种 dtype），× 3 组合 = 18 份 binary。
 */
#include "arch35/data_compare.h"
#include "arch35/data_compare_empty.h"

template <bool templateType, bool isEmptyTensor>
__global__ __aicore__ void data_compare(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DataCompareTilingData);
    GET_TILING_DATA_WITH_STRUCT(DataCompareTilingData, tilingData, tiling);

    if constexpr (isEmptyTensor) {
        NsDataCompare::DataCompareEmpty<DTYPE_X1> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    } else if constexpr (templateType) {
        // Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2
        NsDataCompare::DataCompareKernel<DTYPE_X1> op;
        op.InitGroup(x1, x2, y, workspace, &tilingData);
        op.ProcessGroup();
    } else {
        // Normal 模板
        NsDataCompare::DataCompareKernel<DTYPE_X1> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
}
