/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/not_equal_v2_tiling_data.h"
#include "../op_kernel/not_equal_v2_tiling_key.h"

namespace optiling {
struct NotEqualV2CompileInfo {};

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto& x1 = context->GetInputShape(0)->GetStorageShape();
    auto size = 1;
    for (auto i = 0; i < x1.GetDimNum(); i++)
        size *= x1[i];

    auto& tiling = *context->GetTilingData<NotEqualTilingData>();
    tiling.size = size;

    auto platform_info = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
    auto plaform = platform_ascendc::PlatformAscendC(platform_info);
    auto dtype = context->GetInputTensor(0)->GetDataType();
    auto data_size = ge::GetSizeInBytes(size, dtype);
    auto block_dim = plaform.GetCoreNumAiv();
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_BF16) {
        if (size < 1 << 13)
            block_dim = 1;
        else if (size < 1 << 21)
            block_dim /= 2;
    } else {
        if (data_size < 1 << 16)
            block_dim = 1;
        else if (data_size < 1 << 24)
            block_dim /= 2;
    }
    context->SetBlockDim(block_dim);
    auto tiling_key = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tiling_key);
    auto workspace_sizes = context->GetWorkspaceSizes(1);
    workspace_sizes[0] = plaform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(NotEqualV2).Tiling(TilingFunc).TilingParse<NotEqualV2CompileInfo>(TilingParse);
} // namespace optiling
