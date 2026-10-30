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
 * \file arange_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/arange_tiling_data.h"
#include "../op_kernel/arange_tiling_key.h"
#define DIVIDE_AND_ALIGN(size, split, align) ((((size) / (split)) + ((align) - 1)) & ~((align) - 1))

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t DTYPE_SIZE2 = 2;
const uint32_t DTYPE_SIZE4 = 4;
const uint32_t DTYPE_SIZE8 = 8;

struct ArangeCompileInfo {};

// tiling 分发入口
static ge::graphStatus ArangeTilingFunc(gert::TilingContext* context)
{
    ArangeTilingData* tiling = context->GetTilingData<ArangeTilingData>();
    uint32_t totalLength = context->GetOutputShape(0)->GetOriginShape().GetShapeSize();
    ge::DataType dtype_out = context->GetOutputDesc(0)->GetDataType();
    uint32_t dtype_size = DTYPE_SIZE2;
    uint64_t tilingkey = 0;
    switch (dtype_out) {
        case ge::DataType::DT_FLOAT16:
        case ge::DataType::DT_BF16:
            dtype_size = DTYPE_SIZE2;
            break;
        case ge::DataType::DT_FLOAT:
            dtype_size = DTYPE_SIZE4;
            tilingkey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
            break;
        case ge::DataType::DT_INT32:
            dtype_size = DTYPE_SIZE4;
            break;
        case ge::DataType::DT_INT64:
            dtype_size = DTYPE_SIZE8;
            break;
        default:
            dtype_size = DTYPE_SIZE2;
            break;
    }
    context->SetTilingKey(tilingkey);
    uint64_t ub_size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    /*单次api计算大小：将ub 10等份后并按BLOCK_SIZE对齐*/
    uint64_t ub_unit_size = DIVIDE_AND_ALIGN(ub_size, 10, BLOCK_SIZE);
    uint32_t totalNum = totalLength;
    uint32_t unitNum = ub_unit_size / dtype_size;
    uint32_t unitLoops = totalNum / unitNum;
    uint32_t tailNum = totalNum - unitNum * unitLoops;
    if (tailNum > 0)
        unitLoops += 1;

    tiling->dtypeSize = dtype_size;
    tiling->totalNum = totalNum;
    tiling->unitNum = unitNum;
    tiling->unitLoops = unitLoops;
    tiling->tailNum = tailNum;

    context->SetBlockDim(1);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForArange([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Arange).Tiling(ArangeTilingFunc).TilingParse<ArangeCompileInfo>(TilingParseForArange);
} // namespace optiling
