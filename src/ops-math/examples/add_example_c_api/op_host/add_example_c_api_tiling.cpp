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
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../op_kernel/add_example_c_api_tiling_data.h"
#include "../op_kernel/add_example_c_api_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int32_t DIMS_LIMIT = 4;
constexpr int64_t BUFFER_NUM = 3;
constexpr int64_t TYPE_SIZE = 4;

struct AddExampleCApiCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = Ops::Base::EnsureNotScalar(inputY->GetStorageShape());
    auto outZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
    auto outShapeZ = Ops::Base::EnsureNotScalar(outZ->GetStorageShape());

    OP_CHECK_IF(
        inputShapeX.GetDimNum() != DIMS_LIMIT || inputShapeY.GetDimNum() != DIMS_LIMIT ||
            outShapeZ.GetDimNum() != DIMS_LIMIT,
        OP_LOGE(
            context, "AddExampleCApi: inputx,inputy,outputz shape dim = %zu, %zu, %zu, should be equal 4",
            inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
        return ge::GRAPH_FAILED);

    totalIdx = inputShapeX.GetShapeSize();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AddExampleCApiTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    AddExampleCApiTilingData* tiling = context->GetTilingData<AddExampleCApiTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AddExampleCApiTilingData), 0, sizeof(AddExampleCApiTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalIdx;
    tiling->blockFactor = Ops::Base::CeilDiv(totalIdx, coreNum);
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->ubFactor = Ops::Base::FloorAlign(Ops::Base::FloorDiv((ubCanUse / TYPE_SIZE), BUFFER_NUM), ubBlockSize);

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(ELEMENTWISE_TPL_SCH_MODE_0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddExampleCApi([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddExampleCApi)
    .Tiling(AddExampleCApiTilingFunc)
    .TilingParse<AddExampleCApiCompileInfo>(TilingParseForAddExampleCApi);
} // namespace optiling
