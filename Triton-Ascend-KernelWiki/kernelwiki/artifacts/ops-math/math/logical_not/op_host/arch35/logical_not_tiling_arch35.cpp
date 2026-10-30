/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logical_not_tiling_arch35.cpp
 * \brief
 */

#include "tiling/tiling_api.h"
#include "math/logical_not/op_kernel/arch35/logical_not_dag.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {
const uint64_t LOGICAL_NOT_SYS_WORKSPACE = 16777216; // 16M

class LogicalNotTiling {
public:
    explicit LogicalNotTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunTiling();
private:
    gert::TilingContext *tilingContext;
};

ge::graphStatus LogicalNotTiling::RunTiling()
{
OP_CHECK_IF(tilingContext == nullptr,
                      OP_LOGE("CheckContextValid", "tilingContext is nullptr"),
                      return ge::GRAPH_FAILED);

    auto tiling = tilingContext->GetTilingData<EleBaseTilingDataV2>();
OP_CHECK_IF((tiling == nullptr),
                      OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "tiling_data", "nullptr", "not nullptr"),
                      return ge::GRAPH_FAILED);

    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
OP_CHECK_IF(elewiseBaseTiling.DoTiling<LogicalNotOp::LogicalNotDag<int8_t>::OpDag>(*tiling) != ge::GRAPH_SUCCESS,
                     OP_LOGE(tilingContext->GetNodeName(), "elewiseBaseTiling DoTiling failed"),
                     return ge::GRAPH_FAILED);

    // set workspace/tilingkey/blockdim
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = LOGICAL_NOT_SYS_WORKSPACE;

    tilingContext->SetTilingKey(101UL);
    tilingContext->SetBlockDim(tiling->blockNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4LogicalNot(gert::TilingContext *context)
{
    auto compileInfo = context->GetCompileInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    LogicalNotTiling logicalNotTiling(context);
    return logicalNotTiling.RunTiling();
}


ge::graphStatus TilingPrepareForLogicalNot(gert::TilingParseContext *context)
{
    OP_LOGD("ElewiseTiling", "Enter TilingPrepareForLogicalNot.");
    auto compileInfoPtr = context->GetCompiledInfo<ElewiseCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogicalNot).Tiling(Tiling4LogicalNot)
                            .TilingParse<ElewiseCompileInfo>(TilingPrepareForLogicalNot);
}  // namespace optiling