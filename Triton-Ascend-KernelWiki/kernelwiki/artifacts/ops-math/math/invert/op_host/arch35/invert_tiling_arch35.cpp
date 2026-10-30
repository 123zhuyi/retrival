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
 * \file invert_tiling_arch35.cc
 * \brief invert tiling
 */

#include "invert_tiling_arch35.h"
#include <iostream>
#include <graph/utils/type_utils.h>
#include "op_host/tiling_base_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "math/invert/op_kernel/arch35/invert_dag.h"

namespace optiling {
constexpr uint64_t INVERT_TILING_KEY = 101;
constexpr size_t INVERT_WORKSPACE_RESERVE_BYTE = 16777216; // 16 * 1024 * 1024

ge::graphStatus InvertTiling::SetTilingData()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = INVERT_WORKSPACE_RESERVE_BYTE;
    context_->SetTilingKey(INVERT_TILING_KEY);
    context_->SetBlockDim(td_->blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InvertTiling::CheckAndGetOutputDtype(ge::DataType& outputDtype)
{
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    outputDtype = outputDesc->GetDataType();
    OP_CHECK_IF((inputDtype != outputDtype), 
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
            Ops::Base::ToString(outputDtype),
            "the dtype of y must be the same as " + Ops::Base::ToString(inputDtype) + " of x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InvertTiling::RunTiling()
{
    ge::DataType outputDtype = ge::DataType::DT_MAX;
    OP_CHECK_IF(
        CheckAndGetOutputDtype(outputDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "get output dtype failed."), return ge::GRAPH_FAILED);

    ge::graphStatus res = ge::GRAPH_FAILED;
    td_ = context_->GetTilingData<EleBaseTilingData>();
    switch (outputDtype) {
        case ge::DataType::DT_INT8: {
            res = DoTiling<InvertDag<int8_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_INT16: {
            res = DoTiling<InvertDag<int16_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_INT32: {
            res = DoTiling<InvertDag<int32_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_INT64: {
            res = DoTiling<InvertDag<int64_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_UINT8: {
            res = DoTiling<InvertDag<uint8_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_UINT16: {
            res = DoTiling<InvertDag<uint16_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_UINT32: {
            res = DoTiling<InvertDag<uint32_t>::OpDag>(*td_);
            break;
        }
        case ge::DataType::DT_UINT64: {
            res = DoTiling<InvertDag<uint64_t>::OpDag>(*td_);
            break;
        }
        default: {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
                Ops::Base::ToString(outputDtype),
                "the dtype of y must be within the range "
                "{DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64}");
            
            return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF(
        res != ge::GRAPH_SUCCESS, OP_LOGE(context_->GetNodeName(), "do elewise tiling failed."),
        return ge::GRAPH_FAILED);

    res = SetTilingData();
    return res;
}

static ge::graphStatus TilingInvert(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingInvert");
    auto compileInfo = reinterpret_cast<const ElewiseCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    InvertTiling invertTiling(context);
    return invertTiling.RunTiling();
}
static ge::graphStatus TilingPrepareForInvert([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Invert).Tiling(TilingInvert).TilingParse<ElewiseCompileInfo>(TilingPrepareForInvert);
} // namespace optiling