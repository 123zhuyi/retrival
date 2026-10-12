/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_ARA_tiling.cpp
\brief
*/
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

#include "math/grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_dag.h"
#include "math/grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_struct.h"
#include "math/grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_tilingkey.h"
#include "grouped_bias_add_grad_ARA_tiling_arch35.h"

namespace optiling {

using namespace Ops::Base;

static ge::graphStatus convertCompileInfo(
    const GroupedBiasAddGradCompileInfoArch35* compileInfo, ReduceOpCompileInfo* reduceCompileInfo,
    gert::TilingContext* context)
{
    reduceCompileInfo->cacheLineSize = static_cast<uint64_t>(compileInfo->clSize);
    reduceCompileInfo->ubBlockSize = static_cast<uint64_t>(compileInfo->blockSize);
    reduceCompileInfo->vRegSize = static_cast<uint64_t>(compileInfo->vRegSize);
    reduceCompileInfo->vectorCoreNum = static_cast<uint64_t>(compileInfo->coreNum);

    uint64_t ubSize = static_cast<uint64_t>(compileInfo->ubSize);
    if (ubSize <= static_cast<uint64_t>(CACHE_BUF_SIZE)) {
        std::string valueMsg = std::to_string(ubSize);
        std::string reasonMsg = "ubSize should be larger than " + std::to_string(CACHE_BUF_SIZE);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
            valueMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    reduceCompileInfo->ubSize = ubSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(
    gert::TilingContext* context, const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
    ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    ReduceOpTilingData reduceTiling;

    if (opInput.inputDtype == ge::DT_FLOAT) {
        status = Tiling4ReduceOp<GBAGradTPL::GBAGDag<float>::OpDag>(context, opInput, key, compileInfo, &reduceTiling);
    } else if (opInput.inputDtype == ge::DT_FLOAT16) {
        status = Tiling4ReduceOp<GBAGradTPL::GBAGDag<half>::OpDag>(context, opInput, key, compileInfo, &reduceTiling);
    } else {
        status =
            Tiling4ReduceOp<GBAGradTPL::GBAGDag<bfloat16_t>::OpDag>(context, opInput, key, compileInfo, &reduceTiling);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "grad_y", Ops::Base::ToString(opInput.inputDtype).c_str(),
            "int8/uint8/bfloat16/float16/float/int32/int64"),
        return ge::GRAPH_FAILED);

    // set Tiling data
    GroupedBiasAddGradARATilingData* tiling = context->GetTilingData<GroupedBiasAddGradARATilingData>();
    tiling->reduceTiling = reduceTiling;

    return status;
}

static ge::graphStatus GetRIGinputParam(gert::TilingContext* context, ReduceOpInputParam& opInput)
{
    // get x_grad output
    auto gradYInputShapePtr = context->GetInputShape(GRAD_Y_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYInputShapePtr);
    auto gradYInputShape = gradYInputShapePtr->GetStorageShape();
    int32_t gradYDimNum_ = gradYInputShape.GetDimNum();
    if (gradYDimNum_ != ARA_DIM_NUM) {
        std::string dimMsg = std::to_string(gradYDimNum_);
        std::string correctMsg = std::to_string(ARA_DIM_NUM);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "grad_y",
            dimMsg.c_str(), correctMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (ReduceOpTmpl::GetInputDtype(context, GRAD_Y_INPUT_IDX, opInput.inputDtype) == ge::GRAPH_FAILED) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "grad_y",
            "", "Failed to get input dtype of grad_y");
        return ge::GRAPH_FAILED;
    }

    opInput.axes.resize(1);
    opInput.axes[0] = 1;
    opInput.shape.resize(ARA_DIM_NUM);
    opInput.shape[0] = gradYInputShape.GetDim(0);
    opInput.shape[1] = gradYInputShape.GetDim(1);
    opInput.shape[ARA_DIM_NUM - 1] = gradYInputShape.GetDim(ARA_DIM_NUM - 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4GroupedBiasAddGradARA(
    gert::TilingContext* context, const GroupedBiasAddGradCompileInfoArch35* compileInfo)
{
    OP_LOGI(context, "entering Tiling4GroupedBiasAddGradARA");

    ReduceOpCompileInfo reduceCompileInfo;
    if (convertCompileInfo(compileInfo, &reduceCompileInfo, context) == ge::GRAPH_FAILED) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "compileInfo",
            "", "convert compile info Failed for Reduce template");
        return ge::GRAPH_FAILED;
    }

    ReduceOpInputParam opInput;
    if (GetRIGinputParam(context, opInput) == ge::GRAPH_FAILED) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "opInput",
            "", "ReduceOp get input param of grad_y failed");
        return ge::GRAPH_FAILED;
    }

    groupedBiasAddGradARATilingKey key;
    if (DoTiling(context, &reduceCompileInfo, opInput, key.ReduceTiling) == ge::GRAPH_FAILED) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "tilingData",
            "", "DoTiling Failed for GroupedBiasAddGrad ARA");
        return ge::GRAPH_FAILED;
    }

    key.templateNum = GroupedBiasAddGradTilingModeArch35::IS_REDUCE_T;

    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key.ReduceTiling, static_cast<uint32_t>(key.templateNum), 0);
    context->SetTilingKey(tilingKey);

    OP_LOGI(context, "leaving Tiling4GBAG_ARA");
    return ge::GRAPH_SUCCESS;
}

bool IsGroupedBiasAddGradARA(const gert::TilingContext* context)
{
    auto groupIdxInputShapePtr = context->GetOptionalInputShape(GROUP_IDX_INPUT_IDX);
    if (groupIdxInputShapePtr == nullptr) {
        return true;
    } else {
        return false;
    }
}

} // namespace optiling
