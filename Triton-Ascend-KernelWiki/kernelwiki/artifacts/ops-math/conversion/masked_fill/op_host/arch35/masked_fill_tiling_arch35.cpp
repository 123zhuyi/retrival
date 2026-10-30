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
 * /file masked_fill_tiling_arch35.cpp
 * /brief masked_fill_tiling source file
 */

#include "op_host/math_tiling_templates_registry.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/masked_fill_dag.h"
#include "../../op_kernel/arch35/masked_fill_struct.h"
#include "masked_fill_tiling_arch35.h"

using namespace Ops::Base;
using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr uint64_t MASKED_FILL_COMMON_TILING_PRIORITY = 0;
static constexpr uint32_t MASKED_FILL_INTPUT_X_INDEX = 0;
static constexpr uint32_t MASKED_FILL_INTPUT_MASK_INDEX = 1;
static constexpr uint32_t MASKED_FILL_INTPUT_VALUE_INDEX = 2;
static constexpr uint32_t MASKED_FILL_OUTPUT_Y_INDEX = 0;

ge::graphStatus MaskedFillTiling::GetShapeAttrsInfo()
{
    return ge::GRAPH_SUCCESS;
}

bool MaskedFillTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaskedFillTiling::MaskedFillCheckOutputDtype()
{
    auto inputDesc = context_->GetInputDesc(MASKED_FILL_INTPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputValueDtype = inputDesc->GetDataType();

    auto outputDesc = context_->GetOutputDesc(MASKED_FILL_OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    ge::DataType outputValueDtype = outputDesc->GetDataType();

    OP_CHECK_IF((inputValueDtype != outputValueDtype),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "value(input) and y(output)",
            Ops::Base::ToString(inputValueDtype) + " and " + Ops::Base::ToString(outputValueDtype),
            "The dtypes of value(input) and y(output) must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedFillTiling::DoOpTiling()
{
    auto inputDesc = context_->GetInputDesc(MASKED_FILL_INTPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType inputDType = inputDesc->GetDataType();

    auto maskDesc = context_->GetInputDesc(MASKED_FILL_INTPUT_MASK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskDesc);
    ge::DataType maskDType = maskDesc->GetDataType();

    auto valueDesc = context_->GetInputDesc(MASKED_FILL_INTPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    ge::DataType valueDType = valueDesc->GetDataType();

    // Check if the mask is of type bool
    if (maskDType != ge::DT_BOOL && maskDType != ge::DT_INT8) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "mask(input)",
            Ops::Base::ToString(maskDType),
            "Bool or Int8");
        return ge::GRAPH_FAILED;
    }

    // Check if the input and value have the same dtype
    if (inputDType != valueDType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x(input) and value(input)",
            Ops::Base::ToString(inputDType) + " and " + Ops::Base::ToString(valueDType),
            "The dtypes of x(input) and value(input) must be the same");
        return ge::GRAPH_FAILED;
    }

    // Check if the value shape size is 1
    auto valueStorageShape = context_->GetInputShape(MASKED_FILL_INTPUT_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueStorageShape);
    auto valueShape = EnsureNotScalar(valueStorageShape->GetStorageShape());
    if (valueShape.GetShapeSize() != 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "value(input)",
            std::to_string(valueShape.GetShapeSize()),
            "The shape size of value(input) must be 1");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(MaskedFillCheckOutputDtype() == ge::GRAPH_FAILED,
        OP_LOGE(context_, "Check output dtype failed"), return ge::GRAPH_FAILED);

    // Handle different data types
    return HandleDataTypes(inputDType);
}

ge::graphStatus MaskedFillTiling::HandleDataTypes(ge::DataType inputDType)
{
    if (inputDType == ge::DT_INT64) {
        BroadcastBaseTiling<MaskedFill<int64_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT32) {
        BroadcastBaseTiling<MaskedFill<int32_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_BF16) {
        BroadcastBaseTiling<MaskedFill<half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<MaskedFill<half>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_FLOAT) {
        BroadcastBaseTiling<MaskedFill<float>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_INT8) {
        BroadcastBaseTiling<MaskedFill<int8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (inputDType == ge::DT_BOOL) {
        BroadcastBaseTiling<MaskedFill<int8_t>::OpDag> brcBaseTiling(context_);
        brcBaseTiling.DoTiling();
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x(input)",
            Ops::Base::ToString(inputDType),
            "Bool, Float16, BFloat16, Float, Int8, Int32 and Int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedFillTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaskedFillTiling::GetTilingKey() const
{
    return tilingKey;
}

ge::graphStatus MaskedFillTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedFillTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedFillTiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForMaskedFill(gert::TilingContext* context)
{
    OP_LOGD("MaskedFillTiling", "Enter TilingForMaskedFill");
    if (context == nullptr) {
        OP_LOGE("MaskedFillTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const MaskedFillCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc MaskedFillTiling");
    return Ops::Math::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepareForMaskedFill(gert::TilingParseContext *context)
{
    auto compileInfoPtr = context->GetCompiledInfo<MaskedFillCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MaskedFill)
    .Tiling(TilingForMaskedFill)
    .TilingParse<MaskedFillCompileInfo>(TilingPrepareForMaskedFill);

REGISTER_OPS_TILING_TEMPLATE(MaskedFill, MaskedFillTiling, MASKED_FILL_COMMON_TILING_PRIORITY);
} // namespace optiling