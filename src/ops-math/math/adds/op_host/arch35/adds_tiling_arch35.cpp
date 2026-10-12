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
 * \file adds_tiling.cpp
 * \brief Adds 算子 Host Tiling 实现（atvoss 框架 - Elewise 模式）
 */

#include "register/op_def_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/log/log.h"
#include "../../op_kernel/arch35/adds_dag.h"
#include "../../op_kernel/arch35/adds_tiling_data.h"
#include "../../op_kernel/arch35/adds_struct.h"
#include "adds_tiling_arch35.h"

namespace optiling {

using namespace ge;
using namespace AddsOp;

constexpr uint64_t WORKSPACE_RESERVE_BYTE = 0;

ge::graphStatus AddsTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = WORKSPACE_RESERVE_BYTE;
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddsTiling::CalcOutputDtype()
{
    auto inputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDesc);
    ge::DataType inputDtype = inputDesc->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(inputDtype != this->outputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "x and y",
            (Ops::Base::ToString(inputDtype) + " and " + Ops::Base::ToString(this->outputDtype)).c_str(),
            "The dtypes of x and y must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddsTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "AddsTiling CheckShape enter.");
    auto inputStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShape);
    const gert::Shape& inputShape = inputStorageShape->GetStorageShape();

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputShape = outputStorageShape->GetStorageShape();

    if (inputShape != outputShape) {
        std::string inputShapeStr = Ops::Base::ToString(inputShape);
        std::string outputShapeStr = Ops::Base::ToString(outputShape);
        std::string shapesStr = inputShapeStr + " and " + outputShapeStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "x and y",
            shapesStr.c_str(), "The shapes of x and y must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * \brief Adds 算子 Tiling 函数
 * 
 * 执行流程：
 * 1. 获取 TilingData
 * 2. 创建 ElewiseBaseTiling 对象
 * 3. 根据输入 dtype 选择对应 OpDag 模板实例化
 * 4. 执行 DoTiling（自动计算多核切分、UB切分等）
 * 5. 获取标量值并填充到 tilingData->scalarValue
 * 6. 生成并设置 TilingKey
 */
ge::graphStatus AddsTiling::RunTiling()
{
    // 获取 TilingData
    tiling = tilingContext->GetTilingData<AddsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    
    // 创建 ElewiseBaseTiling 对象
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED,
               OP_LOGE(tilingContext->GetNodeName(), "get output dtype failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext->GetNodeName(), "check shape failed"),
               return ge::GRAPH_FAILED);
    
    // 根据输入 dtype 选择对应的 OpDag 模板实例化
    // bf16/fp16/int32/fp32: 使用 AddsOp<T, CAST_MODE_NONE, CAST_MODE_RINT>（fp32→fp32会优化掉）
    // int16: 使用 AddsInt16Op（特殊截断处理）
    ge::graphStatus ret;
    uint64_t dType=0;
    switch (this->outputDtype) {
        case ge::DT_FLOAT16:
            dType = TPL_FP16;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsOp<half, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_FLOAT:
            dType = TPL_FP32;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsOp<float, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_BF16:
            dType = TPL_BF16;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsOp<bfloat16_t, NsAdds::CAST_MODE_NONE, NsAdds::CAST_MODE_RINT>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_INT16:
            dType = TPL_INT16;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsInt16Op::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_INT32:
            dType = TPL_INT32;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsOp<int32_t, NsAdds::CAST_MODE_RINT, NsAdds::CAST_MODE_TRUNC>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_INT64:
            dType = TPL_INT64;
            ret = elewiseBaseTiling.DoTiling<NsAdds::AddsOp<int64_t, NsAdds::CAST_MODE_RINT, NsAdds::CAST_MODE_TRUNC>::OpDag>(tiling->baseTiling);
            break;
        default:
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                tilingContext->GetNodeName(),
                "y",
                Ops::Base::ToString(this->outputDtype).c_str(),
                "The dtype of y must be within the range [DT_FLOAT16, DT_FLOAT, DT_BF16, DT_INT16, DT_INT32, DT_INT64].");
            return ge::GRAPH_FAILED;
    }
    
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS,
        OP_LOGE(tilingContext->GetNodeName(), "Adds: ElewiseBaseTiling DoTiling failed"),
        return ret);
    // 获取标量值（从算子属性）
    // Adds 算子的 value 属性类型为 float
    auto runtimeAttrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, runtimeAttrs);
    tiling->scalarValue = 1;
    
    const float* valuePtr = runtimeAttrs->GetAttrPointer<float>(0);
    if (valuePtr != nullptr) {
        tiling->scalarValue = *valuePtr;
    }
    // 设置 TilingKey（schMode + dtype）
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<uint64_t>(tiling->baseTiling.scheMode),
        dType);
    tilingContext->SetTilingKey(tilingKey);
    ge::graphStatus result=SetTilingData();
    OP_LOGI(tilingContext, "Adds: Tiling success, scalarValue=%f, dtype=%d, tilingKey=%lu",
            tiling->scalarValue, dType, tilingKey);
    
    return result;
}

static ge::graphStatus TilingForAdds(gert::TilingContext* context)
{
    OP_LOGD("AddsTiling", "Enter TilingForAdds");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"),
               return ge::GRAPH_FAILED);

    OP_LOGD("AddsTiling", "Enter new AddsTiling");
    AddsTiling addsTiling(context);
    return addsTiling.RunTiling();
}


/**
 * \brief Tiling Parse 函数（可选，当前无需特殊处理）
 */
static ge::graphStatus TilingParseForAdds([[maybe_unused]] gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<AddsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * \brief 注册 Tiling 函数
 */
IMPL_OP_OPTILING(Adds)
    .Tiling(TilingForAdds)
    .TilingParse<AddsCompileInfo>(TilingParseForAdds);

}  // namespace optiling