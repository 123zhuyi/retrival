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
 * \file reciprocal_grad_tiling_arch35.cpp
 * \brief ReciprocalGrad 算子 Host Tiling 实现（atvoss 框架 - Elewise 模式）
 */

#include "register/op_def_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/tiling_base_util.h"
#include "op_common/log/log.h"
#include "../../op_kernel/arch35/reciprocal_grad_dag.h"
#include "../../op_kernel/arch35/reciprocal_grad_tiling_data.h"
#include "../../op_kernel/arch35/reciprocal_grad_struct.h"
#include "reciprocal_grad_tiling_arch35.h"

namespace optiling {

using namespace ge;
using namespace ReciprocalGradOp;

constexpr uint64_t WORKSPACE_RESERVE_BYTE = 0;

ge::graphStatus ReciprocalGradTiling::SetTilingData()
{
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, currentWorkspace);
    currentWorkspace[0] = WORKSPACE_RESERVE_BYTE;
    tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalGradTiling::CalcOutputDtype()
{
    auto inputDescY = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDescY);
    ge::DataType yDtype = inputDescY->GetDataType();

    auto inputDescDy = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputDescDy);
    ge::DataType dyDtype = inputDescDy->GetDataType();

    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();

    OP_CHECK_IF(
        yDtype != this->outputDtype || dyDtype != this->outputDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext->GetNodeName(), "y, dy and z",
                                               (Ops::Base::ToString(yDtype) + ", " + Ops::Base::ToString(dyDtype) +
                                                " and " + Ops::Base::ToString(this->outputDtype))
                                                   .c_str(),
                                               "The dtypes of y, dy and z must be the same"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReciprocalGradTiling::CheckShape()
{
    OP_LOGD(tilingContext->GetNodeName(), "ReciprocalGradTiling CheckShape enter.");
    auto inputStorageShapeY = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShapeY);
    const gert::Shape& yShape = Ops::Base::EnsureNotScalar(inputStorageShapeY->GetStorageShape());

    auto inputStorageShapeDy = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputStorageShapeDy);
    const gert::Shape& dyShape = Ops::Base::EnsureNotScalar(inputStorageShapeDy->GetStorageShape());

    auto outputStorageShape = tilingContext->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputStorageShape);
    const gert::Shape& outputShape = Ops::Base::EnsureNotScalar(outputStorageShape->GetStorageShape());

    if (yShape != outputShape || dyShape != outputShape) {
        std::string yShapeStr = Ops::Base::ToString(yShape);
        std::string dyShapeStr = Ops::Base::ToString(dyShape);
        std::string outputShapeStr = Ops::Base::ToString(outputShape);
        std::string shapesStr = yShapeStr + ", " + dyShapeStr + " and " + outputShapeStr;
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext->GetNodeName(), "y, dy and z", shapesStr.c_str(),
                                               "The shapes of y, dy and z must be the same");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

/**
 * \brief ReciprocalGrad 算子 Tiling 函数
 *
 * 执行流程：
 * 1. 获取 TilingData
 * 2. 创建 ElewiseBaseTiling 对象
 * 3. 根据输入 dtype 选择对应 OpDag 模板实例化
 * 4. 执行 DoTiling（自动计算多核切分、UB切分等）
 * 5. 生成并设置 TilingKey
 */
ge::graphStatus ReciprocalGradTiling::RunTiling()
{
    // 获取 TilingData
    tiling = tilingContext->GetTilingData<ReciprocalGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);

    // 获取输入 desc，确定数据类型
    ElewiseBaseTiling elewiseBaseTiling(tilingContext);
    OP_CHECK_IF(CalcOutputDtype() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "get output dtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
                return ge::GRAPH_FAILED);

    // 根据输入 dtype 选择对应的 OpDag 模板实例化
    ge::graphStatus ret;
    switch (this->outputDtype) {
        case ge::DT_FLOAT16:
            ret = elewiseBaseTiling.DoTiling<NsReciprocalGrad::ReciprocalGradCompute<half>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_FLOAT:
            ret = elewiseBaseTiling.DoTiling<NsReciprocalGrad::ReciprocalGradCompute<float>::OpDag>(tiling->baseTiling);
            break;
        case ge::DT_BF16:
            ret = elewiseBaseTiling.DoTiling<NsReciprocalGrad::ReciprocalGradCompute<bfloat16_t>::OpDag>(
                tiling->baseTiling);
            break;
        default:
            OP_LOGE(tilingContext, "ReciprocalGrad: Unsupported dtype=%u", static_cast<uint32_t>(this->outputDtype));
            return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext, "ReciprocalGrad: ElewiseBaseTiling DoTiling failed"),
                return ret);

    // 设置 TilingKey（仅 schMode，dtype 由编译期宏 DTYPE_X 注入）
    uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(tiling->baseTiling.scheMode));
    tilingContext->SetTilingKey(tilingKey);
    ge::graphStatus result = SetTilingData();
    OP_LOGI(tilingContext, "ReciprocalGrad: Tiling success, tilingKey=%lu", tilingKey);
    return result;
}
static ge::graphStatus TilingForReciprocalGrad(gert::TilingContext* context)
{
    OP_LOGD("ReciprocalGradTiling", "Enter ReciprocalGradTiling");
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "Tiling context is null"), return ge::GRAPH_FAILED);

    OP_LOGD("ReciprocalGradTiling", "Enter new ReciprocalGradTiling");
    ReciprocalGradTiling reciprocalGradTiling(context);
    return reciprocalGradTiling.RunTiling();
}

/**
 * \brief Tiling Parse 函数（可选，当前无需特殊处理）
 */
static ge::graphStatus TilingParseForReciprocalGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<ReciprocalGradCompileInfo>();
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
IMPL_OP_OPTILING(ReciprocalGrad)
    .Tiling(TilingForReciprocalGrad)
    .TilingParse<ReciprocalGradCompileInfo>(TilingParseForReciprocalGrad);

} // namespace optiling
