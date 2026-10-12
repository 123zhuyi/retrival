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
 * \file fill_diagonal_v2_tiling.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "fill_diagonal_v2_tiling.h"

constexpr size_t MAX_DIM_NUM = 8;

namespace optiling {
inline ge::graphStatus SetTilingKey4FillDiagonalV2(gert::TilingContext* context, uint64_t step, uint64_t end, bool wrap)
{
    uint64_t endThreshold = 1000000;
    uint64_t stepThreshold1B = 1000;
    uint64_t stepThreshold2B = 300;
    uint64_t stepThreshold4B = 200;
    uint64_t stepThreshold8B = 80;
    uint64_t stepThreshold = 0;
    auto dtype = context->GetInputDesc(0)->GetDataType();
    int64_t tilingKey = 0;
    if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL) {
        stepThreshold = stepThreshold1B;
    } else if (dtype == ge::DT_INT16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        stepThreshold = stepThreshold2B;
    } else if (dtype == ge::DT_INT32 || dtype == ge::DT_FLOAT) {
        stepThreshold = stepThreshold4B;
    } else if (dtype == ge::DT_INT64 || dtype == ge::DT_DOUBLE) {
        stepThreshold = stepThreshold8B;
    } else {
        OP_LOGE(context, "Unsupported data type %d.", dtype);
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    bool isAscend950 = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950;
    if (!isAscend950 && wrap && (end > endThreshold) &&
        (step < stepThreshold)) { // dense: A2A3 only, A5 uses sparse due to L2 cache coherence issue
        tilingKey = 1;
    } else {
        tilingKey = 0;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGD(context, "Tiling key is %ld.", tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4FillDiagonalV2(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const FillDiagonalV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto tilingCoreNum = compileInfo->totalCoreNum;
    FillDiagonalV2TilingData tiling;
    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);
    auto shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);

    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                                                         "The dim num of x must be less than or equal to 8"),
                return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dtype = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_UINT8,
                                                    ge::DT_BOOL,    ge::DT_INT8,  ge::DT_INT16,  ge::DT_INT32,
                                                    ge::DT_BF16,    ge::DT_INT64};
    OP_CHECK_IF(supportedDtypes.count(dtype) == 0,
                OP_LOGE_WITH_INVALID_INPUT_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dtype).c_str(),
                                                 "DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_BOOL, DT_INT8, "
                                                 "DT_INT16, DT_INT32, DT_BFLOAT16, DT_INT64"),
                return ge::GRAPH_FAILED);

    auto fillValueDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, fillValueDesc);
    OP_CHECK_IF(fillValueDesc->GetDataType() != dtype,
                OP_LOGE(context->GetNodeName(), "fill_value dtype %s must be same as x dtype %s",
                        Ops::Base::ToString(fillValueDesc->GetDataType()).c_str(), Ops::Base::ToString(dtype).c_str()),
                return ge::GRAPH_FAILED);

    uint64_t totalLength = 1;
    for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= shape->GetStorageShape().GetDim(i);
    }
    const uint64_t blockLength = totalLength / tilingCoreNum;
    const uint64_t lastBlockLength = totalLength - totalLength / tilingCoreNum * (tilingCoreNum - 1);
    uint64_t step = 0;
    constexpr uint64_t matrixDim = 2;
    if (shape->GetStorageShape().GetDimNum() == matrixDim) {
        step = shape->GetStorageShape().GetDim(1) + 1;
    } else {
        uint64_t mult = 1;
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); ++i) {
            step += mult;
            mult *= shape->GetStorageShape().GetDim(i);
        }
    }
    auto wrapPtr = attrPtr->GetAttrPointer<bool>(0);
    uint64_t end = totalLength;
    if (shape->GetStorageShape().GetDimNum() == matrixDim &&
        shape->GetStorageShape().GetDim(0) > shape->GetStorageShape().GetDim(1) && !(*wrapPtr)) {
        end = shape->GetStorageShape().GetDim(1) * shape->GetStorageShape().GetDim(1);
    }
    tiling.set_totalLength(totalLength);
    tiling.set_step(step);
    tiling.set_end(end);
    tiling.set_ubSize(compileInfo->ubSizePlatForm);
    tiling.set_blockLength(blockLength);
    tiling.set_lastBlockLength(lastBlockLength);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    if (SetTilingKey4FillDiagonalV2(context, step, end, *wrapPtr) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(tilingCoreNum);
    size_t* currentWorkspaceSize = context->GetWorkspaceSizes(1);
    currentWorkspaceSize[0] = compileInfo->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4FillDiagonalV2(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Tiling Prepare For FillDiagonalV2 start.");
    auto compileInfo = context->GetCompiledInfo<FillDiagonalV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->totalCoreNum < 1) {
        OP_LOGE(context->GetNodeName(), "Core num is %lu.", compileInfo->totalCoreNum);
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "Tiling Prepare For FillDiagonalV2 end.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the FillDiagonalV2 op.
IMPL_OP_OPTILING(FillDiagonalV2)
    .Tiling(Tiling4FillDiagonalV2)
    .TilingParse<FillDiagonalV2CompileInfo>(TilingPrepare4FillDiagonalV2);
} // namespace optiling
