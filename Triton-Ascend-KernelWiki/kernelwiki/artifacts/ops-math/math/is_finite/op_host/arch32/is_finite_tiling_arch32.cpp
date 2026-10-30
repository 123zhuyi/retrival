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
 * \file is_finite_tiling_arch32.cpp
 * \brief
 */
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "is_finite_tiling_regbase.h"
#include "is_finite_tiling_arch32.h"
#include "op_host/tiling_base_util.h"

using namespace ge;
using namespace IsFiniteNs;

namespace optiling {

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;

static ge::graphStatus TilingPrepare4IsFiniteArch32(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<IsFiniteCompileInfoArch32>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isRegbase = (Ops::Base::IsRegbaseSocVersion(context)) ? true : false;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "IsFinite GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus IsFiniteTilingArch32(gert::TilingContext* tilingContext)
{
    OP_CHECK_IF(tilingContext == nullptr, OP_LOGE("IsFiniteTiling", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Entering IsFiniteTilingArch32");
    auto compileInfo = reinterpret_cast<const IsFiniteCompileInfoArch32*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
    if (compileInfo->isRegbase) {
        OP_LOGD(tilingContext, "Entering IsFiniteRegbaseTiling");
        IsFiniteRegbaseTiling IsFiniteOpTiling(tilingContext);
        return IsFiniteOpTiling.RunTiling();
    }

    auto tempInputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(tempInputDesc == nullptr, OP_LOGE(tilingContext, "InputDesc == nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* shape = tilingContext->GetInputShape(0);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(tilingContext, "InputShape == nullptr"), return ge::GRAPH_FAILED);

    IsFiniteNs::IsFiniteTilingData* tilingData = tilingContext->GetTilingData<IsFiniteNs::IsFiniteTilingData>();
    IsFiniteNs::IsFiniteTiling::IsFiniteCommonTiling<gert::Shape>(shape->GetStorageShape(), *tilingData, 
                                                                  compileInfo->totalCoreNum, compileInfo->ubSize);

    uint32_t D_T_X = IS_FINITE_TPL_FP32, D_T_Y = IS_FINITE_TPL_BOOL;
    ge::DataType dtype_x = tilingContext->GetInputDesc(0)->GetDataType();
    if (dtype_x == ge::DataType::DT_FLOAT) {
        D_T_X = IS_FINITE_TPL_FP32;
    } else if(dtype_x == ge::DataType::DT_FLOAT16) {
        D_T_X = IS_FINITE_TPL_FP16;
    } else if(dtype_x == ge::DataType::DT_BF16) {
        D_T_X = IS_FINITE_TPL_BF16;
    }

    tilingContext->SetBlockDim(tilingData->needCoreNum);
    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X, D_T_Y); //模板参数配置
    tilingContext->SetTilingKey(tilingKey);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = WORK_SPACE_SIZE;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(IsFinite).Tiling(IsFiniteTilingArch32).TilingParse<IsFiniteCompileInfoArch32>(TilingPrepare4IsFiniteArch32);
} // namespace optiling