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
 * \file floor_mod_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "register/tilingdata_base.h"

#include "../op_kernel/floor_mod_tiling_data.h"
#include "../op_kernel/floor_mod_tiling_key.h"
#include "torch_extension/tiling_utils.h"
#include "op_host/tiling_base_util.h"

using namespace ge;

namespace FloorModNs {

class FloorModTiling {
public:
    constexpr static int64_t MINIMUM_ELEMENT_PER_CORE = 1024;

    constexpr static int64_t DATA_BLOCK = 64;
    constexpr static int64_t RESERVERD_UB_SIZE = 1024;

    template <typename T>
    static void FloorModCommonTiling(
        T x, FloorModTilingData& tilingData, uint32_t coreNum, uint64_t ubSize, uint32_t ubDivider)
    {
        if (ubDivider == 0) {
            return;
        }

        int64_t elementCount = 1;

        for (uint16_t i = 0; i < TilingUtils::GetDimNum(x); i++) {
            elementCount *= TilingUtils::GetDim(x, i);
        }

        uint32_t blockDim = (elementCount + MINIMUM_ELEMENT_PER_CORE - 1) / MINIMUM_ELEMENT_PER_CORE;

        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim == 0) {
            blockDim = 1;
        }

        uint32_t dataBlockSize = DATA_BLOCK;
        uint32_t usableUbSize = uint32_t(ubSize - RESERVERD_UB_SIZE - sizeof(FloorModTilingData)) / ubDivider;
        usableUbSize = usableUbSize / dataBlockSize * dataBlockSize;

        uint64_t perCoreDataCount = elementCount / blockDim;
        perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;

        uint64_t tempTailDataCount = elementCount - perCoreDataCount * blockDim;
        uint64_t tailDataCoreNum = 0;
        uint64_t lastCoreDataCount = 0;

        tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
        lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;

        tilingData.usableUbSize = usableUbSize;
        tilingData.needCoreNum = blockDim;
        tilingData.totalDataCount = elementCount;
        tilingData.perCoreDataCount = perCoreDataCount;
        tilingData.tailDataCoreNum = tailDataCoreNum;
        tilingData.lastCoreDataCount = lastCoreDataCount;
    }
};

} // namespace FloorModNs

using namespace FloorModNs;

namespace optiling {

struct FloorModCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t UB_DIVIDER_FP32 = 50;
constexpr uint32_t UB_DIVIDER_FP16 = 46;
constexpr uint32_t UB_DIVIDER_INT32 = 66;

static ge::graphStatus TilingPrepare4FloorModTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<FloorModCompileInfo>();
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
            context, "FloorMod GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
            compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKeyParams(
    ge::DataType dtype, uint32_t& dTypeX1, uint32_t& dTypeX2, uint32_t& dTypeY, uint32_t& ubDivider)
{
    if (dtype == ge::DataType::DT_FLOAT) {
        dTypeX1 = FLOOR_MOD_TPL_FP32;
        dTypeX2 = FLOOR_MOD_TPL_FP32;
        dTypeY = FLOOR_MOD_TPL_FP32;
        ubDivider = UB_DIVIDER_FP32;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        dTypeX1 = FLOOR_MOD_TPL_FP16;
        dTypeX2 = FLOOR_MOD_TPL_FP16;
        dTypeY = FLOOR_MOD_TPL_FP16;
        ubDivider = UB_DIVIDER_FP16;
    } else if (dtype == ge::DataType::DT_BF16) {
        dTypeX1 = FLOOR_MOD_TPL_BF16;
        dTypeX2 = FLOOR_MOD_TPL_BF16;
        dTypeY = FLOOR_MOD_TPL_BF16;
        ubDivider = UB_DIVIDER_FP16;
    } else if (dtype == ge::DataType::DT_INT32) {
        dTypeX1 = FLOOR_MOD_TPL_INT32;
        dTypeX2 = FLOOR_MOD_TPL_INT32;
        dTypeY = FLOOR_MOD_TPL_INT32;
        ubDivider = UB_DIVIDER_INT32;
    } else {
        dTypeX1 = FLOOR_MOD_TPL_FP32;
        dTypeX2 = FLOOR_MOD_TPL_FP32;
        dTypeY = FLOOR_MOD_TPL_FP32;
        ubDivider = UB_DIVIDER_FP32;
    }
}

static ge::graphStatus FloorModTilingForGe(gert::TilingContext* tilingContext)
{
    OP_CHECK_IF(
        tilingContext == nullptr, OP_LOGE("FloorModTiling", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Entering FloorModTilingForGe");
    auto compileInfo = reinterpret_cast<const FloorModCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    auto tempInputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(tempInputDesc == nullptr, OP_LOGE(tilingContext, "InputDesc == nullptr"), return ge::GRAPH_FAILED);
    const gert::StorageShape* shape = tilingContext->GetInputShape(0);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(tilingContext, "InputShape == nullptr"), return ge::GRAPH_FAILED);

    FloorModNs::FloorModTilingData* tilingData = tilingContext->GetTilingData<FloorModNs::FloorModTilingData>();

    uint32_t D_T_X1, D_T_X2, D_T_Y, ubDivider;
    ge::DataType dtype_x = tilingContext->GetInputDesc(0)->GetDataType();

    SetTilingKeyParams(dtype_x, D_T_X1, D_T_X2, D_T_Y, ubDivider);

    FloorModNs::FloorModTiling::FloorModCommonTiling<gert::Shape>(
        shape->GetStorageShape(), *tilingData, compileInfo->totalCoreNum, compileInfo->ubSize, ubDivider);

    tilingContext->SetBlockDim(tilingData->needCoreNum);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X1, D_T_X2, D_T_Y);

    tilingContext->SetTilingKey(tilingKey);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = WORK_SPACE_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FloorMod).Tiling(FloorModTilingForGe).TilingParse<FloorModCompileInfo>(TilingPrepare4FloorModTiling);

} // namespace optiling