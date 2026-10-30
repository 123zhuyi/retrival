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
 * \file assign_tiling_arch35.cpp
 * \brief
 */

#include "assign_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {

constexpr int64_t INDEX_INPUT_X = 0;
constexpr int64_t UB_FACTOR_MIN_BETY = 2048;
constexpr int64_t N_BUFFER = 2;
constexpr int64_t ONE_BLK_BYTE = 32;
constexpr int64_t WORKSPACE_SIZE = 32;

constexpr int64_t INDEX_INPUT_REF = 0;
constexpr int64_t INDEX_INPUT_VALUE = 1;
constexpr int64_t DIGIT_TEN = 10;

static int64_t GetRemainder(int64_t u_value, int64_t d_value)
{
    int64_t res_value = 0;
    if (d_value == 0) {
        return u_value;
    }
    res_value = u_value % d_value;

    return res_value;
}

static bool IsInvalidTypeForAssign(const DataType dtype)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT,  ge::DT_FLOAT16,   ge::DT_BF16,     ge::DT_INT64,
                                                   ge::DT_UINT64, ge::DT_INT32,     ge::DT_UINT32,   ge::DT_INT16,
                                                   ge::DT_UINT16, ge::DT_INT8,      ge::DT_UINT8,    ge::DT_DOUBLE,
                                                   ge::DT_BOOL,   ge::DT_COMPLEX32, ge::DT_COMPLEX64};
    bool dtypeInValid = (supportedDtype.count(dtype) == 0);
    return dtypeInValid;
}

static ge::graphStatus CheckDtypeForAssign(const gert::TilingContext* context)
{
    auto refPtr = context->GetInputDesc(INDEX_INPUT_REF);
    OP_CHECK_NULL_WITH_CONTEXT(context, refPtr);
    auto refDtype = refPtr->GetDataType();
    OP_CHECK_IF(IsInvalidTypeForAssign(refDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "ref",
            Ops::Base::ToString(refDtype).c_str(),
            "The dtype of ref must be within the range [DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_UINT64, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8, DT_DOUBLE, DT_BOOL, DT_COMPLEX32, DT_COMPLEX64]."),
        return ge::GRAPH_FAILED);

    auto valuePtr = context->GetInputDesc(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuePtr);
    auto valueDtype = valuePtr->GetDataType();
    OP_CHECK_IF(
        valueDtype != refDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "ref, value",
            (Ops::Base::ToString(refDtype) + ", " + Ops::Base::ToString(valueDtype)).c_str(),
            "The dtypes of ref and value must be the same."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckShapeForAssign(const gert::TilingContext* context, AssignTilingParam& tilingParam)
{
    auto refShapePtr = context->GetInputShape(INDEX_INPUT_REF);
    OP_CHECK_NULL_WITH_CONTEXT(context, refShapePtr);
    auto refShape = refShapePtr->GetStorageShape();

    auto valueShapePtr = context->GetInputShape(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueShapePtr);
    auto valueShape = valueShapePtr->GetStorageShape();
    if (valueShape.GetShapeSize() == 1) {
        tilingParam.tilingKey = DIGIT_TEN;
        return ge::GRAPH_SUCCESS;
    }

    tilingParam.tilingKey = 0;
    OP_CHECK_IF(refShape != valueShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "ref, value",
            "ref_shape, value_shape",
            "The shapes of ref and value must be the same."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(TensorMoveTilingData& tilingData, const AssignTilingParam& tilingParam)
{
    tilingData.set_totalCoreNum(tilingParam.totalCoreNum);
    tilingData.set_usedCoreNum(tilingParam.usedCoreNum);
    tilingData.set_ubFactor(tilingParam.ubFactor);
    tilingData.set_tailBlockTailUbFactor(tilingParam.tailBlockTailUbFactor);
    tilingData.set_blockFactor(tilingParam.blockFactor);
    tilingData.set_tailBlockFactor(tilingParam.tailBlockFactor);
    tilingData.set_tilingKey(tilingParam.tilingKey);
}

static void PrintTilingData(const gert::TilingContext *context, TensorMoveTilingData &tilingData)
{
    OP_LOGI(context->GetNodeName(), "Assign tilingData: totalCoreNum:%ld, usedCoreNum:%ld, ubFactor:%ld, tailBlockTailUbFactor:%ld, "
        "blockFactor:%ld, tailBlockFactor:%ld, tilingKey:%ld ", tilingData.get_totalCoreNum(), tilingData.get_usedCoreNum(),
        tilingData.get_ubFactor(), tilingData.get_tailBlockTailUbFactor(), tilingData.get_blockFactor(),
        tilingData.get_tailBlockFactor(), tilingData.get_tilingKey());
}

static void CalcBlockFactor(AssignTilingParam &tilingParam, int64_t numel)
{
    tilingParam.uo = CeilDiv(numel, tilingParam.ubFactor);
    tilingParam.tailBlockTailUbFactor = GetRemainder(numel, tilingParam.ubFactor);

    int64_t coreData = CeilDiv(tilingParam.uo, tilingParam.totalCoreNum);
    tilingParam.usedCoreNum = CeilDiv(tilingParam.uo, coreData);
    tilingParam.blockFactor = CeilDiv(tilingParam.uo, tilingParam.usedCoreNum);
    tilingParam.tailBlockFactor = tilingParam.uo - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;
    if (tilingParam.tailBlockTailUbFactor == 0) {
        tilingParam.tailBlockTailUbFactor = tilingParam.ubFactor;
    }
}

static ge::graphStatus DoTiling(const gert::TilingContext* context, AssignTilingParam& tilingParam)
{
    auto xShapePtr = context->GetInputShape(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    int64_t numel = xShape.GetShapeSize();
    // 获取输入数据类型所占的字节数
    auto inputXPtr = context->GetInputDesc(INDEX_INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXPtr);
    auto dtype = inputXPtr->GetDataType();
    tilingParam.bytesForOneData = ge::GetSizeByDataType(dtype);

    int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / tilingParam.bytesForOneData;
    // 计算ubFactor
    if (numel >= maxUbAvailable) {
        tilingParam.ubFactor = maxUbAvailable;
    } else {
        tilingParam.ubFactor = numel;
    }
    CalcBlockFactor(tilingParam, numel);
    if (tilingParam.usedCoreNum == tilingParam.totalCoreNum || tilingParam.blockFactor > 1) {
        return ge::GRAPH_SUCCESS;
    }
    // 当使用核数不满并且每个核只有一次循环时可以进行调整，增大使用核数，来提高性能
    if (GetRemainder(numel, tilingParam.totalCoreNum) == 0) {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum);
    } else {
        tilingParam.ubFactor = FloorDiv(numel, tilingParam.totalCoreNum - 1);
    }
    tilingParam.ubFactor = CeilAlign(tilingParam.ubFactor, ONE_BLK_BYTE / tilingParam.bytesForOneData);
    int64_t ubFactorMin = UB_FACTOR_MIN_BETY / tilingParam.bytesForOneData;
    tilingParam.ubFactor = tilingParam.ubFactor < ubFactorMin ? ubFactorMin : tilingParam.ubFactor;
    CalcBlockFactor(tilingParam, numel);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AssignTilingForAscendC(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "AssignTilingForAscendC running begin.");

    OP_CHECK_IF(CheckDtypeForAssign(context) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "The dtype check failed."),
        return ge::GRAPH_FAILED);

    AssignTilingParam tilingParam;
    OP_CHECK_IF(CheckShapeForAssign(context, tilingParam) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "The shape check failed."), return ge::GRAPH_FAILED);

    auto compileInfo = reinterpret_cast<const AssignCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    tilingParam.totalCoreNum = compileInfo->coreNum;
    tilingParam.ubSize = compileInfo->ubSize;

    OP_CHECK_IF(DoTiling(context, tilingParam) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "Dotiling failed."),
        return ge::GRAPH_FAILED);

    // tilingkey由数据类型所占字节表示(1/2/4/8)
    tilingParam.tilingKey += tilingParam.bytesForOneData;

    TensorMoveTilingData tilingData;
    SetTilingData(tilingData, tilingParam);
    OP_CHECK_IF(tilingData.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
        OP_LOGE(context->GetNodeName(), "set tiling data fail."), return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(tilingData.get_usedCoreNum());
    context->SetTilingKey(tilingData.get_tilingKey());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WORKSPACE_SIZE;

    PrintTilingData(context, tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Assign(gert::TilingContext* context)
{
    return AssignTilingForAscendC(context);
}

static ge::graphStatus TilingPrepare4Assign(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareAssignForAscendC entering.");
    auto compileInfo = context->GetCompiledInfo<AssignCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the Assign op.
IMPL_OP_OPTILING(Assign).Tiling(Tiling4Assign).TilingParse<AssignCompileInfo>(TilingPrepare4Assign);
} // namespace optiling
