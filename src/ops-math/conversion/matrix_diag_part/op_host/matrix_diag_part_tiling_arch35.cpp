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
 * \file matrix_diag_part_tiling_arch35.cpp
 * \brief Implemention of MatrixDiagPart tiling
 */

#include "matrix_diag_part_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

constexpr int64_t PER_CORE_MIN = 1024;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

struct MatrixDiagPartCompileInfo {};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInputAndComputeDims(gert::TilingContext* context, int64_t& d, int64_t& n,
                                                   int64_t& matrixSize, int64_t& totalOutputElements)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    auto rank = inputShapeX.GetDimNum();

    if (rank < 2) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                  "input rank must be >= 2");
        return ge::GRAPH_FAILED;
    }

    if (rank > 8) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                  "input rank must be <= 8");
        return ge::GRAPH_FAILED;
    }

    int64_t M = inputShapeX.GetDim(rank - 2);
    n = inputShapeX.GetDim(rank - 1);
    d = std::min(M, n);
    matrixSize = M * n;
    int64_t totalInputElements = inputShapeX.GetShapeSize();
    int64_t batchTotal = (matrixSize > 0) ? (totalInputElements / matrixSize) : 0;
    totalOutputElements = batchTotal * d;

    if (d < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "diagLen", std::to_string(d).c_str(),
                                              "diagLen must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8,
                                                   ge::DT_UINT8};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                              "only support float16, float32, int32, int8, uint8");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetupMemoryAndWorkspace(gert::TilingContext* context, uint64_t ubSize)
{
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MatrixDiagPartTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "Enter TilingMatrixDiagPart");
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t d = 0;
    int64_t n = 0;
    int64_t matrixSize = 0;
    int64_t totalOutputElements = 0;
    OP_CHECK_IF(ValidateInputAndComputeDims(context, d, n, matrixSize, totalOutputElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputAndComputeDims error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "CheckDtype error"),
                return ge::GRAPH_FAILED);

    MatrixDiagPartTilingData* tiling = context->GetTilingData<MatrixDiagPartTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(MatrixDiagPartTilingData), 0, sizeof(MatrixDiagPartTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalOutputElements = totalOutputElements;
    tiling->diagLen = d;
    tiling->inputRowStride = n + 1;
    tiling->matrixSize = matrixSize;

    int64_t perCoreElements = Ops::Base::CeilDiv(totalOutputElements, coreNum);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = (totalOutputElements == 0) ? 1 : Ops::Base::CeilDiv(totalOutputElements, perCoreElements);
    context->SetBlockDim(needCoreNum);

    OP_CHECK_IF(SetupMemoryAndWorkspace(context, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetupMemoryAndWorkspace error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = GET_TPL_TILING_KEY(MODE_DEFAULT);
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMatrixDiagPart([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatrixDiagPart)
    .Tiling(MatrixDiagPartTilingFunc)
    .TilingParse<MatrixDiagPartCompileInfo>(TilingParseForMatrixDiagPart);
} // namespace optiling
