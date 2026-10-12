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
 * \file square_sum_all_tiling.cpp
 * \brief SquareSumAll tiling for Ascend 950 (DAV3510 / arch35).
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/square_sum_all_tiling_data.h"
#include "../../op_kernel/arch35/square_sum_all_tiling_key.h"

namespace optiling {
namespace {
constexpr size_t X1_INDEX = 0;
constexpr size_t X2_INDEX = 1;
constexpr size_t Y1_INDEX = 0;
constexpr size_t Y2_INDEX = 1;
constexpr int64_t MIN_RANK = 0;
constexpr int64_t MAX_RANK = 8;
constexpr int64_t ELEMENT_BYTES = sizeof(float);
constexpr int64_t VECTOR_ELEMENTS = 64;
constexpr int64_t VECTOR_BYTES = VECTOR_ELEMENTS * ELEMENT_BYTES;
constexpr int64_t MAX_TILE_ELEMENTS = 4096;
constexpr int64_t INPUT_QUEUE_COUNT = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UB_RESERVED_BYTES = 8 * 1024;
constexpr int64_t RESULT_LOCAL_BYTES = 64;
constexpr int64_t WORKSPACE_SLOT_BYTES = 32;
constexpr int64_t WORKSPACE_REGIONS = 2;
constexpr int64_t GPU_ALIGNED_MIN_ELEMENTS = 8192;
constexpr int64_t GPU_BLOCK_ELEMENTS = 128;
constexpr int64_t GPU_MAX_PARTIALS = 1728;
constexpr int64_t GPU_RESULT_LOCAL_BYTES = 2 * VECTOR_BYTES;
constexpr int64_t GPU_PARTIAL_LOCAL_BYTES_PER_ELEMENT = WORKSPACE_SLOT_BYTES;
constexpr int64_t GPU_INPUT_QUEUE_DEPTH = 2;
constexpr int64_t GPU_ACCUMULATOR_REGION_COUNT = 4;
constexpr int64_t GPU_BATCH_RESULT_BYTES_PER_PARTIAL = WORKSPACE_REGIONS * WORKSPACE_SLOT_BYTES;
constexpr int64_t GPU_BATCH_LOCAL_BYTES_PER_PARTIAL = INPUT_QUEUE_COUNT * GPU_INPUT_QUEUE_DEPTH * GPU_BLOCK_ELEMENTS *
                                                          ELEMENT_BYTES +
                                                      GPU_ACCUMULATOR_REGION_COUNT * VECTOR_ELEMENTS * ELEMENT_BYTES +
                                                      GPU_BATCH_RESULT_BYTES_PER_PARTIAL;
constexpr int64_t GPU_UB_RESERVED_BYTES = 32 * 1024;
constexpr int64_t GPU_TPIPE_BUDGET_BYTES = 224 * 1024;
constexpr int64_t MAX_VECTOR_CHUNKS = std::numeric_limits<uint16_t>::max();
constexpr uint64_t LEGACY_TILING_KEY = 0;
constexpr uint64_t GPU_ALIGNED_TILING_KEY = 1;
constexpr size_t WORKSPACE_COUNT = 1;

struct SquareSumAllCompileInfo {};

struct KernelPlan {
    int64_t usedCoreNum = 0;
    int64_t tileElements = 0;
    int64_t userWorkspaceBytes = 0;
    int64_t ubRequirementBytes = 0;
    uint64_t tilingKey = LEGACY_TILING_KEY;
};

int64_t CeilDivPositive(int64_t value, int64_t divisor) { return value / divisor + (value % divisor != 0); }

// 私有格式未在 Ascend 950 OpDef 中注册；这里保留显式拒绝，防止异常描述绕过注册层。
bool IsPrivateFormat(ge::Format format)
{
    return format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_C1HWNCoC0;
}

bool IsNdFormat(ge::Format format) { return format == ge::FORMAT_ND; }

bool IsFourDimPublicInputFormat(ge::Format format) { return format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC; }

bool IsSupportedFormatTuple(ge::Format x1Format, ge::Format x2Format, ge::Format y1Format, ge::Format y2Format)
{
    if (x1Format != x2Format || y1Format != y2Format) {
        return false;
    }
    return (IsNdFormat(x1Format) && y1Format == ge::FORMAT_ND) ||
           (IsFourDimPublicInputFormat(x1Format) && y1Format == ge::FORMAT_ND);
}

bool IsScalarStorageShape(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 || (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

ge::graphStatus ValidateDtypeAndFormat(gert::TilingContext* context, ge::Format& inputFormat)
{
    const gert::CompileTimeTensorDesc* x1Desc = context->GetInputDesc(X1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    const gert::CompileTimeTensorDesc* x2Desc = context->GetInputDesc(X2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    const gert::CompileTimeTensorDesc* y1Desc = context->GetOutputDesc(Y1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Desc);
    const gert::CompileTimeTensorDesc* y2Desc = context->GetOutputDesc(Y2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Desc);

    const std::string actualDtypes = std::to_string(static_cast<int32_t>(x1Desc->GetDataType())) + ", " +
                                     std::to_string(static_cast<int32_t>(x2Desc->GetDataType())) + ", " +
                                     std::to_string(static_cast<int32_t>(y1Desc->GetDataType())) + ", " +
                                     std::to_string(static_cast<int32_t>(y2Desc->GetDataType()));
    OP_CHECK_IF(x1Desc->GetDataType() != ge::DT_FLOAT || x2Desc->GetDataType() != ge::DT_FLOAT ||
                    y1Desc->GetDataType() != ge::DT_FLOAT || y2Desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x1, x2, y1, y2", actualDtypes,
                                                       "all inputs and outputs must use float32"),
                return ge::GRAPH_FAILED);

    const ge::Format x1Format = x1Desc->GetStorageFormat();
    const ge::Format x2Format = x2Desc->GetStorageFormat();
    const ge::Format y1Format = y1Desc->GetStorageFormat();
    const ge::Format y2Format = y2Desc->GetStorageFormat();
    const std::string actualFormats = std::to_string(static_cast<int32_t>(x1Format)) + ", " +
                                      std::to_string(static_cast<int32_t>(x2Format)) + ", " +
                                      std::to_string(static_cast<int32_t>(y1Format)) + ", " +
                                      std::to_string(static_cast<int32_t>(y2Format));
    OP_CHECK_IF(
        IsPrivateFormat(x1Format) || IsPrivateFormat(x2Format) || IsPrivateFormat(y1Format) ||
            IsPrivateFormat(y2Format),
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context->GetNodeName(), "x1, x2, y1, y2", actualFormats,
                                                "FRACTAL_Z, C1HWNCoC0 and NC1HWC0 are not supported on Ascend 950"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsSupportedFormatTuple(x1Format, x2Format, y1Format, y2Format),
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context->GetNodeName(), "x1, x2, y1, y2", actualFormats,
                                                        "supported formats are ND, NCHW and NHWC; NCHW/NHWC inputs "
                                                        "map to ND outputs"),
                return ge::GRAPH_FAILED);
    inputFormat = x1Format;
    return ge::GRAPH_SUCCESS;
}

// 逐维校验并累乘元素数。单独抽出来是为了让调用方 ValidateShapeAndGetSize 的非空非注释
// 行数留在 CodeCheck 的 50 行阈值内——这些 OP_LOGE_FOR_INVALID_* 宏每处就占 3 行。
ge::graphStatus AccumulateAndValidateDims(gert::TilingContext* context, const gert::Shape& x1Shape,
                                          const gert::Shape& x2Shape, int64_t rank, int64_t& totalElements)
{
    const std::string inputShapes = Ops::Base::ToString(x1Shape) + ", " + Ops::Base::ToString(x2Shape);

    totalElements = 1;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = x1Shape.GetDim(static_cast<size_t>(i));
        const int64_t x2Dim = x2Shape.GetDim(static_cast<size_t>(i));
        OP_CHECK_IF(x1Dim <= 0 || x2Dim <= 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                           "runtime dimensions must all be greater than zero"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(x1Dim != x2Dim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                           "runtime storage shapes must be identical"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(totalElements > std::numeric_limits<int64_t>::max() / x1Dim,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "x1, x2", inputShapes,
                                                              "input element count overflows int64"),
                    return ge::GRAPH_FAILED);
        totalElements *= x1Dim;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateShapeAndGetSize(gert::TilingContext* context, ge::Format inputFormat, int64_t& totalElements)
{
    const gert::StorageShape* x1StorageShape = context->GetInputShape(X1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1StorageShape);
    const gert::StorageShape* x2StorageShape = context->GetInputShape(X2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2StorageShape);
    const gert::StorageShape* y1StorageShape = context->GetOutputShape(Y1_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1StorageShape);
    const gert::StorageShape* y2StorageShape = context->GetOutputShape(Y2_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2StorageShape);
    const gert::Shape& x1Shape = x1StorageShape->GetStorageShape();
    const gert::Shape& x2Shape = x2StorageShape->GetStorageShape();
    const gert::Shape& y1Shape = y1StorageShape->GetStorageShape();
    const gert::Shape& y2Shape = y2StorageShape->GetStorageShape();

    const int64_t x1Rank = static_cast<int64_t>(x1Shape.GetDimNum());
    const int64_t x2Rank = static_cast<int64_t>(x2Shape.GetDimNum());
    const std::string actualRanks = std::to_string(x1Rank) + ", " + std::to_string(x2Rank);
    OP_CHECK_IF(x1Rank != x2Rank,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x1, x2", actualRanks,
                                                          "input ranks must be equal"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(x1Rank < MIN_RANK || x1Rank > MAX_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x1, x2", std::to_string(x1Rank),
                                                         "input rank must be in [0, 8]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(IsFourDimPublicInputFormat(inputFormat) && x1Rank != 4,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x1, x2", std::to_string(x1Rank),
                                                         "NCHW and NHWC inputs must have rank 4"),
                return ge::GRAPH_FAILED);

    // helper 已在失败处打过 ERROR，这里只透传状态，不重复记日志。
    const ge::graphStatus dimStatus = AccumulateAndValidateDims(context, x1Shape, x2Shape, x1Rank, totalElements);
    if (dimStatus != ge::GRAPH_SUCCESS) {
        return dimStatus;
    }

    const std::string outputShapes = Ops::Base::ToString(y1Shape) + ", " + Ops::Base::ToString(y2Shape);
    OP_CHECK_IF(!IsScalarStorageShape(y1Shape) || !IsScalarStorageShape(y2Shape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "y1, y2", outputShapes,
                                                       "each output must contain exactly one scalar element"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum, int64_t& ubSize,
                                uint64_t& systemWorkspaceSize)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "invalid AIV core count: %ld", coreNum), return ge::GRAPH_FAILED);

    uint64_t platformUbSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    OP_CHECK_IF(platformUbSize == 0 || platformUbSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context, "invalid UB size: %lu", platformUbSize), return ge::GRAPH_FAILED);
    ubSize = static_cast<int64_t>(platformUbSize);
    systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

bool TryBuildGpuAlignedPlan(int64_t totalElements, int64_t coreNum, int64_t ubSize, KernelPlan& plan)
{
    if (totalElements < GPU_ALIGNED_MIN_ELEMENTS) {
        return false;
    }
    const int64_t partialCount = std::min(CeilDivPositive(totalElements, GPU_BLOCK_ELEMENTS), GPU_MAX_PARTIALS);
    const int64_t gridStride = partialCount * GPU_BLOCK_ELEMENTS;
    const int64_t chunkCount = CeilDivPositive(totalElements, gridStride);
    if (chunkCount > MAX_VECTOR_CHUNKS) {
        return false;
    }
    const int64_t requestedCores = std::max<int64_t>(1, totalElements / MAX_TILE_ELEMENTS);
    const int64_t usedCoreNum = std::min({coreNum, partialCount, requestedCores});
    const int64_t maxCorePartialCount = CeilDivPositive(partialCount, usedCoreNum);
    const int64_t packedPartials = CeilDivPositive(partialCount, GPU_BLOCK_ELEMENTS) * GPU_BLOCK_ELEMENTS;
    const int64_t fixedTpipeBytes = GPU_RESULT_LOCAL_BYTES + packedPartials * GPU_PARTIAL_LOCAL_BYTES_PER_ELEMENT;
    if (fixedTpipeBytes >= GPU_TPIPE_BUDGET_BYTES) {
        return false;
    }
    const int64_t batchCapacityByTpipe = (GPU_TPIPE_BUDGET_BYTES - fixedTpipeBytes) / GPU_BATCH_LOCAL_BYTES_PER_PARTIAL;
    if (batchCapacityByTpipe <= 0) {
        return false;
    }
    const int64_t batchPartialCapacity = std::min(batchCapacityByTpipe, maxCorePartialCount);
    const int64_t ubRequirementBytes = GPU_UB_RESERVED_BYTES + fixedTpipeBytes +
                                       batchPartialCapacity * GPU_BATCH_LOCAL_BYTES_PER_PARTIAL;
    if (ubSize < ubRequirementBytes) {
        return false;
    }

    plan.usedCoreNum = usedCoreNum;
    plan.tileElements = MAX_TILE_ELEMENTS;
    plan.userWorkspaceBytes = partialCount * WORKSPACE_SLOT_BYTES * WORKSPACE_REGIONS;
    plan.ubRequirementBytes = ubRequirementBytes;
    plan.tilingKey = GPU_ALIGNED_TILING_KEY;
    return true;
}

ge::graphStatus BuildLegacyPlan(gert::TilingContext* context, int64_t totalElements, int64_t coreNum, int64_t ubSize,
                                KernelPlan& plan)
{
    plan.usedCoreNum = std::min(std::max<int64_t>(1, totalElements / MAX_TILE_ELEMENTS), coreNum);
    const int64_t unalignedMergeLocalBytes = plan.usedCoreNum * WORKSPACE_SLOT_BYTES;
    const int64_t mergeLocalBytes = (unalignedMergeLocalBytes / VECTOR_BYTES +
                                     (unalignedMergeLocalBytes % VECTOR_BYTES != 0)) *
                                    VECTOR_BYTES;
    const int64_t fixedUbBytes = UB_RESERVED_BYTES + RESULT_LOCAL_BYTES + mergeLocalBytes;
    OP_CHECK_IF(ubSize <= fixedUbBytes,
                OP_LOGE(context, "UB is too small: available=%ld, fixed requirement=%ld", ubSize, fixedUbBytes),
                return ge::GRAPH_FAILED);
    const int64_t bytesPerTileElement = INPUT_QUEUE_COUNT * DOUBLE_BUFFER * ELEMENT_BYTES;
    plan.tileElements = std::min((ubSize - fixedUbBytes) / bytesPerTileElement, MAX_TILE_ELEMENTS);
    plan.tileElements = plan.tileElements / VECTOR_ELEMENTS * VECTOR_ELEMENTS;
    OP_CHECK_IF(plan.tileElements < VECTOR_ELEMENTS,
                OP_LOGE(context, "UB cannot hold one float32 vector tile: tileElements=%ld", plan.tileElements),
                return ge::GRAPH_FAILED);
    plan.userWorkspaceBytes = plan.usedCoreNum * WORKSPACE_SLOT_BYTES * WORKSPACE_REGIONS;
    plan.ubRequirementBytes = fixedUbBytes + plan.tileElements * bytesPerTileElement;
    plan.tilingKey = LEGACY_TILING_KEY;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WriteTilingResult(gert::TilingContext* context, int64_t totalElements, int64_t ubSize,
                                  uint64_t systemWorkspaceSize, const KernelPlan& plan)
{
    SquareSumAllTilingData* tilingData = context->GetTilingData<SquareSumAllTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->totalElements = totalElements;
    tilingData->usedCoreNum = plan.usedCoreNum;
    tilingData->baseCoreElements = totalElements / plan.usedCoreNum;
    tilingData->extraCoreCount = totalElements % plan.usedCoreNum;
    tilingData->tileElements = plan.tileElements;
    const uint64_t userWorkspaceSize = static_cast<uint64_t>(plan.userWorkspaceBytes);
    OP_CHECK_IF(systemWorkspaceSize > std::numeric_limits<size_t>::max() - userWorkspaceSize,
                OP_LOGE(context, "workspace size overflows size_t"), return ge::GRAPH_FAILED);
    size_t* workspaceSizes = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(systemWorkspaceSize + userWorkspaceSize);

    OP_CHECK_IF(context->SetBlockDim(static_cast<uint32_t>(plan.usedCoreNum)) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetBlockDim failed for %ld cores", plan.usedCoreNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetScheduleMode failed"),
                return ge::GRAPH_FAILED);
    ASCENDC_TPL_SEL_PARAM(context, plan.tilingKey);

    OP_LOGI(context,
            "SquareSumAll tiling: N=%ld, key=%lu, cores=%ld, base=%ld, extra=%ld, tile=%ld, ub=%ld, "
            "ubRequirement=%ld, workspace=%zu",
            totalElements, plan.tilingKey, plan.usedCoreNum, tilingData->baseCoreElements, tilingData->extraCoreCount,
            plan.tileElements, ubSize, plan.ubRequirementBytes, workspaceSizes[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTiling(gert::TilingContext* context, int64_t totalElements, int64_t coreNum, int64_t ubSize,
                           uint64_t systemWorkspaceSize)
{
    KernelPlan plan;
    if (!TryBuildGpuAlignedPlan(totalElements, coreNum, ubSize, plan)) {
        OP_CHECK_IF(BuildLegacyPlan(context, totalElements, coreNum, ubSize, plan) != ge::GRAPH_SUCCESS,
                    OP_LOGD(context->GetNodeName(), "legacy kernel plan failed"), return ge::GRAPH_FAILED);
    }
    return WriteTilingResult(context, totalElements, ubSize, systemWorkspaceSize, plan);
}

ge::graphStatus TilingForSquareSumAll(gert::TilingContext* context)
{
    const SquareSumAllCompileInfo* compileInfo = context->GetCompileInfo<SquareSumAllCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // The helpers below already report the failure at ERROR level with the offending
    // dtypes/formats/shapes via OP_LOGE_FOR_INVALID_*; a second ERROR line here would
    // only repeat it with less detail, so these stay DEBUG breadcrumbs.
    ge::Format inputFormat = ge::FORMAT_RESERVED;
    OP_CHECK_IF(ValidateDtypeAndFormat(context, inputFormat) != ge::GRAPH_SUCCESS,
                OP_LOGD(context->GetNodeName(), "dtype or format validation failed"), return ge::GRAPH_FAILED);
    int64_t totalElements = 0;
    OP_CHECK_IF(ValidateShapeAndGetSize(context, inputFormat, totalElements) != ge::GRAPH_SUCCESS,
                OP_LOGD(context->GetNodeName(), "shape validation failed"), return ge::GRAPH_FAILED);
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    uint64_t systemWorkspaceSize = 0;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum, ubSize, systemWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGD(context->GetNodeName(), "platform query failed"), return ge::GRAPH_FAILED);
    return FillTiling(context, totalElements, coreNum, ubSize, systemWorkspaceSize);
}

ge::graphStatus TilingParseForSquareSumAll([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(SquareSumAll)
    .Tiling(TilingForSquareSumAll)
    .TilingParse<SquareSumAllCompileInfo>(TilingParseForSquareSumAll);
} // namespace optiling
