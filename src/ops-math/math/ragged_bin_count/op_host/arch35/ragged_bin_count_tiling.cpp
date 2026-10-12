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
 * \file ragged_bin_count_tiling.cpp
 * \brief Tiling implementation for RaggedBinCount on DAV_3510.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "log/log.h"
#include "op_host/math_tiling_templates_registry.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/const_util.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/ragged_bin_count_precision_policy.h"
#include "../../op_kernel/arch35/ragged_bin_count_tiling_data.h"
#include "../../op_kernel/arch35/ragged_bin_count_tiling_key.h"

namespace optiling {
namespace {
using namespace Ops::Math::OpTiling;

constexpr int64_t INPUT_SPLITS = 0;
constexpr int64_t INPUT_VALUES = 1;
constexpr int64_t INPUT_SIZE = 2;
constexpr int64_t INPUT_WEIGHTS = 3;
constexpr int64_t OUTPUT_RESULT = 0;
constexpr int64_t ATTR_BINARY_OUTPUT = 0;

constexpr size_t RANK_SPLITS = 1U;
constexpr size_t MIN_VALUES_RANK = 1U;
constexpr size_t MAX_VALUES_RANK = 2U;
constexpr size_t MIN_WEIGHTS_RANK = 0U;
constexpr size_t MAX_WEIGHTS_RANK = 2U;
constexpr size_t RANK_SIZE = 1U;
constexpr size_t RANK_OUTPUT = 2U;
constexpr int64_t MIN_SPLITS_NUM = 2;
constexpr int64_t PER_CORE_MIN_ELEMENTS = 1024;
constexpr int64_t MIN_CORE_NUM = 1;
constexpr uint32_t DCACHE_SIZE = 128U * 1024U;
constexpr uint32_t STATIC_UB_ESTIMATE = 0U;
constexpr size_t OUTPUT_ELEMENT_BYTES = sizeof(float);
constexpr uint32_t SCHEDULE_MODE_SYNC_ALL = 1U;

constexpr uint32_t MAPPING_MODE_ROW = 0U;
constexpr uint32_t MAPPING_MODE_VALUE = 1U;
// The schMode shifts are not redeclared here: they live in ragged_bin_count_tiling_data.h as
// RBC_MAPPING_MODE_SHIFT / RBC_BINARY_OUTPUT_SHIFT so this encoder and the kernel-entry decoder read
// from one definition.

// How much extra write-back traffic the privatised path may buy with one saved global atomic.
//
// Privatising turns `numValues` scattered global atomics into `numValues` UB atomics plus one bulk
// write-back of `outputElements` floats *per participating core*. The write-back is a contiguous
// atomic DMA, so it costs far less per element than a scattered atomic under contention; a factor of
// eight keeps the trade conservative while still admitting every case the long tail is made of
// (their `outputElements * cores` is orders of magnitude below `numValues`).
constexpr int64_t PRIVATE_WRITEBACK_FACTOR = 8;
constexpr uint64_t UB_ALIGN_BYTES = 32U;

struct RaggedBinCountCompileInfo {};

ge::graphStatus CheckRankInRange(const gert::TilingContext* context, const gert::StorageShape* shape, size_t minRank,
                                 size_t maxRank, const char* inputName)
{
    const size_t rank = shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(rank < minRank || rank > maxRank,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), inputName, Ops::Base::ToString(shape->GetStorageShape()),
                    "rank must be in [" + std::to_string(minRank) + ", " + std::to_string(maxRank) + "], but got " +
                        std::to_string(rank)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t SafeCeilDiv(int64_t dividend, int64_t divisor)
{
    return dividend / divisor + static_cast<int64_t>(dividend % divisor != 0);
}

int64_t GetRequiredCoreNum(int64_t workload, int64_t maxCoreNum)
{
    if (workload <= 0) {
        return MIN_CORE_NUM;
    }
    int64_t perCoreElements = SafeCeilDiv(workload, maxCoreNum);
    perCoreElements = std::max(perCoreElements, PER_CORE_MIN_ELEMENTS);
    return std::min(maxCoreNum, SafeCeilDiv(workload, perCoreElements));
}

ge::graphStatus GetSafeElementCount(const gert::TilingContext* context, const gert::StorageShape* storageShape,
                                    const char* inputName, int64_t& elementCount)
{
    const gert::Shape& shape = storageShape->GetStorageShape();
    elementCount = 1;
    for (size_t dimIndex = 0U; dimIndex < shape.GetDimNum(); ++dimIndex) {
        const int64_t dim = shape.GetDim(dimIndex);
        OP_CHECK_IF(dim < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), inputName, Ops::Base::ToString(shape),
                                                          "shape must be concrete and non-negative at tiling time"),
                    return ge::GRAPH_FAILED);
        if (dim == 0 || elementCount == 0) {
            elementCount = 0;
            continue;
        }
        OP_CHECK_IF(elementCount > std::numeric_limits<int64_t>::max() / dim,
                    OP_LOGE(context->GetNodeName(), "%s element count overflows int64.", inputName),
                    return ge::GRAPH_FAILED);
        elementCount *= dim;
    }
    return ge::GRAPH_SUCCESS;
}

// The built-in GE verifier caps weights at rank 2.  Within that public rank range, canndev validates
// weights by element count alone (op_proto/runtime/bincount_ops.cc:104-107), and the kernel addresses
// weights linearly through the flattened values order.  Thus [0, 3] and [2, 0] are both empty, while
// a flat [6] may pair with [2, 3] values.  GetSafeElementCount also rejects negative dimensions and
// guards the multiply against int64 overflow.
ge::graphStatus CheckWeightsShape(const gert::TilingContext* context, const gert::StorageShape* valuesShape,
                                  const gert::StorageShape* weightsShape, bool& hasWeights)
{
    const gert::Shape& values = valuesShape->GetStorageShape();
    const gert::Shape& weights = weightsShape->GetStorageShape();
    OP_CHECK_IF(
        CheckRankInRange(context, weightsShape, MIN_WEIGHTS_RANK, MAX_WEIGHTS_RANK, "weights") != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "weights", Ops::Base::ToString(weights),
                                              "weights rank must not exceed the GE verifier limit"),
        return ge::GRAPH_FAILED);
    int64_t valuesCount = 0;
    int64_t weightsCount = 0;
    OP_CHECK_IF(GetSafeElementCount(context, valuesShape, "values", valuesCount) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "values", Ops::Base::ToString(values),
                                                      "values element count validation failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetSafeElementCount(context, weightsShape, "weights", weightsCount) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "weights", Ops::Base::ToString(weights),
                                                      "weights element count validation failed"),
                return ge::GRAPH_FAILED);

    // Zero elements is the contract for "no weights"; every bin then accumulates 1.0 per value.
    if (weightsCount == 0) {
        hasWeights = false;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(
        weightsCount != valuesCount,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "weights, values",
                                               Ops::Base::ToString(weights) + ", " + Ops::Base::ToString(values),
                                               "non-empty weights must have the same element count as values, got " +
                                                   std::to_string(weightsCount) + " vs " + std::to_string(valuesCount)),
        return ge::GRAPH_FAILED);
    hasWeights = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDataTypes(gert::TilingContext* context)
{
    const gert::CompileTimeTensorDesc* splitsDesc = context->GetInputDesc(INPUT_SPLITS);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitsDesc);
    const gert::CompileTimeTensorDesc* valuesDesc = context->GetInputDesc(INPUT_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context, valuesDesc);
    const gert::CompileTimeTensorDesc* sizeDesc = context->GetInputDesc(INPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, sizeDesc);
    const gert::CompileTimeTensorDesc* weightsDesc = context->GetInputDesc(INPUT_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightsDesc);
    const gert::CompileTimeTensorDesc* outputDesc = context->GetOutputDesc(OUTPUT_RESULT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    const ge::DataType valuesDtype = valuesDesc->GetDataType();
    const ge::DataType sizeDtype = sizeDesc->GetDataType();
    OP_CHECK_IF(splitsDesc->GetDataType() != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "splits",
                                          Ops::Base::ToString(splitsDesc->GetDataType()), "int64"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        valuesDtype != ge::DT_INT32 && valuesDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "values", Ops::Base::ToString(valuesDtype), "int32 or int64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        sizeDtype != valuesDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "size, values",
                                               Ops::Base::ToString(sizeDtype) + ", " + Ops::Base::ToString(valuesDtype),
                                               "size and values must have the same dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(weightsDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "weights",
                                          Ops::Base::ToString(weightsDesc->GetDataType()), "float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "output",
                                          Ops::Base::ToString(outputDesc->GetDataType()), "float32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// The five shapes the tiling reads, so fetching and null-checking them is one call instead of ten lines.
struct InputShapes {
    const gert::StorageShape* splits = nullptr;
    const gert::StorageShape* values = nullptr;
    const gert::StorageShape* size = nullptr;
    const gert::StorageShape* weights = nullptr;
    const gert::StorageShape* output = nullptr;
};

// Everything the schedule and the kernel are sized from, derived from the shapes plus the const `size`.
struct ProblemSize {
    int64_t numSplits = 0;
    int64_t numRows = 0;
    int64_t numValues = 0;
    int64_t numBins = 0;
    int64_t outputElements = 0;
    bool hasWeights = false;
};

// The platform limits the schedule decisions are taken against.
struct PlatformBudget {
    int64_t maxCoreNum = 0;
    uint64_t localMemorySize = 0U;
    uint64_t systemWorkspaceSize = 0U;
};

// The decisions: how many cores, which mapping, whether to privatise, and the resulting tiling key.
struct ScheduleParams {
    int64_t coreNum = MIN_CORE_NUM;
    uint32_t mappingMode = MAPPING_MODE_ROW;
    bool binaryOutput = false;
    uint32_t privateHistElems = 0U;
    uint32_t schMode = 0U;
};

ge::graphStatus FetchInputShapes(gert::TilingContext* context, InputShapes& shapes)
{
    shapes.splits = context->GetInputShape(INPUT_SPLITS);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.splits);
    shapes.values = context->GetInputShape(INPUT_VALUES);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.values);
    shapes.size = context->GetInputShape(INPUT_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.size);
    shapes.weights = context->GetInputShape(INPUT_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.weights);
    shapes.output = context->GetOutputShape(OUTPUT_RESULT);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapes.output);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeRanks(const gert::TilingContext* context, const InputShapes& shapes)
{
    OP_CHECK_IF(shapes.splits->GetStorageShape().GetDimNum() != RANK_SPLITS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "splits",
                                                      Ops::Base::ToString(shapes.splits->GetStorageShape()),
                                                      "splits must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        CheckRankInRange(context, shapes.values, MIN_VALUES_RANK, MAX_VALUES_RANK, "values") != ge::GRAPH_SUCCESS,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "values",
                                              Ops::Base::ToString(shapes.values->GetStorageShape()),
                                              "values rank validation failed"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        shapes.size->GetStorageShape().GetDimNum() != RANK_SIZE || shapes.size->GetStorageShape().GetDim(0U) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "size",
                                              Ops::Base::ToString(shapes.size->GetStorageShape()),
                                              "size must have exact shape [1]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(shapes.output->GetStorageShape().GetDimNum() != RANK_OUTPUT,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "output",
                                                      Ops::Base::ToString(shapes.output->GetStorageShape()),
                                                      "output must be 2D"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// splits length, values element count, weights presence and the const `size` -- everything that comes
// straight off an input. The output extent it implies is checked separately in ResolveOutputExtent.
ge::graphStatus ResolveCounts(gert::TilingContext* context, const InputShapes& shapes, ProblemSize& problem)
{
    problem.numSplits = shapes.splits->GetStorageShape().GetDim(0U);
    OP_CHECK_IF(problem.numSplits < MIN_SPLITS_NUM,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "splits", Ops::Base::ToString(shapes.splits->GetStorageShape()),
                    "splits must contain at least two elements, but got " + std::to_string(problem.numSplits)),
                return ge::GRAPH_FAILED);
    problem.numRows = problem.numSplits - 1;

    OP_CHECK_IF(GetSafeElementCount(context, shapes.values, "values", problem.numValues) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to calculate the values element count safely."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightsShape(context, shapes.values, shapes.weights, problem.hasWeights) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "weights",
                                                      Ops::Base::ToString(shapes.weights->GetStorageShape()),
                                                      "weights shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!Ops::Base::GetConstInt(context, INPUT_SIZE, problem.numBins),
                OP_LOGE(context->GetNodeName(), "Failed to get the value-dependent input size."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(problem.numBins < 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "size", std::to_string(problem.numBins),
                                                      "size must be non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// numRows * numBins, guarded against int64 and size_t overflow before it is multiplied out, then
// cross-checked against the shape GE actually allocated the output with.
ge::graphStatus ResolveOutputExtent(const gert::TilingContext* context, const InputShapes& shapes, ProblemSize& problem)
{
    OP_CHECK_IF(problem.numBins != 0 && problem.numRows > std::numeric_limits<int64_t>::max() / problem.numBins,
                OP_LOGE(context->GetNodeName(), "The output element count overflows int64."), return ge::GRAPH_FAILED);
    problem.outputElements = problem.numRows * problem.numBins;
    OP_CHECK_IF(static_cast<uint64_t>(problem.outputElements) >
                    static_cast<uint64_t>(std::numeric_limits<size_t>::max() / OUTPUT_ELEMENT_BYTES),
                OP_LOGE(context->GetNodeName(), "The output byte size overflows size_t."), return ge::GRAPH_FAILED);

    const gert::Shape& outputStorageShape = shapes.output->GetStorageShape();
    OP_CHECK_IF(
        outputStorageShape.GetDim(0U) != problem.numRows || outputStorageShape.GetDim(1U) != problem.numBins,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "output", Ops::Base::ToString(outputStorageShape),
            "output shape must be [" + std::to_string(problem.numRows) + ", " + std::to_string(problem.numBins) + "]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResolvePlatformBudget(gert::TilingContext* context, PlatformBudget& budget)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    budget.maxCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(budget.maxCoreNum <= 0,
                OP_LOGE(context->GetNodeName(), "The AIV core number must be greater than zero."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context->GetNodeName(), "UB size %lu is too small for the SIMT DCache reservation.", ubSize),
                return ge::GRAPH_FAILED);
    // Everything the SIMT DCache does not claim is the dynamic UB budget TPipe allocates the private
    // histogram from, so the privatisation decision and SetDynUBufSize must use the same number.
    budget.localMemorySize = ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE;
    budget.systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// Per-core privatisation: accumulate the whole output in UB, then fold the private copies into global
// memory with one atomic DMA each. This replaces O(numValues) scattered global atomics -- which
// serialise hard once several values land in the same bin -- with O(outputElements * cores) contiguous
// writes, so it only pays off while the output stays small relative to the value count. Both conditions
// have to hold, and the UB fit is checked first so the traffic estimate cannot overflow: outputElements
// is bounded by the UB budget by then.
uint32_t DecidePrivateHistElems(const ProblemSize& problem, const PlatformBudget& budget, int64_t coreNum)
{
    if (problem.outputElements <= 0 || problem.numValues <= 0) {
        return 0U;
    }
    const uint64_t histBytes = static_cast<uint64_t>(problem.outputElements) * OUTPUT_ELEMENT_BYTES;
    const uint64_t alignedHistBytes = ((histBytes + UB_ALIGN_BYTES - 1U) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    if (alignedHistBytes > budget.localMemorySize) {
        return 0U;
    }
    if (problem.outputElements * coreNum > problem.numValues * PRIVATE_WRITEBACK_FACTOR) {
        return 0U;
    }
    return static_cast<uint32_t>(problem.outputElements);
}

ge::graphStatus ResolveUserWorkspaceSize(const gert::TilingContext* context, const ProblemSize& problem,
                                         const ScheduleParams& schedule, uint64_t& userWorkspaceSize)
{
    userWorkspaceSize = NsRaggedBinCount::USER_WORKSPACE_HEADER_BYTES;
    const bool usePreciseValue = schedule.mappingMode == MAPPING_MODE_VALUE && problem.hasWeights &&
                                 !schedule.binaryOutput &&
                                 NsRaggedBinCount::UsePreciseValuePath(problem.numValues, problem.numBins);
    if (!usePreciseValue) {
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t coreNum = static_cast<uint64_t>(schedule.coreNum);
    const uint64_t outputElements = static_cast<uint64_t>(problem.outputElements);
    constexpr uint64_t partitions = static_cast<uint64_t>(NsRaggedBinCount::PRECISE_VALUE_PARTITIONS_PER_CORE);
    constexpr uint64_t slotBytes = NsRaggedBinCount::PRECISE_VALUE_PARTIAL_BYTES;
    constexpr uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    OP_CHECK_IF(outputElements != 0U && coreNum > maxValue / outputElements,
                OP_LOGE(context->GetNodeName(), "The precise VALUE workspace entry count overflows uint64."),
                return ge::GRAPH_FAILED);
    uint64_t entryCount = coreNum * outputElements;
    OP_CHECK_IF(entryCount > maxValue / partitions,
                OP_LOGE(context->GetNodeName(), "The precise VALUE workspace partition count overflows uint64."),
                return ge::GRAPH_FAILED);
    entryCount *= partitions;
    OP_CHECK_IF(entryCount > (maxValue - userWorkspaceSize) / slotBytes,
                OP_LOGE(context->GetNodeName(), "The precise VALUE workspace byte count overflows uint64."),
                return ge::GRAPH_FAILED);
    userWorkspaceSize += entryCount * slotBytes;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResolveSchedule(gert::TilingContext* context, const ProblemSize& problem, const PlatformBudget& budget,
                                ScheduleParams& schedule)
{
    const int64_t valueCoreNum = GetRequiredCoreNum(problem.numValues, budget.maxCoreNum);
    const int64_t workload = std::max({problem.numSplits, problem.numValues, problem.outputElements});
    schedule.coreNum = GetRequiredCoreNum(workload, budget.maxCoreNum);
    OP_CHECK_IF(schedule.coreNum < MIN_CORE_NUM || schedule.coreNum > budget.maxCoreNum,
                OP_LOGE(context->GetNodeName(), "The required AIV core number %ld is invalid.", schedule.coreNum),
                return ge::GRAPH_FAILED);

    const bool lacksRowParallelism = problem.numRows < valueCoreNum;
    const bool hasLongAverageRow = SafeCeilDiv(problem.numValues, problem.numRows) > PER_CORE_MIN_ELEMENTS;
    if (lacksRowParallelism && hasLongAverageRow) {
        schedule.mappingMode = MAPPING_MODE_VALUE;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > static_cast<size_t>(ATTR_BINARY_OUTPUT)) {
        const bool* binaryOutputPtr = attrs->GetBool(ATTR_BINARY_OUTPUT);
        if (binaryOutputPtr != nullptr) {
            schedule.binaryOutput = *binaryOutputPtr;
        }
    }

    const bool usePreciseValue = schedule.mappingMode == MAPPING_MODE_VALUE && problem.hasWeights &&
                                 !schedule.binaryOutput &&
                                 NsRaggedBinCount::UsePreciseValuePath(problem.numValues, problem.numBins);
    schedule.privateHistElems = usePreciseValue ? 0U : DecidePrivateHistElems(problem, budget, schedule.coreNum);
    schedule.schMode = (schedule.mappingMode << RBC_MAPPING_MODE_SHIFT) |
                       (static_cast<uint32_t>(schedule.binaryOutput) << RBC_BINARY_OUTPUT_SHIFT) |
                       static_cast<uint32_t>(problem.hasWeights);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PublishTiling(gert::TilingContext* context, const ProblemSize& problem, const PlatformBudget& budget,
                              const ScheduleParams& schedule)
{
    RaggedBinCountTilingData* tilingData = context->GetTilingData<RaggedBinCountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(RaggedBinCountTilingData), 0, sizeof(RaggedBinCountTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "Failed to initialize RaggedBinCount tiling data."),
                return ge::GRAPH_FAILED);
    tilingData->numRows = problem.numRows;
    tilingData->numSplits = problem.numSplits;
    tilingData->numValues = problem.numValues;
    tilingData->numBins = problem.numBins;
    tilingData->outputElements = problem.outputElements;
    tilingData->usedCoreNum = static_cast<uint32_t>(schedule.coreNum);
    tilingData->privateHistElems = schedule.privateHistElems;

    context->SetTilingKey(GET_TPL_TILING_KEY(schedule.schMode));
    context->SetBlockDim(static_cast<uint32_t>(schedule.coreNum));
    context->SetScheduleMode(SCHEDULE_MODE_SYNC_ALL);

    OP_CHECK_IF(budget.localMemorySize > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context->GetNodeName(), "The local memory size %lu exceeds the Host API range.",
                        budget.localMemorySize),
                return ge::GRAPH_FAILED);
    const auto localMemoryResult = context->SetDynUBufSize(static_cast<uint32_t>(budget.localMemorySize));
    OP_CHECK_IF(localMemoryResult != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "SetDynUBufSize failed."),
                return ge::GRAPH_FAILED);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1U);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    uint64_t userWorkspaceSize = 0U;
    OP_CHECK_IF(ResolveUserWorkspaceSize(context, problem, schedule, userWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to resolve the user workspace size."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        userWorkspaceSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            budget.systemWorkspaceSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - userWorkspaceSize,
        OP_LOGE(context->GetNodeName(), "The system and user workspace size addition overflows size_t."),
        return ge::GRAPH_FAILED);
    workspaceSizes[0] = static_cast<size_t>(budget.systemWorkspaceSize + userWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RaggedBinCountTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(CheckDataTypes(context) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "splits, values, size, weights, output",
                                                      "see the preceding dtype diagnostic",
                                                      "RaggedBinCount dtype validation failed"),
                return ge::GRAPH_FAILED);

    InputShapes shapes;
    OP_CHECK_IF(FetchInputShapes(context, shapes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to fetch the RaggedBinCount input shapes."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeRanks(context, shapes) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount shape rank validation failed."),
                return ge::GRAPH_FAILED);

    ProblemSize problem;
    OP_CHECK_IF(ResolveCounts(context, shapes, problem) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount input count resolution failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ResolveOutputExtent(context, shapes, problem) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount output extent resolution failed."),
                return ge::GRAPH_FAILED);

    PlatformBudget budget;
    OP_CHECK_IF(ResolvePlatformBudget(context, budget) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount platform budget resolution failed."),
                return ge::GRAPH_FAILED);

    ScheduleParams schedule;
    OP_CHECK_IF(ResolveSchedule(context, problem, budget, schedule) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "RaggedBinCount schedule resolution failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(PublishTiling(context, problem, budget, schedule) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Failed to publish the RaggedBinCount tiling."),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(),
            "RaggedBinCount tiling: rows=%ld, values=%ld, bins=%ld, output=%ld, mapping=%u, binary=%d, "
            "weights=%d, cores=%ld, key=%u, privateHistElems=%u.",
            problem.numRows, problem.numValues, problem.numBins, problem.outputElements, schedule.mappingMode,
            static_cast<int32_t>(schedule.binaryOutput), static_cast<int32_t>(problem.hasWeights), schedule.coreNum,
            schedule.schMode, schedule.privateHistElems);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForRaggedBinCount(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(RaggedBinCount)
    .Tiling(RaggedBinCountTilingFunc)
    .TilingParse<RaggedBinCountCompileInfo>(TilingParseForRaggedBinCount);
} // namespace optiling
