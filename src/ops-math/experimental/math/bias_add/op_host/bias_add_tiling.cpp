/**
 * This file is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 Yang Zhenze, Chongqing University of Posts and Telecommunications (CQUPT).
 * All Rights Reserved.
 *
 * Author (account):
 * - Yang Zhenze <@gcw_5x5Ew5Ms>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>
#include <graph/utils/type_utils.h>
#include "securec.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/bias_add_tiling_data.h"
#include "../op_kernel/bias_add_tiling_key.h"

using namespace ge;

namespace optiling {
namespace {
constexpr const char* NCHW_STR = "NCHW";
constexpr const char* NHWC_STR = "NHWC";
constexpr const char* NCDHW_STR = "NCDHW";
constexpr const char* NDHWC_STR = "NDHWC";
constexpr size_t C_INDEX_NCXX = 1;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t UB_RESERVE_SIZE = 4096U;
constexpr uint64_t BF16_BYTES_PER_ELEM = 2U;
constexpr uint64_t BIAS_CACHE_LIMIT_DIVISOR = 4U;
constexpr uint64_t BIAS_CACHE_MIN_TILES_PER_CORE = 4U;
constexpr uint32_t BASE_BUFFER_NUM = 2U;
constexpr uint64_t TINY_FLOOR_BYTES = 1536U;
constexpr uint64_t THIN_SMALL_VECTOR_MIN_ELEMS = 90U;
constexpr uint64_t U1_MAX_ROWS_PER_TILE = 255U;
const gert::Shape G_VEC_1_SHAPE = {1};

struct BiasAddCompileInfo {
    std::vector<int64_t> broadcastBiasShape;
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    bool isUnknownRank = false;
};

enum class SecondOrderSchedule {
    T1_NOQUEUE,
    T2_BASE,
    S1_SEGMENT_ADDS,
    S2_SHORT_SEGMENT_FALLBACK,
    A1_CHANNEL_CACHE,
    A2_CHANNEL_CACHE_DOUBLE_BUFFER,
    U1_SMALL_VECTOR,
    U1_DIRECT,
    U1_BF16_CAST,
    U2_SUPERCYCLE,
    U3_BASE,
    BASE,
};

enum class ScheduleBackend {
    NATIVE,
    SMALL_RUNTIME_VECTOR,
    NORMAL_VECTOR_MATERIALIZE,
    BF16_CAST_VECTOR_MATERIALIZE,
    KCYCLE_ADD,
    GENERIC,
};

enum class PublicScheduleFamily {
    TINY,
    SEGMENT,
    ALIGNED,
    UNALIGNED,
    BASE,
};

struct ScheduleTaxonomy {
    PublicScheduleFamily family = PublicScheduleFamily::BASE;
    SecondOrderSchedule secondOrder = SecondOrderSchedule::BASE;
    ScheduleBackend backend = ScheduleBackend::GENERIC;
};

struct ShapeFeatures {
    uint64_t totalElements = 0;
    uint64_t channelSize = 0;
    uint64_t innerSize = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint32_t typeLength = 0;
    uint32_t platformCoreNum = 0;
    uint64_t usableUb = 0;
    uint64_t bytesPerElementBudget = 0;
    uint64_t alignElems = 1;
    uint64_t channelStrideBytes = 0;
    bool channelStrideAligned = false;
};

struct KCyclePolicy {
    bool enabled = false;
    uint64_t biasCacheElems = 0;
    uint64_t biasCacheBytes = 0;
    uint64_t superCycleSize = 0;
    uint64_t kCycleCount = 0;
    uint64_t bytesPerElementBudget = 0;
};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return G_VEC_1_SHAPE;
    }
    return inShape;
}

static ge::graphStatus CheckDataFormat(const std::string& attrDataFormat)
{
    static const std::unordered_set<std::string> checkList = {NCHW_STR, NHWC_STR, NCDHW_STR, NDHWC_STR};
    return checkList.count(attrDataFormat) > 0 ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

static uint64_t ProductFrom(const gert::Shape& shape, size_t start)
{
    uint64_t result = 1;
    for (size_t i = start; i < shape.GetDimNum(); ++i) {
        result *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return result;
}

static ge::graphStatus ResolveLayoutInfo(gert::TilingContext* context, const gert::Shape& xShape,
                                         const gert::Shape& biasShape, const std::string& attrDataFormat,
                                         uint64_t& channelSize, uint64_t& innerSize)
{
    const ge::Format xFormat = context->GetInputDesc(0)->GetStorageFormat();
    const size_t xDimNum = xShape.GetDimNum();

    if (xDimNum == 1) {
        channelSize = static_cast<uint64_t>(xShape.GetDim(0));
        innerSize = 1;
    } else {
        switch (xFormat) {
            case ge::FORMAT_ND:
                if (attrDataFormat == NCHW_STR || attrDataFormat == NCDHW_STR) {
                    channelSize = static_cast<uint64_t>(xShape.GetDim(C_INDEX_NCXX));
                    innerSize = ProductFrom(xShape, C_INDEX_NCXX + 1);
                } else {
                    channelSize = static_cast<uint64_t>(xShape.GetDim(xDimNum - 1));
                    innerSize = 1;
                }
                break;
            case ge::FORMAT_NCHW:
            case ge::FORMAT_NCDHW:
                channelSize = static_cast<uint64_t>(xShape.GetDim(C_INDEX_NCXX));
                innerSize = ProductFrom(xShape, C_INDEX_NCXX + 1);
                break;
            case ge::FORMAT_NHWC:
            case ge::FORMAT_NDHWC:
                channelSize = static_cast<uint64_t>(xShape.GetDim(xDimNum - 1));
                innerSize = 1;
                break;
            default:
                OP_LOGE(context->GetNodeName(), "the format of x is not supported.");
                return ge::GRAPH_FAILED;
        }
    }

    OP_CHECK_IF(channelSize < 1, OP_LOGE(context->GetNodeName(), "channelSize must be >= 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(innerSize < 1, OP_LOGE(context->GetNodeName(), "innerSize must be >= 1."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(channelSize != static_cast<uint64_t>(biasShape.GetDim(0)),
                OP_LOGE(context->GetNodeName(), "bias length must equal the resolved channel dimension."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, uint32_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t& totalElements, uint64_t& channelSize,
                                         uint64_t& innerSize, ge::DataType& dataType)
{
    auto xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    auto& xShape = EnsureNotScalar(xStorageShape->GetStorageShape());

    auto biasStorageShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasStorageShape);
    auto& biasShape = EnsureNotScalar(biasStorageShape->GetStorageShape());

    OP_CHECK_IF(xShape.GetDimNum() < 1, OP_LOGE(context->GetNodeName(), "the x shape rank must >= 1."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(biasShape.GetDimNum() != 1, OP_LOGE(context->GetNodeName(), "the bias shape rank must equal to 1."),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    const uint32_t attrNum = attrs == nullptr ? 0 : attrs->GetAttrNum();
    std::string attrDataFormat = attrNum == 0 ? NHWC_STR : attrs->GetStr(0);
    OP_CHECK_IF(CheckDataFormat(attrDataFormat) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "[attr]data_format only supports NCHW, NHWC, NCDHW, NDHWC."),
                return ge::GRAPH_FAILED);

    auto xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto biasDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    dataType = xDesc->GetDataType();
    OP_CHECK_IF(dataType != biasDesc->GetDataType(), OP_LOGE(context->GetNodeName(), "dtype of x and bias must match."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dataType != outputDesc->GetDataType(),
                OP_LOGE(context->GetNodeName(), "dtype of x and output must match."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_INT32,
        OP_LOGE(context->GetNodeName(), "input dtype only supports float, float16, bf16 and int32."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        ResolveLayoutInfo(context, xShape, biasShape, attrDataFormat, channelSize, innerSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "ResolveLayoutInfo error."), return ge::GRAPH_FAILED);

    totalElements = static_cast<uint64_t>(xShape.GetShapeSize());
    OP_CHECK_IF(totalElements == 0, OP_LOGE(context->GetNodeName(), "totalElements must be > 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetBytesPerElementBudget(ge::DataType dataType, uint32_t typeLength)
{
    if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
        return 24U;
    }
    return static_cast<uint64_t>(typeLength) * 6U;
}

static ShapeFeatures BuildShapeFeatures(uint64_t totalElements, uint64_t channelSize, uint64_t innerSize,
                                        ge::DataType dataType, uint32_t typeLength, uint64_t ubSize, uint32_t coreNum)
{
    ShapeFeatures features;
    features.totalElements = totalElements;
    features.channelSize = channelSize;
    features.innerSize = innerSize;
    features.dataType = dataType;
    features.typeLength = typeLength;
    features.platformCoreNum = coreNum;
    // If UB can't even cover the reserve, report 0 usable (downstream tileDataNum becomes
    // 0 and tiling fails cleanly) rather than the old fallback of using the FULL ubSize,
    // which was more aggressive than the normal reserved path. In practice UB is ~192KB so
    // this branch never triggers on 910B; it just removes a wrong-direction fallback.
    features.usableUb = ubSize > UB_RESERVE_SIZE ? (ubSize - UB_RESERVE_SIZE) : 0U;
    features.bytesPerElementBudget = GetBytesPerElementBudget(dataType, typeLength);
    features.alignElems = std::max<uint64_t>(1U, BLOCK_SIZE / typeLength);
    features.channelStrideBytes = channelSize * typeLength;
    features.channelStrideAligned = (features.channelStrideBytes % BLOCK_SIZE) == 0U;
    return features;
}

static bool IsComputeDtype(ge::DataType dataType)
{
    return dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT ||
           dataType == ge::DT_INT32;
}

static bool IsTinyNoQueueCandidate(const ShapeFeatures& features)
{
    return features.totalElements * features.typeLength <= TINY_FLOOR_BYTES && features.dataType != ge::DT_BF16;
}

static uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return align == 0U ? value : ((value + align - 1U) / align) * align;
}

static uint64_t GetFloatAlignedElems(uint64_t channelSize)
{
    constexpr uint64_t kFloatAlignElems = 32U / sizeof(float);
    return AlignUp(channelSize, kFloatAlignElems);
}

static uint64_t GetBf16AlignedElems(uint64_t channelSize)
{
    constexpr uint64_t kBf16AlignElems = 32U / BF16_BYTES_PER_ELEM;
    return AlignUp(channelSize, kBf16AlignElems);
}

static uint64_t RequiredBytesForSchedule(const ShapeFeatures& features, SecondOrderSchedule schedule,
                                         uint64_t tileDataNum, uint64_t materializeElems, uint64_t biasCacheBytes = 0U,
                                         uint32_t bufferNum = BASE_BUFFER_NUM)
{
    const uint64_t cAligned = AlignUp(features.channelSize, features.alignElems);
    const uint64_t floatAligned = GetFloatAlignedElems(features.channelSize);
    const uint64_t bf16Aligned = GetBf16AlignedElems(features.channelSize);

    switch (schedule) {
        case SecondOrderSchedule::T1_NOQUEUE:
            // TINY threshold is a performance discriminator only. Correctness is
            // protected by this UB budget check before entering the tiny kernel.
            return tileDataNum * features.typeLength * 3U;
        case SecondOrderSchedule::U1_SMALL_VECTOR:
            if (features.dataType == ge::DT_BF16) {
                const uint64_t rows = features.channelSize == 0U ? 0U : (features.totalElements / features.channelSize);
                return features.totalElements * BF16_BYTES_PER_ELEM * 2U + features.totalElements * sizeof(float) * 3U +
                       rows * floatAligned * sizeof(float) + bf16Aligned * BF16_BYTES_PER_ELEM +
                       floatAligned * sizeof(float);
            }
            {
                // Non-BF16 ThinTiny real buffer sum (was a flat 3*total + materializeElems
                // that didn't itemize the GatherMask tmp). Worst case is the out-of-place
                // kernel: x + y + biasFull (3*total), plus aligned bias (cAligned) and the
                // tmp (rows*cAligned). The tmp was previously a fixed constant that under-
                // allocated when rows*cAligned was large; itemizing it here keeps the UB
                // check and the emitted brcTmpBytes in sync.
                (void)materializeElems;
                const uint64_t rows = features.channelSize == 0U ? 0U : (features.totalElements / features.channelSize);
                return features.totalElements * features.typeLength * 3U + cAligned * features.typeLength +
                       rows * cAligned * features.typeLength;
            }
        case SecondOrderSchedule::U1_DIRECT:
            return tileDataNum * features.typeLength * 2U + materializeElems * features.typeLength +
                   cAligned * features.typeLength;
        case SecondOrderSchedule::U1_BF16_CAST:
            return tileDataNum * BF16_BYTES_PER_ELEM * 2U + tileDataNum * sizeof(float) * 3U +
                   bf16Aligned * BF16_BYTES_PER_ELEM + floatAligned * sizeof(float) + materializeElems * sizeof(float);
        case SecondOrderSchedule::U2_SUPERCYCLE:
        case SecondOrderSchedule::BASE:
        default: {
            uint64_t requiredBytes = tileDataNum * features.typeLength * (2U * bufferNum + 1U) + biasCacheBytes;
            if (features.dataType == ge::DT_BF16) {
                requiredBytes += tileDataNum * sizeof(float) * 3U;
            }
            return requiredBytes;
        }
    }
}

static bool IsThinTinyVectorBroadcastCandidate(const ShapeFeatures& features)
{
    if (features.innerSize != 1U || features.channelSize == 0U ||
        (features.totalElements % features.channelSize) != 0U) {
        return false;
    }
    if ((features.totalElements / features.channelSize) > U1_MAX_ROWS_PER_TILE) {
        return false;
    }
    if (features.channelStrideAligned) {
        return false;
    }
    if (features.totalElements < THIN_SMALL_VECTOR_MIN_ELEMS) {
        return false;
    }
    // GatherMask compaction safe bound (see IsGatherMaskCompactionSafe for the full note):
    // the old "WA at cBlocks=7 -> cap <=5" was misattributed; verified bit-exact on-device
    // through cBlocks=32, and TBE runs the equivalent vreducev2 at cBlocks=16. Inlined here
    // because the helper is declared later.
    constexpr uint64_t kSafeGatherBlocksInline = 32U;
    if (features.dataType == ge::DT_BF16) {
        const uint64_t floatAligned = GetFloatAlignedElems(features.channelSize);
        const uint64_t cBlocks = floatAligned / (32U / sizeof(float));
        if (cBlocks > kSafeGatherBlocksInline) {
            return false;
        }
    } else if (features.dataType == ge::DT_INT32 || features.dataType == ge::DT_FLOAT ||
               features.dataType == ge::DT_FLOAT16) {
        const uint64_t cBlocks = features.alignElems == 0U ?
                                     0U :
                                     (AlignUp(features.channelSize, features.alignElems) / features.alignElems);
        if (cBlocks > kSafeGatherBlocksInline) {
            return false;
        }
    } else {
        return false;
    }
    // Upper bound = whole-tensor materialize must fit UB (computed from the real buffer
    // sum, not a cosmetic element cap). Combined with the rows<=U1_MAX_ROWS_PER_TILE limit
    // above, this is the structural eligibility bound; no hardcoded element threshold.
    return RequiredBytesForSchedule(features, SecondOrderSchedule::U1_SMALL_VECTOR, features.totalElements,
                                    features.totalElements) <= features.usableUb;
}

static bool IsUnalignedInner1(const ShapeFeatures& features)
{
    return features.innerSize == 1U && features.channelSize != 0U &&
           (features.totalElements % features.channelSize) == 0U && !features.channelStrideAligned;
}

static uint64_t GetCBlocks(const ShapeFeatures& features)
{
    const uint64_t cAligned = AlignUp(features.channelSize, features.alignElems);
    return features.alignElems == 0U ? 0U : (cAligned / features.alignElems);
}

static bool IsGatherMaskCompactionSafe(uint64_t cBlocks)
{
    // GatherMask Copy+compaction is correct well past the old <=5 cap. The historical
    // "WA at cBlocks=7" was misattributed: it was a bf16 cross-tile WAR hazard + an old
    // tail/offset bug, both since fixed. Verified bit-exact on-device through cBlocks=32,
    // and the reference TBE BiasAdd runs the equivalent vector broadcast (vreducev2) at
    // comparable cBlocks. Relaxing to 32 routes large-C unaligned innerSize=1 shapes
    // (channel count not 32B-aligned) through the vector Copy+GatherMask broadcast (bias
    // materialized once per core, reused across tiles) instead of the scalar KCycle
    // biasCache build, removing the scalar/vector serialization on that path. Larger
    // cBlocks still fall back to KCycle via the UB-capacity check; this is a
    // correctness-verified bound, not a shape-specific gate.
    constexpr uint64_t kSafeGatherBlocks = 32U;
    return cBlocks <= kSafeGatherBlocks;
}

static bool IsNormalU1Dtype(ge::DataType dataType)
{
    return dataType == ge::DT_INT32 || dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16;
}

static uint64_t GetRows(const ShapeFeatures& features)
{
    if (features.channelSize == 0U || (features.totalElements % features.channelSize) != 0U) {
        return 0U;
    }
    return features.totalElements / features.channelSize;
}

static uint64_t SelectU1CoreNum(const ShapeFeatures& features)
{
    // 0 still means "not U1-eligible" (rows==0). Previously rows<coreNum also returned 0,
    // giving up multi-core entirely (and dropping U1) when rows was just below the core
    // count — an under-utilization. Now use min(rows, coreNum) so a shape with a few rows
    // still runs U1 on that many cores instead of falling back to a slower path. Only
    // affects shapes whose row count is below the core count.
    const uint64_t rows = GetRows(features);
    if (rows == 0U) {
        return 0U;
    }
    return std::min<uint64_t>(rows, std::max<uint64_t>(1U, features.platformCoreNum));
}

static uint64_t SelectU1DirectRowsPerTile(const ShapeFeatures& features)
{
    if (!IsNormalU1Dtype(features.dataType) || features.typeLength == 0U) {
        return 0U;
    }
    const uint64_t rows = GetRows(features);
    const uint64_t cAligned = AlignUp(features.channelSize, features.alignElems);
    const uint64_t fixedBytes = cAligned * features.typeLength;
    const uint64_t perRowBytes = features.channelSize * features.typeLength * 2U + cAligned * features.typeLength;
    if (rows == 0U || perRowBytes == 0U || features.usableUb <= fixedBytes) {
        return 0U;
    }
    const uint64_t rowsByUb = (features.usableUb - fixedBytes) / perRowBytes;
    return std::min<uint64_t>(std::min<uint64_t>(rowsByUb, U1_MAX_ROWS_PER_TILE), rows);
}

static uint64_t SelectU1Bf16RowsPerTile(const ShapeFeatures& features)
{
    if (features.dataType != ge::DT_BF16) {
        return 0U;
    }
    const uint64_t rows = GetRows(features);
    const uint64_t bf16Aligned = GetBf16AlignedElems(features.channelSize);
    const uint64_t floatAligned = GetFloatAlignedElems(features.channelSize);
    const uint64_t fixedBytes = bf16Aligned * BF16_BYTES_PER_ELEM + floatAligned * sizeof(float);
    const uint64_t perRowBytes = features.channelSize * BF16_BYTES_PER_ELEM * 2U +
                                 features.channelSize * sizeof(float) * 3U + floatAligned * sizeof(float);
    if (rows == 0U || perRowBytes == 0U || features.usableUb <= fixedBytes) {
        return 0U;
    }
    const uint64_t rowsByUb = (features.usableUb - fixedBytes) / perRowBytes;
    return std::min<uint64_t>(std::min<uint64_t>(rowsByUb, U1_MAX_ROWS_PER_TILE), rows);
}

static bool IsUnalignedU1DirectCandidate(const ShapeFeatures& features)
{
    if (!IsUnalignedInner1(features) || !IsNormalU1Dtype(features.dataType) ||
        (features.channelSize % features.alignElems) == 0U || SelectU1CoreNum(features) == 0U ||
        !IsGatherMaskCompactionSafe(GetCBlocks(features))) {
        return false;
    }
    const uint64_t rowsPerTile = SelectU1DirectRowsPerTile(features);
    if (rowsPerTile == 0U) {
        return false;
    }
    const uint64_t cAligned = AlignUp(features.channelSize, features.alignElems);
    const uint64_t tileDataNum = rowsPerTile * features.channelSize;
    return RequiredBytesForSchedule(features, SecondOrderSchedule::U1_DIRECT, tileDataNum, rowsPerTile * cAligned) <=
           features.usableUb;
}

static bool IsUnalignedU1Bf16Candidate(const ShapeFeatures& features)
{
    if (!IsUnalignedInner1(features) || features.dataType != ge::DT_BF16 || SelectU1CoreNum(features) == 0U) {
        return false;
    }
    const uint64_t floatAligned = GetFloatAlignedElems(features.channelSize);
    const uint64_t cBlocks = floatAligned / (32U / sizeof(float));
    if (!IsGatherMaskCompactionSafe(cBlocks)) {
        return false;
    }
    const uint64_t rowsPerTile = SelectU1Bf16RowsPerTile(features);
    if (rowsPerTile == 0U) {
        return false;
    }
    const uint64_t tileDataNum = rowsPerTile * features.channelSize;
    return RequiredBytesForSchedule(features, SecondOrderSchedule::U1_BF16_CAST, tileDataNum,
                                    rowsPerTile * floatAligned) <= features.usableUb;
}

static PublicScheduleFamily ClassifyPublicScheduleFamily(const ShapeFeatures& features)
{
    // First-order schedule taxonomy. DType, tile size and UB policy are second-order
    // parameters inside the family; they should not create point-shaped families.
    //
    // TINY claims the fixed-floor regime where even vector-broadcast setup isn't worth
    // it. The threshold is in BYTES, but the scalar FillBias broadcast cost it avoids is
    // per-ELEMENT, so a byte gate mis-routes 2-byte dtypes (fp16/bf16) with the same
    // element count as a 4-byte sibling onto the slow scalar path (controlled evidence:
    // C19/t570 fp16 5.07x scalar vs int32 1.54x vector; C17/t510 fp16 4.48x vs int32
    // 2.23x). Let a vector-broadcast-eligible shape take the vector path even under the
    // byte floor; its own eligibility gate (>=90 elems, rows<=255, unaligned) keeps the
    // genuine sub-90-element floor points (t15/t30/t65) on the scalar tiny path.
    if (features.totalElements * features.typeLength <= TINY_FLOOR_BYTES &&
        !IsThinTinyVectorBroadcastCandidate(features)) {
        return PublicScheduleFamily::TINY;
    }
    if (features.innerSize > 1U) {
        return PublicScheduleFamily::SEGMENT;
    }
    if (features.innerSize == 1U && features.channelStrideAligned) {
        return PublicScheduleFamily::ALIGNED;
    }
    if (features.innerSize == 1U) {
        return PublicScheduleFamily::UNALIGNED;
    }
    return PublicScheduleFamily::BASE;
}

static SecondOrderSchedule SelectSecondOrderSchedule(const ShapeFeatures& features, PublicScheduleFamily family)
{
    switch (family) {
        case PublicScheduleFamily::TINY:
            return IsTinyNoQueueCandidate(features) ? SecondOrderSchedule::T1_NOQUEUE : SecondOrderSchedule::T2_BASE;
        case PublicScheduleFamily::SEGMENT:
            if ((features.innerSize * features.typeLength) % BLOCK_SIZE == 0U) {
                return SecondOrderSchedule::S1_SEGMENT_ADDS;
            }
            return SecondOrderSchedule::S2_SHORT_SEGMENT_FALLBACK;
        case PublicScheduleFamily::ALIGNED:
            // Same BASE kernel; second-order policy only changes tiling/core/cache parameters.
            // A1 means the whole aligned channel-cache tile fits in UB; A2 keeps the same kernel
            // and lets tiling split the work into multiple channel-cache tiles.
            {
                const uint64_t biasCacheBytes = ((features.channelSize * features.typeLength + BLOCK_SIZE - 1U) /
                                                 BLOCK_SIZE) *
                                                BLOCK_SIZE;
                return RequiredBytesForSchedule(features, SecondOrderSchedule::A1_CHANNEL_CACHE, features.totalElements,
                                                features.totalElements, biasCacheBytes) <= features.usableUb ?
                           SecondOrderSchedule::A1_CHANNEL_CACHE :
                           SecondOrderSchedule::A2_CHANNEL_CACHE_DOUBLE_BUFFER;
            }
        case PublicScheduleFamily::UNALIGNED:
            if (IsThinTinyVectorBroadcastCandidate(features)) {
                return SecondOrderSchedule::U1_SMALL_VECTOR;
            }
            if (IsUnalignedU1DirectCandidate(features)) {
                return SecondOrderSchedule::U1_DIRECT;
            }
            if (IsUnalignedU1Bf16Candidate(features)) {
                return SecondOrderSchedule::U1_BF16_CAST;
            }
            return SecondOrderSchedule::U2_SUPERCYCLE;
        case PublicScheduleFamily::BASE:
        default:
            return SecondOrderSchedule::BASE;
    }
}

static ScheduleBackend SelectScheduleBackend(const ShapeFeatures& features, PublicScheduleFamily family,
                                             SecondOrderSchedule secondOrder)
{
    switch (secondOrder) {
        case SecondOrderSchedule::T1_NOQUEUE:
        case SecondOrderSchedule::T2_BASE:
        case SecondOrderSchedule::S1_SEGMENT_ADDS:
        case SecondOrderSchedule::S2_SHORT_SEGMENT_FALLBACK:
        case SecondOrderSchedule::A1_CHANNEL_CACHE:
        case SecondOrderSchedule::A2_CHANNEL_CACHE_DOUBLE_BUFFER:
            return ScheduleBackend::NATIVE;
        case SecondOrderSchedule::U1_SMALL_VECTOR:
            return features.dataType == ge::DT_BF16 ? ScheduleBackend::BF16_CAST_VECTOR_MATERIALIZE :
                                                      ScheduleBackend::SMALL_RUNTIME_VECTOR;
        case SecondOrderSchedule::U1_DIRECT:
            return ScheduleBackend::NORMAL_VECTOR_MATERIALIZE;
        case SecondOrderSchedule::U1_BF16_CAST:
            return ScheduleBackend::BF16_CAST_VECTOR_MATERIALIZE;
        case SecondOrderSchedule::U2_SUPERCYCLE:
            return ScheduleBackend::KCYCLE_ADD;
        case SecondOrderSchedule::U3_BASE:
        case SecondOrderSchedule::BASE:
        default:
            break;
    }
    return family == PublicScheduleFamily::UNALIGNED ? ScheduleBackend::GENERIC : ScheduleBackend::NATIVE;
}

static ScheduleTaxonomy SelectScheduleTaxonomy(const ShapeFeatures& features)
{
    ScheduleTaxonomy taxonomy;
    taxonomy.family = ClassifyPublicScheduleFamily(features);
    taxonomy.secondOrder = SelectSecondOrderSchedule(features, taxonomy.family);
    taxonomy.backend = SelectScheduleBackend(features, taxonomy.family, taxonomy.secondOrder);
    return taxonomy;
}

static bool TryEmitTinyNoQueue(gert::TilingContext* context, const ShapeFeatures& features, BiasAddTilingData* tiling)
{
    if (!IsTinyNoQueueCandidate(features)) {
        return false;
    }
    if (RequiredBytesForSchedule(features, SecondOrderSchedule::T1_NOQUEUE, features.totalElements,
                                 features.totalElements) > features.usableUb) {
        return false;
    }
    tiling->totalElements = static_cast<int64_t>(features.totalElements);
    tiling->channelSize = static_cast<int64_t>(features.channelSize);
    tiling->innerSize = static_cast<int64_t>(features.innerSize);
    tiling->smallCoreDataNum = static_cast<int64_t>(features.totalElements);
    tiling->bigCoreDataNum = static_cast<int64_t>(features.totalElements);
    tiling->finalBigTileNum = 1;
    tiling->finalSmallTileNum = 1;
    tiling->tileDataNum = static_cast<int64_t>(features.totalElements);
    tiling->biasCacheElems = 0;
    tiling->brcTmpBytes = 0;
    tiling->smallTailDataNum = static_cast<int64_t>(features.totalElements);
    tiling->bigTailDataNum = static_cast<int64_t>(features.totalElements);
    tiling->tailBlockNum = 0;
    tiling->useFastPath = 1;
    context->SetBlockDim(1U);
    context->SetTilingKey(GET_TPL_TILING_KEY(BIAS_ADD_TPL_SCH_MODE_TINY_NOQUEUE));
    return true;
}

static bool TryEmitThinTinyVectorBroadcast(gert::TilingContext* context, const ShapeFeatures& features,
                                           BiasAddTilingData* tiling)
{
    if (!IsThinTinyVectorBroadcastCandidate(features)) {
        return false;
    }
    tiling->totalElements = static_cast<int64_t>(features.totalElements);
    tiling->channelSize = static_cast<int64_t>(features.channelSize);
    tiling->innerSize = static_cast<int64_t>(features.innerSize);
    tiling->smallCoreDataNum = static_cast<int64_t>(features.totalElements);
    tiling->bigCoreDataNum = static_cast<int64_t>(features.totalElements);
    tiling->finalBigTileNum = 1;
    tiling->finalSmallTileNum = 1;
    tiling->tileDataNum = static_cast<int64_t>(features.totalElements);
    tiling->biasCacheElems = 0;
    // brcTmp holds the GatherMask source: rows * cAligned elements. The non-BF16
    // ThinTiny kernel reads brcTmpBytes for this buffer, so it must be sized to the
    // real need (a previous fixed 8192 under-allocated for large rows*cAligned and only
    // avoided corruption because brcTmpBuffer_ is the last UB buffer). The BF16 ThinTiny
    // kernel self-computes its float tmp and ignores this field; we still size it for the
    // float worst case so the value is meaningful either way.
    {
        const uint64_t rows = features.totalElements / features.channelSize;
        const uint64_t cAlignedT = AlignUp(features.channelSize, features.alignElems);
        const uint64_t tmpBytesT = rows * cAlignedT * features.typeLength;
        const uint64_t tmpBytesBf16 = rows * GetFloatAlignedElems(features.channelSize) * sizeof(float);
        tiling->brcTmpBytes = static_cast<int64_t>(features.dataType == ge::DT_BF16 ? tmpBytesBf16 : tmpBytesT);
    }
    tiling->smallTailDataNum = static_cast<int64_t>(features.totalElements);
    tiling->bigTailDataNum = static_cast<int64_t>(features.totalElements);
    tiling->tailBlockNum = 0;
    tiling->useFastPath = 1;
    context->SetBlockDim(1U);
    context->SetTilingKey(GET_TPL_TILING_KEY(BIAS_ADD_TPL_SCH_MODE_THIN_TINY_VECTOR_BROADCAST));
    return true;
}

static bool TryEmitUnalignedU1Direct(gert::TilingContext* context, const ShapeFeatures& features,
                                     BiasAddTilingData* tiling)
{
    // UNALIGNED/U1/DIRECT: whole-tile bias materialization for non-BF16 dtype.
    // Boundary is UB capacity plus GatherMask compaction safety, not a point-shaped C/T window.
    if (!IsUnalignedU1DirectCandidate(features)) {
        return false;
    }

    const uint64_t coreNum = SelectU1CoreNum(features);
    const uint64_t rowsPerTile = SelectU1DirectRowsPerTile(features);
    const uint64_t kChannel = features.channelSize;
    const uint64_t kCAligned = AlignUp(kChannel, features.alignElems);
    const uint64_t kCBlocks = GetCBlocks(features);
    if (!IsGatherMaskCompactionSafe(kCBlocks)) {
        return false;
    }

    // SelectU1CoreNum returns min(rows, coreNum) and is non-zero here (gated by the candidate
    // check above), so coreNum <= rows always holds; no rows<coreNum guard is needed.
    const uint64_t rows = features.totalElements / kChannel;
    if (coreNum == 0U || rowsPerTile == 0U) {
        return false;
    }

    const uint64_t tileDataNum = rowsPerTile * kChannel;
    const uint64_t kTmpBytes = rowsPerTile * kCAligned * features.typeLength;
    const uint64_t requiredBytes = RequiredBytesForSchedule(features, SecondOrderSchedule::U1_DIRECT, tileDataNum,
                                                            rowsPerTile * kCAligned);
    if (requiredBytes > features.usableUb) {
        return false;
    }

    const uint64_t smallRows = rows / coreNum;
    const uint64_t tailBlockNum = rows % coreNum;
    const uint64_t bigRows = smallRows + 1U;
    const uint64_t smallCoreDataNum = smallRows * kChannel;
    const uint64_t bigCoreDataNum = bigRows * kChannel;
    const uint64_t finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1U) / tileDataNum;
    const uint64_t finalBigTileNum = (bigCoreDataNum + tileDataNum - 1U) / tileDataNum;
    const uint64_t smallTailDataNum = smallCoreDataNum - tileDataNum * (finalSmallTileNum - 1U);
    const uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * (finalBigTileNum - 1U);

    tiling->totalElements = static_cast<int64_t>(features.totalElements);
    tiling->channelSize = static_cast<int64_t>(features.channelSize);
    tiling->innerSize = static_cast<int64_t>(features.innerSize);
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->biasCacheElems = 0;
    tiling->brcTmpBytes = static_cast<int64_t>(kTmpBytes);
    tiling->useFastPath = 2;
    tiling->superCycleSize = 0;
    tiling->kCycleCount = 0;
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE));
    return true;
}

static bool TryEmitUnalignedU1Bf16Cast(gert::TilingContext* context, const ShapeFeatures& features,
                                       BiasAddTilingData* tiling)
{
    // UNALIGNED/U1/BF16_CAST: same whole-tile materialization model wrapped by BF16 cast decorator.
    // Boundary is UB capacity plus GatherMask compaction safety, not a point-shaped C/T window.
    if (!IsUnalignedU1Bf16Candidate(features)) {
        return false;
    }

    const uint64_t coreNum = SelectU1CoreNum(features);
    const uint64_t rowsPerTile = SelectU1Bf16RowsPerTile(features);
    const uint64_t kChannel = features.channelSize;
    const uint64_t kFloatAligned = GetFloatAlignedElems(kChannel);
    const uint64_t kCBlocks = (kFloatAligned / (32U / sizeof(float)));
    if (!IsGatherMaskCompactionSafe(kCBlocks)) {
        return false;
    }
    const uint64_t kTmpBytes = rowsPerTile * kFloatAligned * sizeof(float);

    // SelectU1CoreNum returns min(rows, coreNum) and is non-zero here (gated by the candidate
    // check above), so coreNum <= rows always holds; no rows<coreNum guard is needed.
    const uint64_t rows = features.totalElements / kChannel;
    if (coreNum == 0U || rowsPerTile == 0U) {
        return false;
    }

    const uint64_t tileDataNum = rowsPerTile * kChannel;
    const uint64_t requiredBytes = RequiredBytesForSchedule(features, SecondOrderSchedule::U1_BF16_CAST, tileDataNum,
                                                            rowsPerTile * kFloatAligned);
    if (requiredBytes > features.usableUb) {
        return false;
    }

    const uint64_t smallRows = rows / coreNum;
    const uint64_t tailBlockNum = rows % coreNum;
    const uint64_t bigRows = smallRows + 1U;
    const uint64_t smallCoreDataNum = smallRows * kChannel;
    const uint64_t bigCoreDataNum = bigRows * kChannel;
    const uint64_t finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1U) / tileDataNum;
    const uint64_t finalBigTileNum = (bigCoreDataNum + tileDataNum - 1U) / tileDataNum;
    const uint64_t smallTailDataNum = smallCoreDataNum - tileDataNum * (finalSmallTileNum - 1U);
    const uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * (finalBigTileNum - 1U);

    tiling->totalElements = static_cast<int64_t>(features.totalElements);
    tiling->channelSize = static_cast<int64_t>(features.channelSize);
    tiling->innerSize = static_cast<int64_t>(features.innerSize);
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->biasCacheElems = 0;
    tiling->brcTmpBytes = static_cast<int64_t>(kTmpBytes);
    tiling->useFastPath = 3;
    tiling->superCycleSize = 0;
    tiling->kCycleCount = 0;
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    context->SetTilingKey(GET_TPL_TILING_KEY(BIAS_ADD_TPL_SCH_MODE_BROADCAST_UB_TILE));
    return true;
}

static uint64_t Gcd(uint64_t a, uint64_t b)
{
    while (b != 0U) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static void ApplyAlignedChannelCorePolicy(const ShapeFeatures& features, uint32_t& finalCoreNum)
{
    if (features.innerSize != 1U || features.channelSize == 0U || !features.channelStrideAligned ||
        !IsComputeDtype(features.dataType)) {
        return;
    }
    uint32_t alignedCoreNum = 0;
    const uint32_t maxAlignedCoreNum = std::min<uint32_t>(finalCoreNum, 32U);
    for (uint32_t candidate = maxAlignedCoreNum; candidate >= 1U; --candidate) {
        if ((candidate % 4U) != 0U) {
            continue;
        }
        if (features.totalElements % (static_cast<uint64_t>(candidate) * features.channelSize) == 0U) {
            alignedCoreNum = candidate;
            break;
        }
    }
    if (alignedCoreNum > 0U) {
        finalCoreNum = alignedCoreNum;
    }
}

static KCyclePolicy SelectKCyclePolicy(const ShapeFeatures& features, uint32_t finalCoreNum)
{
    KCyclePolicy policy;
    if (features.innerSize != 1U || features.channelSize == 0U || features.channelStrideAligned ||
        (features.channelSize % features.alignElems) == 0U || !IsComputeDtype(features.dataType)) {
        return policy;
    }

    uint64_t g = Gcd(features.alignElems, features.channelSize);
    policy.kCycleCount = features.alignElems / g;
    policy.superCycleSize = policy.kCycleCount * features.channelSize;
    const uint64_t scBytes = policy.superCycleSize * features.typeLength;
    const uint64_t scUbLimit = features.usableUb / BIAS_CACHE_LIMIT_DIVISOR;
    if (scBytes > scUbLimit) {
        policy.superCycleSize = 0;
        policy.kCycleCount = 0;
        return policy;
    }

    // Cache exactly one superCycle of bias. (The old Mx multiplier was hardwired to 1, so
    // its shrink loop was dead code; removed. Reintroduce a real Mx>1 policy here if ever
    // needed.)
    const uint64_t cacheElems = policy.superCycleSize;
    const uint64_t cacheBytes = scBytes;
    policy.bytesPerElementBudget = features.bytesPerElementBudget;
    const uint64_t tileUsableUb = features.usableUb - cacheBytes;
    uint64_t scTile = tileUsableUb / policy.bytesPerElementBudget;
    scTile = (scTile / policy.superCycleSize) * policy.superCycleSize;
    if (scTile >= policy.superCycleSize &&
        static_cast<uint64_t>(features.totalElements / finalCoreNum) >= policy.superCycleSize) {
        policy.enabled = true;
        policy.biasCacheElems = cacheElems;
        policy.biasCacheBytes = cacheBytes;
    } else {
        policy.superCycleSize = 0;
        policy.kCycleCount = 0;
    }
    return policy;
}

static uint64_t RoundUpToBlock(uint64_t bytes) { return ((bytes + BLOCK_SIZE - 1U) / BLOCK_SIZE) * BLOCK_SIZE; }

static bool ValidateResourceBudget(const ShapeFeatures& features, uint64_t tileDataNum, uint64_t biasCacheBytes,
                                   uint32_t bufferNum, bool tinyNoQueue)
{
    if (tinyNoQueue) {
        return RequiredBytesForSchedule(features, SecondOrderSchedule::T1_NOQUEUE, tileDataNum, tileDataNum, 0U,
                                        bufferNum) <= features.usableUb;
    }
    return RequiredBytesForSchedule(features, SecondOrderSchedule::BASE, tileDataNum, tileDataNum, biasCacheBytes,
                                    bufferNum) <= features.usableUb;
}

static ge::graphStatus InitWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // BiasAdd is pure elementwise broadcast-add: no kernel uses GM workspace (the entry
    // never passes one to Init and Process never references it). Request 0 instead of a
    // copy-pasted 16MB system reserve. Verified on-device across all schedule families.
    currentWorkspace[0] = 0U;
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingForBiasAdd(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("BiasAdd", "tiling context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    uint32_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetPlatformInfo error."), return ge::GRAPH_FAILED);

    uint64_t totalElements = 0;
    uint64_t channelSize = 0;
    uint64_t innerSize = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalElements, channelSize, innerSize, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetShapeAttrsInfo error."), return ge::GRAPH_FAILED);

    // The kernel reads per-core/per-tile tiling fields as uint32_t (coreDataNum/tileDataNum/
    // channelSize/processDataNum). Those are all <= totalElements, so a single uint32 bound
    // on totalElements prevents the silent truncation the kernel would otherwise do on a
    // >4G-element tensor — fail explicitly instead. (No realistic 910B shape hits this; raise
    // to a per-field check if >4G element tensors ever become a target.)
    OP_CHECK_IF(totalElements > 0xFFFFFFFFULL,
                OP_LOGE(context->GetNodeName(), "BiasAdd: totalElements exceeds uint32 tiling-field limit."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(InitWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetWorkspaceSize error."), return ge::GRAPH_FAILED);

    BiasAddTilingData* tiling = context->GetTilingData<BiasAddTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BiasAddTilingData), 0, sizeof(BiasAddTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "set tiling data error."), return ge::GRAPH_FAILED);

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context->GetNodeName(), "typeLength is 0."), return ge::GRAPH_FAILED);

    const ShapeFeatures features = BuildShapeFeatures(totalElements, channelSize, innerSize, dataType, typeLength,
                                                      ubSize, coreNum);
    const ScheduleTaxonomy taxonomy = SelectScheduleTaxonomy(features);
    const PublicScheduleFamily family = taxonomy.family;
    const SecondOrderSchedule secondOrder = taxonomy.secondOrder;
    const ScheduleBackend backend = taxonomy.backend;
    // Observability: record the tiling decision so production issues can be diagnosed
    // without re-deriving the schedule by hand.
    OP_LOGI(context->GetNodeName(),
            "BiasAdd schedule: family=%d secondOrder=%d backend=%d total=%lu C=%lu inner=%lu dtype=%d",
            static_cast<int>(family), static_cast<int>(secondOrder), static_cast<int>(backend),
            static_cast<unsigned long>(totalElements), static_cast<unsigned long>(channelSize),
            static_cast<unsigned long>(innerSize), static_cast<int>(dataType));

    uint64_t tileDataNum = features.usableUb / features.bytesPerElementBudget;
    tileDataNum = std::max<uint64_t>(features.alignElems, (tileDataNum / features.alignElems) * features.alignElems);
    tileDataNum = std::min<uint64_t>(tileDataNum, totalElements);
    OP_CHECK_IF(tileDataNum == 0, OP_LOGE(context->GetNodeName(), "tileDataNum must be > 0."), return ge::GRAPH_FAILED);

    uint32_t finalCoreNum = static_cast<uint32_t>(std::min<uint64_t>(totalElements, coreNum));
    finalCoreNum = std::max<uint32_t>(1U, finalCoreNum);

    // TINY/T1-T2: single-core noqueue where safe; otherwise fall through to
    // the same structural UNALIGNED/BASE paths as larger shapes.
    uint64_t biasCacheElems = 0;
    uint64_t biasCacheBytes = 0;
    uint64_t superCycleSize = 0;
    if (secondOrder == SecondOrderSchedule::T1_NOQUEUE && TryEmitTinyNoQueue(context, features, tiling)) {
        return ge::GRAPH_SUCCESS;
    }

    // UNALIGNED is three-layered:
    // family = odd-C/non-32B NHWC broadcast;
    // secondOrder = U1 whole-tile materialize or U2 superCycle;
    // backend = small runtime vector, normal materialize, bf16 cast materialize, or kcycle add.
    if (secondOrder == SecondOrderSchedule::U1_DIRECT && backend == ScheduleBackend::NORMAL_VECTOR_MATERIALIZE &&
        TryEmitUnalignedU1Direct(context, features, tiling)) {
        return ge::GRAPH_SUCCESS;
    }

    if (secondOrder == SecondOrderSchedule::U1_BF16_CAST && backend == ScheduleBackend::BF16_CAST_VECTOR_MATERIALIZE &&
        TryEmitUnalignedU1Bf16Cast(context, features, tiling)) {
        return ge::GRAPH_SUCCESS;
    }

    // UNALIGNED small/U1-lite: runtime vector broadcast for small odd-C NHWC.
    // This replaces point-shaped thin exact gates.
    if (secondOrder == SecondOrderSchedule::U1_SMALL_VECTOR &&
        (backend == ScheduleBackend::SMALL_RUNTIME_VECTOR ||
         backend == ScheduleBackend::BF16_CAST_VECTOR_MATERIALIZE) &&
        TryEmitThinTinyVectorBroadcast(context, features, tiling)) {
        return ge::GRAPH_SUCCESS;
    }

    if (family == PublicScheduleFamily::ALIGNED) {
        ApplyAlignedChannelCorePolicy(features, finalCoreNum);
    }
    // K-cycle Mx: non-aligned innerSize==1. superCycle=lcm(alignElems,C).
    // Mx cache = cacheMul * superCycle. tileDataNum from isolation block.
    uint64_t kCycleCount = 0;
    bool useKCycle = false;
    uint64_t kCycleBytesPerElementBudget = features.bytesPerElementBudget;
    const KCyclePolicy kCyclePolicy = (secondOrder == SecondOrderSchedule::U2_SUPERCYCLE &&
                                       backend == ScheduleBackend::KCYCLE_ADD) ?
                                          SelectKCyclePolicy(features, finalCoreNum) :
                                          KCyclePolicy();
    if (kCyclePolicy.enabled) {
        useKCycle = true;
        biasCacheBytes = kCyclePolicy.biasCacheBytes;
        biasCacheElems = kCyclePolicy.biasCacheElems;
        superCycleSize = kCyclePolicy.superCycleSize;
        kCycleCount = kCyclePolicy.kCycleCount;
        kCycleBytesPerElementBudget = kCyclePolicy.bytesPerElementBudget;
    }
    if (!useKCycle) {
        superCycleSize = 0;
        kCycleCount = 0;
        biasCacheElems = 0;
        biasCacheBytes = 0;
    }

    // Small bias full cache: cache all bias values when they fit in UB,
    // regardless of channel stride alignment. Eliminates per-element GM
    // reads for small/medium NHWC shapes (fourth category).
    if (innerSize == 1U && channelSize > 0 && biasCacheElems == 0 &&
        (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT ||
         dataType == ge::DT_INT32)) {
        const uint64_t requiredBiasCacheBytes = channelSize * typeLength;
        const uint64_t biasCacheLimit = features.usableUb / BIAS_CACHE_LIMIT_DIVISOR;
        if (requiredBiasCacheBytes <= biasCacheLimit) {
            biasCacheBytes = RoundUpToBlock(requiredBiasCacheBytes);
            biasCacheElems = biasCacheBytes / typeLength;
        }
    }
    const uint64_t initialSmallCoreDataNum = totalElements / finalCoreNum;
    const uint64_t initialTailBlockNum = totalElements % finalCoreNum;
    const uint64_t initialBigCoreDataNum = initialTailBlockNum == 0 ? initialSmallCoreDataNum :
                                                                      (initialSmallCoreDataNum + 1U);
    const uint64_t initialTilesPerCore = (initialBigCoreDataNum + tileDataNum - 1U) / tileDataNum;

    if (innerSize == 1U && features.channelStrideAligned &&
        (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT ||
         dataType == ge::DT_INT32)) {
        const uint64_t requiredBiasCacheBytes = channelSize * typeLength;
        const uint64_t biasCacheLimit = features.usableUb / BIAS_CACHE_LIMIT_DIVISOR;
        if (requiredBiasCacheBytes <= biasCacheLimit && initialTilesPerCore >= BIAS_CACHE_MIN_TILES_PER_CORE) {
            biasCacheBytes = RoundUpToBlock(requiredBiasCacheBytes);
            biasCacheElems = biasCacheBytes / typeLength;
        }
    }

    const uint64_t brcTmpBytes = 0U;
    if (useKCycle) {
        const uint64_t tileUsableUb = features.usableUb - biasCacheBytes;
        tileDataNum = tileUsableUb / kCycleBytesPerElementBudget;
        tileDataNum = (tileDataNum / superCycleSize) * superCycleSize;
        if (tileDataNum < superCycleSize) {
            // kCycle is no longer viable at this UB budget. The superCycle-aligned value above is
            // < superCycleSize (i.e. 0), so disabling kCycle without restoring a sane tile would
            // hand the degenerate value to the downstream innerSize-align / auto-shrink (which only
            // shrinks, never grows) and over-fragment the tiling. Reset to the non-kCycle baseline
            // (same formula as the initial tileDataNum) so the shape falls back cleanly to BASE.
            useKCycle = false;
            biasCacheElems = 0;
            biasCacheBytes = 0;
            tileDataNum = features.usableUb / features.bytesPerElementBudget;
            tileDataNum = std::max<uint64_t>(features.alignElems,
                                             (tileDataNum / features.alignElems) * features.alignElems);
        }
        tileDataNum = std::min<uint64_t>(tileDataNum, totalElements);
        if (useKCycle) {
            OP_CHECK_IF(tileDataNum == 0, OP_LOGE(context->GetNodeName(), "tileDataNum must be > 0."),
                        return ge::GRAPH_FAILED);
        }
    } else if (biasCacheBytes > 0) {
        const uint64_t tileUsableUb = features.usableUb - biasCacheBytes;
        tileDataNum = tileUsableUb / features.bytesPerElementBudget;
        tileDataNum = std::max<uint64_t>(features.alignElems,
                                         (tileDataNum / features.alignElems) * features.alignElems);
        if (innerSize == 1U && channelSize > 0 && tileDataNum >= channelSize) {
            const uint64_t channelAligned = (tileDataNum / channelSize) * channelSize;
            if (channelAligned >= tileDataNum / 2U) {
                tileDataNum = channelAligned;
            }
        }
        tileDataNum = std::min<uint64_t>(tileDataNum, totalElements);
        OP_CHECK_IF(tileDataNum == 0, OP_LOGE(context->GetNodeName(), "tileDataNum must be > 0."),
                    return ge::GRAPH_FAILED);
    }

    // Path B: Align tiles and cores to innerSize when 32B-aligned, so per-channel
    // Adds (ComputeSegmentAdds) can fire for innerSize > 1 without scalar FillBias.
    if (innerSize > 1U && (innerSize * typeLength) % BLOCK_SIZE == 0U && tileDataNum >= innerSize) {
        uint64_t alignedTileDataNum = (tileDataNum / innerSize) * innerSize;
        if (alignedTileDataNum >= tileDataNum / 2U) {
            tileDataNum = alignedTileDataNum;
        }
        uint32_t alignedCoreNum = 0;
        const uint32_t maxAlignedCoreNum = std::min<uint32_t>(finalCoreNum, 32U);
        for (uint32_t candidate = maxAlignedCoreNum; candidate >= 1U; --candidate) {
            if (totalElements % (static_cast<uint64_t>(candidate) * innerSize) == 0U) {
                alignedCoreNum = candidate;
                break;
            }
        }
        // Guard: accept only if aligned core count >= 25% of original
        // to avoid parallelism collapse on large totalElements.
        if (alignedCoreNum > 0U && alignedCoreNum >= finalCoreNum / 4U) {
            finalCoreNum = alignedCoreNum;
        }
    }

    // Auto-shrink the BASE tile if the buffer budget is exceeded, instead of failing the
    // whole op. Halve tileDataNum (aligned) down to the alignment floor until it fits; only
    // fail if even the minimal tile can't (e.g. biasCache itself too big). Normal shapes fit
    // at the initial tile and never enter this loop; it just hardens large/odd shapes.
    while (tileDataNum > features.alignElems &&
           !ValidateResourceBudget(features, tileDataNum, biasCacheBytes, BASE_BUFFER_NUM, false)) {
        uint64_t shrunk = (tileDataNum / 2U / features.alignElems) * features.alignElems;
        tileDataNum = shrunk >= features.alignElems ? shrunk : features.alignElems;
    }

    uint64_t smallCoreDataNum;
    uint64_t tailBlockNum;
    uint64_t bigCoreDataNum;
    smallCoreDataNum = totalElements / finalCoreNum;
    tailBlockNum = totalElements % finalCoreNum;
    bigCoreDataNum = tailBlockNum == 0 ? smallCoreDataNum : (smallCoreDataNum + 1U);
    const uint64_t finalSmallTileNum = smallCoreDataNum == 0 ? 0 :
                                                               ((smallCoreDataNum + tileDataNum - 1U) / tileDataNum);
    const uint64_t finalBigTileNum = bigCoreDataNum == 0 ? 0 : ((bigCoreDataNum + tileDataNum - 1U) / tileDataNum);
    const uint64_t smallTailDataNum = finalSmallTileNum == 0 ?
                                          0 :
                                          (smallCoreDataNum - tileDataNum * (finalSmallTileNum - 1U));
    const uint64_t bigTailDataNum = finalBigTileNum == 0 ? 0 : (bigCoreDataNum - tileDataNum * (finalBigTileNum - 1U));
    const bool resourceBudgetOk = ValidateResourceBudget(features, tileDataNum, biasCacheBytes, BASE_BUFFER_NUM, false);
    OP_CHECK_IF(!resourceBudgetOk,
                OP_LOGE(context->GetNodeName(),
                        "BiasAdd UB budget exceeded even at minimal tile (biasCache too large for UB)."),
                return ge::GRAPH_FAILED);

    tiling->totalElements = static_cast<int64_t>(totalElements);
    tiling->channelSize = static_cast<int64_t>(channelSize);
    tiling->innerSize = static_cast<int64_t>(innerSize);
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->biasCacheElems = static_cast<int64_t>(biasCacheElems);
    tiling->brcTmpBytes = static_cast<int64_t>(brcTmpBytes);
    tiling->superCycleSize = static_cast<int64_t>(superCycleSize);
    tiling->kCycleCount = static_cast<int64_t>(kCycleCount);
    tiling->useFastPath = 0;
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);

    context->SetBlockDim(finalCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(BIAS_ADD_TPL_SCH_MODE_BASE));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForBiasAdd(gert::TilingParseContext* context)
{
    if (context == nullptr) {
        OP_LOGE("BiasAdd", "tiling parse context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto compileInfo = context->GetCompiledInfo<BiasAddCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    compileInfo->coreNum = 0;
    compileInfo->ubSize = 0;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BiasAdd).Tiling(TilingForBiasAdd).TilingParse<BiasAddCompileInfo>(TilingPrepareForBiasAdd);
} // namespace optiling
