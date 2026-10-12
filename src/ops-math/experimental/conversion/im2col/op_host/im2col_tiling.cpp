/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include "graph/types.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "util/platform_util.h"
#include "../op_kernel/im2col_tiling_data.h"
#include "../op_kernel/im2col_tiling_key.h"

namespace optiling {
namespace {

constexpr size_t ATTR_KERNEL_SIZE = 0U;
constexpr size_t ATTR_STRIDE = 1U;
constexpr size_t ATTR_DILATION = 2U;
constexpr size_t ATTR_PADDING_MODE = 3U;
constexpr size_t ATTR_PADS = 4U;
constexpr size_t INPUT_X_INDEX = 0U;
constexpr size_t WORKSPACE_INDEX = 0U;
constexpr size_t WORKSPACE_COUNT = 1U;
constexpr size_t PAIR_VALUE_COUNT = 2U;
constexpr size_t PAIR_FIRST_INDEX = 0U;
constexpr size_t PAIR_SECOND_INDEX = 1U;
constexpr size_t NCHW_DIM_COUNT = 4U;
constexpr size_t PADS_VALUE_COUNT = 4U;
constexpr size_t INPUT_FACTOR_COUNT = 4U;
constexpr size_t ROW_FACTOR_COUNT = 5U;
constexpr size_t GROUP_FACTOR_COUNT = 4U;
constexpr size_t OUTPUT_CHANNEL_FACTOR_COUNT = 4U;
constexpr size_t NCHW_BATCH_DIM = 0U;
constexpr size_t NCHW_CHANNEL_DIM = 1U;
constexpr size_t NCHW_HEIGHT_DIM = 2U;
constexpr size_t NCHW_WIDTH_DIM = 3U;
constexpr size_t PAD_TOP_INDEX = 0U;
constexpr size_t PAD_BOTTOM_INDEX = 1U;
constexpr size_t PAD_LEFT_INDEX = 2U;
constexpr size_t PAD_RIGHT_INDEX = 3U;
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;
constexpr int64_t MAX_TILE_ELEMENTS = 32768;
constexpr int64_t CHANNEL_OUTPUT_MAX_BYTES = 16 * 1024;
constexpr int64_t DOUBLE_BUFFER_COUNT = 2;
constexpr int64_t CHANNEL_TRANSPOSE_TILE = 16;
constexpr int64_t CHANNEL_TRANSPOSE_MAX_KERNEL_EXTENT = 5;
constexpr int64_t VECTOR_REPEAT_BYTES = 256;
constexpr int64_t BOOL_GATHER_TILE_BYTES_PER_ELEMENT = 3;
constexpr int64_t CHANNEL_TEMPLATE_BASE_KERNEL_ROWS = 2;
constexpr int64_t COMPACT_TEMPLATE_MIN_CHANNEL_BATCH = 2;
constexpr int64_t SINGLE_BATCH_COUNT = 1;
constexpr int64_t SINGLE_CHANNEL_COUNT = 1;
constexpr int64_t UNIT_SPATIAL_STEP = 1;
constexpr int64_t POINTWISE_KERNEL_EXTENT = 1;
constexpr int64_t FLAT_GATHER_STORE_BLOCK_COUNT = 1;
constexpr int64_t INVALID_INDEX_GUARD_CHANNEL_COUNT = 1;
constexpr int64_t MIN_GROUP_BATCH = 1;
constexpr int64_t MIN_TILE_ELEMENTS = 1;
constexpr int64_t MIN_ACTIVE_CORE_COUNT = 1;
// Empirically tuned path-selection and scheduler thresholds.  Keep them named
// so later performance retuning does not leave unexplained numeric literals in
// the control flow.
constexpr int64_t FP_TEMPLATE_LARGE_KERNEL_MIN_AREA = 8;
constexpr int64_t FP_TEMPLATE_SMALL_KERNEL_MAX_AREA = 4;
constexpr int64_t FP_TEMPLATE_SMALL_KERNEL_MIN_SPATIAL = 512;
constexpr int64_t FLAT_GATHER_MIN_CHANNELS = 256;
constexpr int64_t BOOL_SINGLE_CHANNEL_DOWNSAMPLE_RATIO = 4;
constexpr int64_t CONTIGUOUS_BOOL_GROUP_MAX_OUTPUT_BYTES = 640;
constexpr int64_t CONTIGUOUS_BOOL_GROUP_MAX_CORES = 3;
constexpr int64_t BOOL_OUTPUT_BYTES_PER_CORE = 24 * 1024;
constexpr int64_t BOOL_TINY_POINTWISE_MAX_CHANNELS = 4;
constexpr int64_t BOOL_TINY_POINTWISE_MAX_ELEMENTS = 4096;
constexpr int64_t BOOL_TINY_TEMPLATE_MAX_CORES = 32;
constexpr int64_t BOOL_TINY_CHANNEL_MAX_OUTPUT_BYTES = 512;
constexpr int64_t BOOL_DEFAULT_MAX_CHANNELS_PER_CORE = 8;
constexpr int64_t BOOL_MEDIUM_BATCH_MIN_CHANNELS = 128;
constexpr int64_t BOOL_MEDIUM_BATCH_MAX_OUTPUT_BYTES = 768;
constexpr int64_t BOOL_MEDIUM_BATCH_TARGET_CORES = 8;
constexpr int64_t BOOL_SMALL_BATCH_MIN_CHANNELS = 16;
constexpr int64_t BOOL_SMALL_BATCH_MAX_CHANNELS = 32;
constexpr int64_t BOOL_SMALL_BATCH_MIN_OUTPUT_BYTES = 128;
constexpr int64_t BOOL_SMALL_BATCH_TARGET_CORES = 4;
constexpr int64_t BOOL_LARGE_PLANE_MIN_CHANNELS = 8;
constexpr int64_t BOOL_LARGE_PLANE_MIN_OUTPUT_BYTES = 1024;
constexpr int64_t BOOL_LARGE_PLANE_MAX_OUTPUT_BYTES = 4096;
constexpr int64_t BOOL_LARGE_PLANE_TARGET_CORES = 3;
constexpr int64_t FP_OUTPUT_BYTES_PER_CORE = 1800;
constexpr int64_t IDENTITY_BYTES_PER_CORE = 16 * 1024;
// DataCopyExtParams::blockCount is exposed as uint16_t, but DAV_2201 MTE
// encodes only 12 bits.  Values above 4095 are truncated by hardware and
// silently leave whole output rows unwritten.
constexpr int64_t DATA_COPY_MAX_BLOCK_COUNT = 4095;

struct Im2colCompileInfo {};

struct Im2colParams {
    int64_t n = 0;
    int64_t c = 0;
    int64_t h = 0;
    int64_t w = 0;
    int64_t kernelH = 0;
    int64_t kernelW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t totalRows = 0;
    int64_t totalGroups = 0;
    int64_t totalInputElements = 0;
    int64_t totalOutputElements = 0;
    int64_t typeSize = 0;
    int64_t blockSize = 0;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

static bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool SafeAdd(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static int64_t AlignUp(int64_t value, int64_t alignment)
{
    if (value < 0 || alignment <= 0 || value > std::numeric_limits<int64_t>::max() - (alignment - 1)) {
        return -1;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static int64_t CeilDiv(int64_t value, int64_t divisor) { return (value - 1) / divisor + 1; }

static bool SafeProduct(const int64_t* values, size_t count, int64_t& result)
{
    result = 1;
    for (size_t i = 0; i < count; ++i) {
        if (!SafeMul(result, values[i], result)) {
            return false;
        }
    }
    return true;
}

static bool ReadPair(gert::TilingContext* context, size_t attrIndex, const char* name, int64_t& first, int64_t& second,
                     bool positive)
{
    const auto* values = context->GetAttrs()->GetListInt(attrIndex);
    OP_CHECK_IF(values == nullptr, OP_LOGE(context, "%s is null", name), return false);
    OP_CHECK_IF(values->GetSize() != PAIR_VALUE_COUNT, OP_LOGE(context, "%s must contain two values", name),
                return false);
    const int64_t* data = values->GetData();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "%s data is null", name), return false);
    first = data[PAIR_FIRST_INDEX];
    second = data[PAIR_SECOND_INDEX];
    if (positive) {
        OP_CHECK_IF(first <= 0, OP_LOGE(context, "%s first value must be positive", name), return false);
        OP_CHECK_IF(second <= 0, OP_LOGE(context, "%s second value must be positive", name), return false);
    }
    return true;
}

static bool ReadPads(gert::TilingContext* context, Im2colParams& p)
{
    const auto* values = context->GetAttrs()->GetListInt(ATTR_PADS);
    OP_CHECK_IF(values == nullptr, OP_LOGE(context, "pads is null"), return false);
    OP_CHECK_IF(values->GetSize() != PADS_VALUE_COUNT, OP_LOGE(context, "pads must contain four values"), return false);
    const int64_t* data = values->GetData();
    OP_CHECK_IF(data == nullptr, OP_LOGE(context, "pads data is null"), return false);
    p.padTop = data[PAD_TOP_INDEX];
    p.padBottom = data[PAD_BOTTOM_INDEX];
    p.padLeft = data[PAD_LEFT_INDEX];
    p.padRight = data[PAD_RIGHT_INDEX];
    OP_CHECK_IF(p.padTop < 0, OP_LOGE(context, "padTop must be non-negative"), return false);
    OP_CHECK_IF(p.padBottom < 0, OP_LOGE(context, "padBottom must be non-negative"), return false);
    OP_CHECK_IF(p.padLeft < 0, OP_LOGE(context, "padLeft must be non-negative"), return false);
    OP_CHECK_IF(p.padRight < 0, OP_LOGE(context, "padRight must be non-negative"), return false);
    return true;
}

static ge::graphStatus ReadParams(gert::TilingContext* context, Im2colParams& p)
{
    const auto* inputShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& shape = inputShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != NCHW_DIM_COUNT,
                OP_LOGE(context, "Im2col DAV_2201 kernel requires rank-4 NCHW input"), return ge::GRAPH_FAILED);
    p.n = shape.GetDim(NCHW_BATCH_DIM);
    p.c = shape.GetDim(NCHW_CHANNEL_DIM);
    p.h = shape.GetDim(NCHW_HEIGHT_DIM);
    p.w = shape.GetDim(NCHW_WIDTH_DIM);
    OP_CHECK_IF(p.n <= 0, OP_LOGE(context, "N must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.c <= 0, OP_LOGE(context, "C must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.h <= 0, OP_LOGE(context, "H must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(p.w <= 0, OP_LOGE(context, "W must be positive"), return ge::GRAPH_FAILED);

    const auto* desc = context->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    p.dtype = desc->GetDataType();
    OP_CHECK_IF(
        p.dtype != ge::DT_FLOAT16 && p.dtype != ge::DT_FLOAT && p.dtype != ge::DT_BF16 && p.dtype != ge::DT_BOOL,
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(p.dtype)), return ge::GRAPH_FAILED);
    p.typeSize = ge::GetSizeByDataType(p.dtype);
    OP_CHECK_IF(p.typeSize <= 0, OP_LOGE(context, "invalid dtype size"), return ge::GRAPH_FAILED);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(!ReadPair(context, ATTR_KERNEL_SIZE, "ksizes", p.kernelH, p.kernelW, true),
                OP_LOGE(context, "invalid ksizes"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ReadPair(context, ATTR_STRIDE, "strides", p.strideH, p.strideW, true),
                OP_LOGE(context, "invalid strides"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ReadPair(context, ATTR_DILATION, "dilations", p.dilationH, p.dilationW, true),
                OP_LOGE(context, "invalid dilations"), return ge::GRAPH_FAILED);

    const char* paddingMode = attrs->GetStr(ATTR_PADDING_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingMode);
    OP_CHECK_IF(std::string_view(paddingMode) != "CALCULATED",
                OP_LOGE(context, "DAV_2201 aclnn path only accepts CALCULATED padding"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!ReadPads(context, p), OP_LOGE(context, "invalid pads"), return ge::GRAPH_FAILED);

    __int128 effectiveH = static_cast<__int128>(p.kernelH - 1) * p.dilationH + 1;
    __int128 effectiveW = static_cast<__int128>(p.kernelW - 1) * p.dilationW + 1;
    __int128 numeratorH = static_cast<__int128>(p.h) + p.padTop + p.padBottom - effectiveH;
    __int128 numeratorW = static_cast<__int128>(p.w) + p.padLeft + p.padRight - effectiveW;
    OP_CHECK_IF(numeratorH < 0, OP_LOGE(context, "calculated output height is empty"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(numeratorW < 0, OP_LOGE(context, "calculated output width is empty"), return ge::GRAPH_FAILED);
    __int128 outH = numeratorH / p.strideH + 1;
    __int128 outW = numeratorW / p.strideW + 1;
    OP_CHECK_IF(outH <= 0, OP_LOGE(context, "calculated output height must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outW <= 0, OP_LOGE(context, "calculated output width must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outH > std::numeric_limits<int64_t>::max(), OP_LOGE(context, "output height overflows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outW > std::numeric_limits<int64_t>::max(), OP_LOGE(context, "output width overflows"),
                return ge::GRAPH_FAILED);
    p.outH = static_cast<int64_t>(outH);
    p.outW = static_cast<int64_t>(outW);

    const int64_t inputFactors[] = {p.n, p.c, p.h, p.w};
    const int64_t rowFactors[] = {p.n, p.c, p.kernelH, p.kernelW, p.outH};
    const int64_t groupFactors[] = {p.n, p.c, p.kernelH, p.kernelW};
    OP_CHECK_IF(!SafeProduct(inputFactors, INPUT_FACTOR_COUNT, p.totalInputElements),
                OP_LOGE(context, "input size overflows"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeProduct(rowFactors, ROW_FACTOR_COUNT, p.totalRows), OP_LOGE(context, "row count overflows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeProduct(groupFactors, GROUP_FACTOR_COUNT, p.totalGroups),
                OP_LOGE(context, "group count overflows"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeMul(p.totalRows, p.outW, p.totalOutputElements), OP_LOGE(context, "output size overflows"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// For small feature maps with many channels, transpose a C16 slice to NHWC in
// UB, form several kernel-position planes with block-aligned UB copies, then
// transpose the planes back.  This mirrors the 910B TBE schedule and replaces
// one Gather stream per channel with two vnchwconv operations per C16 tile.
static bool TryResolveChannelTranspose(const Im2colParams& p, uint64_t ubSize, Im2colTilingData& td)
{
    if ((p.dtype != ge::DT_FLOAT16 && p.dtype != ge::DT_BF16 && p.dtype != ge::DT_FLOAT) ||
        (p.typeSize != static_cast<int64_t>(sizeof(uint16_t)) &&
         p.typeSize != static_cast<int64_t>(sizeof(uint32_t))) ||
        p.n != SINGLE_BATCH_COUNT || p.c < CHANNEL_TRANSPOSE_TILE || p.c % CHANNEL_TRANSPOSE_TILE != 0 ||
        p.strideH != UNIT_SPATIAL_STEP || p.strideW != UNIT_SPATIAL_STEP || p.dilationH != UNIT_SPATIAL_STEP ||
        p.dilationW != UNIT_SPATIAL_STEP || p.kernelH <= POINTWISE_KERNEL_EXTENT ||
        p.kernelW <= POINTWISE_KERNEL_EXTENT || p.kernelH > CHANNEL_TRANSPOSE_MAX_KERNEL_EXTENT ||
        p.kernelW > CHANNEL_TRANSPOSE_MAX_KERNEL_EXTENT || p.h > std::numeric_limits<uint16_t>::max() ||
        p.w > std::numeric_limits<uint16_t>::max() || p.outH > std::numeric_limits<uint16_t>::max() ||
        p.outW > std::numeric_limits<uint16_t>::max() || ubSize <= static_cast<uint64_t>(UB_RESERVE_BYTES)) {
        return false;
    }

    int64_t inputSpatial = 0;
    int64_t outputSpatial = 0;
    int64_t outputChannelElements = 0;
    if (!SafeMul(p.h, p.w, inputSpatial) || !SafeMul(p.outH, p.outW, outputSpatial) ||
        !SafeMul(p.kernelH, p.kernelW, outputChannelElements) ||
        !SafeMul(outputChannelElements, outputSpatial, outputChannelElements) || inputSpatial <= 0 ||
        outputSpatial <= 0) {
        return false;
    }
    const int64_t spatialAlignment = p.blockSize / p.typeSize;
    const int64_t inputSpatialAligned = AlignUp(inputSpatial, spatialAlignment);
    if (inputSpatialAligned <= 0) {
        return false;
    }
    if ((p.typeSize == static_cast<int64_t>(sizeof(uint16_t)) &&
         inputSpatialAligned / CHANNEL_TRANSPOSE_TILE > std::numeric_limits<uint8_t>::max()) ||
        (p.typeSize == static_cast<int64_t>(sizeof(uint32_t)) && inputSpatialAligned > DATA_COPY_MAX_BLOCK_COUNT)) {
        return false;
    }
    int64_t inputBufferBytes = 0;
    if (!SafeMul(CHANNEL_TRANSPOSE_TILE * inputSpatialAligned, p.typeSize, inputBufferBytes)) {
        return false;
    }

    const int64_t transposeTmpBytes = 0;
    int64_t groupTile = p.kernelH * p.kernelW;
    const int64_t usable = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    int64_t planeBufferBytes = 0;
    while (groupTile > 0) {
        // B16 and B32 inverse vnchwconv both consume 16 spatial rows per
        // repeat. B32 handles the two 8-float channel halves separately.
        const int64_t planeRowsAligned = AlignUp(groupTile * outputSpatial, CHANNEL_TRANSPOSE_TILE);
        if (planeRowsAligned <= 0) {
            return false;
        }
        const bool transposeRangeOk = planeRowsAligned / CHANNEL_TRANSPOSE_TILE <= std::numeric_limits<uint8_t>::max();
        if (transposeRangeOk && SafeMul(CHANNEL_TRANSPOSE_TILE * planeRowsAligned, p.typeSize, planeBufferBytes) &&
            DOUBLE_BUFFER_COUNT * inputBufferBytes + DOUBLE_BUFFER_COUNT * planeBufferBytes + transposeTmpBytes <=
                usable) {
            break;
        }
        --groupTile;
    }
    if (groupTile <= 0 || inputBufferBytes > std::numeric_limits<uint32_t>::max() ||
        planeBufferBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    td.fastChannel = IM2COL_TILING_FLAG_ENABLED;
    // Work items are C16 tiles.  The kernel derives the logical N*C count from
    // the shape, while the common work splitter uses this field.
    td.totalChannels = p.n * p.c / CHANNEL_TRANSPOSE_TILE;
    td.channelBatch = CHANNEL_TRANSPOSE_TILE;
    td.rawRowStrideElements = inputSpatialAligned;
    td.groupBatch = groupTile;
    td.outputChannelElements = outputChannelElements;
    td.outputGroupStrideElements = outputSpatial;
    td.outBufferBytes = static_cast<uint32_t>(planeBufferBytes);
    td.rawBufferBytes = static_cast<uint32_t>(inputBufferBytes);
    td.indexBufferBytes = static_cast<uint32_t>(inputBufferBytes);
    td.outWideBufferBytes = static_cast<uint32_t>(planeBufferBytes);
    td.rawWideBufferBytes = static_cast<uint32_t>(transposeTmpBytes);
    return true;
}

static bool TryResolveChannelBatch(const Im2colParams& p, uint64_t ubSize, Im2colTilingData& td)
{
    if (ubSize <= static_cast<uint64_t>(UB_RESERVE_BYTES) || p.h > std::numeric_limits<uint16_t>::max()) {
        return false;
    }

    const bool boolGather = p.dtype == ge::DT_BOOL;

    int64_t totalChannels = 0;
    int64_t inputPlaneElements = 0;
    int64_t outputChannelElements = 0;
    int64_t outputChannelBytes = 0;
    const int64_t outputFactors[] = {p.kernelH, p.kernelW, p.outH, p.outW};
    if (!SafeMul(p.n, p.c, totalChannels) || !SafeMul(p.h, p.w, inputPlaneElements) ||
        !SafeProduct(outputFactors, OUTPUT_CHANNEL_FACTOR_COUNT, outputChannelElements) || outputChannelElements <= 0 ||
        outputChannelElements > MAX_TILE_ELEMENTS || !SafeMul(outputChannelElements, p.typeSize, outputChannelBytes) ||
        outputChannelBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    const int64_t kernelArea = p.kernelH * p.kernelW;
    // With a 1x1 kernel and no padding, matching input/output spatial shapes
    // imply an exact copy.  Stride may be greater than one on a singleton
    // dimension, while dilation is irrelevant for a 1x1 kernel.
    const bool identity = p.kernelH == POINTWISE_KERNEL_EXTENT && p.kernelW == POINTWISE_KERNEL_EXTENT &&
                          p.padTop == 0 && p.padBottom == 0 && p.padLeft == 0 && p.padRight == 0 && p.outH == p.h &&
                          p.outW == p.w;
    const int64_t flatOutputStrideBytes = AlignUp(outputChannelBytes, p.blockSize);
    if (flatOutputStrideBytes <= 0) {
        return false;
    }
    const int64_t flatOutputStrideElements = flatOutputStrideBytes / p.typeSize;
    const bool contiguousRawTemplate = flatOutputStrideElements <=
                                       static_cast<int64_t>(IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS);
    // BOOL must widen to half before Gather on DAV_2201.  Once a complete
    // channel fits the bounded Gather template, gathering all kernel planes
    // at once avoids repeating int8<->half conversion for every plane.  FP
    // keeps the empirically safer large-channel threshold.
    const bool fpTemplateGain = kernelArea >= FP_TEMPLATE_LARGE_KERNEL_MIN_AREA ||
                                (kernelArea <= FP_TEMPLATE_SMALL_KERNEL_MAX_AREA &&
                                 p.outH * p.outW >= FP_TEMPLATE_SMALL_KERNEL_MIN_SPATIAL);
    const bool flatGather = !identity && contiguousRawTemplate &&
                            (boolGather || totalChannels >= FLAT_GATHER_MIN_CHANNELS || fpTemplateGain);
    const bool contiguousRaw = flatGather;
    if (boolGather && totalChannels == SINGLE_CHANNEL_COUNT && !identity) {
        int64_t inputChannelBytes = 0;
        if (!SafeMul(inputPlaneElements, p.typeSize, inputChannelBytes)) {
            return false;
        }
        // A single heavily downsampled BOOL channel spends most of its time
        // constructing/fetching the full-channel index template.  The group
        // path builds one short OW index and was measured substantially faster
        // for this class (for example Test001 and Test070).
        if (inputChannelBytes > BOOL_SINGLE_CHANNEL_DOWNSAMPLE_RATIO * outputChannelBytes) {
            return false;
        }
    }
    if (!identity && outputChannelBytes > CHANNEL_OUTPUT_MAX_BYTES) {
        return false;
    }

    int64_t outputGroupBytes = 0;
    int64_t outputGroupElements = 0;
    int64_t outputGroupStrideBytes = 0;
    int64_t outputChannelStrideBytes = 0;
    if (!SafeMul(p.outH, p.outW, outputGroupElements) || !SafeMul(outputGroupElements, p.typeSize, outputGroupBytes)) {
        return false;
    }
    outputGroupStrideBytes = AlignUp(outputGroupBytes, p.blockSize);
    if (flatGather || identity) {
        outputChannelStrideBytes = AlignUp(outputChannelBytes, p.blockSize);
    } else {
        // The padded-input path gathers one kernel position for every channel
        // in the batch.  A physical channel slot is one aligned OH*OW group;
        // MTE3 removes that padding and applies the logical channel stride.
        outputChannelStrideBytes = outputGroupStrideBytes;
    }
    int64_t rawRowBytes = 0;
    int64_t rawChannelBytes = 0;
    int64_t indexBytes = 0;
    int64_t indexBytesPerChannel = 0;
    int64_t rawInputBaseBytes = 0;
    if (!identity) {
        if (p.padLeft > std::numeric_limits<uint8_t>::max() || p.padRight > std::numeric_limits<uint8_t>::max()) {
            return false;
        }
        if (contiguousRaw) {
            rawInputBaseBytes = p.blockSize;
            int64_t inputChannelBytes = 0;
            int64_t inputPlaneWithBaseElements = 0;
            if (!SafeMul(inputPlaneElements, p.typeSize, inputChannelBytes) ||
                !SafeAdd(rawInputBaseBytes / p.typeSize, inputPlaneElements, inputPlaneWithBaseElements)) {
                return false;
            }
            if (!SafeMul(p.w, p.typeSize, rawRowBytes)) {
                return false;
            }
            if (boolGather || inputPlaneWithBaseElements <= std::numeric_limits<uint8_t>::max()) {
                // BOOL reuses one host index template for each channel.  Give
                // every raw channel its own zero prefix so invalid template
                // offsets remain zero when the Gather source base advances.
                int64_t rawChannelBytesUnaligned = 0;
                if (!SafeAdd(rawInputBaseBytes, inputChannelBytes, rawChannelBytesUnaligned)) {
                    return false;
                }
                rawChannelBytes = AlignUp(rawChannelBytesUnaligned, p.blockSize);
                if (rawChannelBytes <= 0) {
                    return false;
                }
            } else {
                rawChannelBytes = inputChannelBytes;
            }
            const int64_t indexElementsPerChannel = outputChannelStrideBytes / p.typeSize;
            int64_t channelIndexBytes = 0;
            if (!SafeMul(indexElementsPerChannel, static_cast<int64_t>(sizeof(uint32_t)), channelIndexBytes)) {
                return false;
            }
            if (boolGather) {
                // The BOOL path advances the Gather source for every channel,
                // so all channels reuse one host-expanded uint32 index vector.
                // Keeping it expanded removes compact-index casts from every
                // active core; the tiling payload is cached and bounded by the
                // fixed template capacity.
                indexBytes = channelIndexBytes;
            } else {
                indexBytesPerChannel = channelIndexBytes;
            }
        } else {
            int64_t paddedW = 0;
            int64_t paddedH = 0;
            if (p.w > std::numeric_limits<int64_t>::max() - p.padLeft ||
                p.w + p.padLeft > std::numeric_limits<int64_t>::max() - p.padRight ||
                p.h > std::numeric_limits<int64_t>::max() - p.padTop ||
                p.h + p.padTop > std::numeric_limits<int64_t>::max() - p.padBottom) {
                return false;
            }
            paddedW = p.w + p.padLeft + p.padRight;
            paddedH = p.h + p.padTop + p.padBottom;
            if (!SafeMul(paddedW, p.typeSize, rawRowBytes)) {
                return false;
            }
            rawRowBytes = AlignUp(rawRowBytes, p.blockSize);
            if (rawRowBytes <= 0) {
                return false;
            }
            if (!SafeMul(paddedH, rawRowBytes, rawChannelBytes) || rawChannelBytes <= 0) {
                return false;
            }
            const int64_t indexElementsPerChannel = (flatGather ? outputChannelStrideBytes : outputGroupStrideBytes) /
                                                    p.typeSize;
            if (!SafeMul(indexElementsPerChannel, static_cast<int64_t>(sizeof(uint32_t)), indexBytesPerChannel)) {
                return false;
            }
        }
    }

    const int64_t usable = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    int64_t outputBufferBytesPerChannel = outputChannelStrideBytes;
    if (!identity && !flatGather &&
        !SafeMul(outputBufferBytesPerChannel, DOUBLE_BUFFER_COUNT, outputBufferBytesPerChannel)) {
        return false;
    }
    int64_t wideOutputBytesPerChannel = 0;
    int64_t wideRawBytesPerChannel = 0;
    int64_t fixedBufferBytes = rawInputBaseBytes;
    if (boolGather && !identity) {
        // BOOL Gather is not available on DAV_2201.  Keep one byte-preserving
        // input/output buffer plus half-width Gather staging buffers.  This
        // lets a channel batch pay the int8<->half conversion once instead of
        // once per kernel group.
        if (!SafeMul(outputBufferBytesPerChannel, static_cast<int64_t>(sizeof(uint16_t)), wideOutputBytesPerChannel) ||
            !SafeMul(rawChannelBytes, static_cast<int64_t>(sizeof(uint16_t)), wideRawBytesPerChannel) ||
            !SafeMul(rawInputBaseBytes, BOOL_GATHER_TILE_BYTES_PER_ELEMENT, fixedBufferBytes)) {
            return false;
        }
    }
    int64_t bytesPerChannel = 0;
    int64_t reservedBytes = 0;
    if (!SafeAdd(outputBufferBytesPerChannel, rawChannelBytes, bytesPerChannel) ||
        !SafeAdd(bytesPerChannel, indexBytesPerChannel, bytesPerChannel) ||
        !SafeAdd(bytesPerChannel, wideOutputBytesPerChannel, bytesPerChannel) ||
        !SafeAdd(bytesPerChannel, wideRawBytesPerChannel, bytesPerChannel) ||
        !SafeAdd(indexBytes, fixedBufferBytes, reservedBytes) || bytesPerChannel <= 0 || reservedBytes > usable ||
        bytesPerChannel > usable - reservedBytes) {
        return false;
    }
    int64_t channelBatch = (usable - reservedBytes) / bytesPerChannel;
    channelBatch = std::min<int64_t>(channelBatch, totalChannels);
    // Level-2 Gather's repeat count is uint8_t, so cap the whole batch to the
    // largest encodable repeat count.
    const int64_t gatherTypeSize = boolGather ? static_cast<int64_t>(sizeof(uint16_t)) : p.typeSize;
    if (gatherTypeSize <= 0) {
        return false;
    }
    const int64_t gatherElementsPerRepeat = VECTOR_REPEAT_BYTES / gatherTypeSize;
    const int64_t maxGatherElements = gatherElementsPerRepeat *
                                      static_cast<int64_t>(std::numeric_limits<uint8_t>::max());
    if (!identity) {
        const int64_t gatherElementsPerChannel = outputChannelStrideBytes / p.typeSize;
        if (gatherElementsPerChannel <= 0) {
            return false;
        }
        channelBatch = std::min<int64_t>(channelBatch, maxGatherElements / gatherElementsPerChannel);
    }
    const int64_t storeBlocksPerChannel = flatGather ? FLAT_GATHER_STORE_BLOCK_COUNT : kernelArea;
    if (storeBlocksPerChannel <= 0) {
        return false;
    }
    channelBatch = std::min<int64_t>(channelBatch, std::numeric_limits<uint16_t>::max() / storeBlocksPerChannel);
    if (channelBatch <= 0) {
        return false;
    }

    int64_t outBufferBytes = 0;
    int64_t rawBufferBytes = 0;
    int64_t batchIndexBytes = indexBytes;
    if (!SafeMul(channelBatch, outputBufferBytesPerChannel, outBufferBytes) ||
        !SafeMul(channelBatch, rawChannelBytes, rawBufferBytes) ||
        (!identity && indexBytesPerChannel != 0 && !SafeMul(channelBatch, indexBytesPerChannel, batchIndexBytes)) ||
        rawBufferBytes > std::numeric_limits<int64_t>::max() - rawInputBaseBytes ||
        outBufferBytes > std::numeric_limits<uint32_t>::max() ||
        rawBufferBytes + rawInputBaseBytes > std::numeric_limits<uint32_t>::max() ||
        batchIndexBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    indexBytes = batchIndexBytes;

    td.fastChannel = IM2COL_TILING_FLAG_ENABLED;
    td.channelIdentity = identity ? IM2COL_TILING_FLAG_ENABLED : IM2COL_TILING_FLAG_DISABLED;
    td.channelFlatGather = flatGather ? IM2COL_TILING_FLAG_ENABLED : IM2COL_TILING_FLAG_DISABLED;
    td.channelContiguousRaw = contiguousRaw ? IM2COL_TILING_FLAG_ENABLED : IM2COL_TILING_FLAG_DISABLED;
    td.totalChannels = totalChannels;
    td.channelBatch = channelBatch;
    td.rawRowStrideElements = rawRowBytes / p.typeSize;
    td.rawChannelStrideElements = identity ? 0 : rawChannelBytes / p.typeSize;
    td.outputChannelElements = outputChannelElements;
    td.outputGroupStrideElements = outputGroupStrideBytes / p.typeSize;
    td.outputChannelStrideElements = outputChannelStrideBytes / p.typeSize;
    td.outRowStrideElements = outputGroupStrideBytes / p.typeSize;
    td.rawInputBaseElements = rawInputBaseBytes / p.typeSize;
    td.outBufferBytes = static_cast<uint32_t>(outBufferBytes);
    const int64_t rawBufferBytesWithBase = rawBufferBytes + rawInputBaseBytes;
    const int64_t rawBufferBytesAligned = AlignUp(rawBufferBytesWithBase, p.blockSize);
    if (rawBufferBytesAligned < 0 || rawBufferBytesAligned > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    td.rawBufferBytes = static_cast<uint32_t>(rawBufferBytesAligned);
    td.indexBufferBytes = static_cast<uint32_t>(indexBytes);
    if (boolGather && !identity) {
        int64_t outWideBufferBytes = 0;
        int64_t rawWideBufferBytes = 0;
        const int64_t rawBufferElements = rawBufferBytesWithBase;
        if (!SafeMul(outBufferBytes, static_cast<int64_t>(sizeof(uint16_t)), outWideBufferBytes) ||
            !SafeMul(rawBufferElements, static_cast<int64_t>(sizeof(uint16_t)), rawWideBufferBytes) ||
            outWideBufferBytes > std::numeric_limits<uint32_t>::max() ||
            rawWideBufferBytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const int64_t outWideBufferBytesAligned = AlignUp(outWideBufferBytes, p.blockSize);
        const int64_t rawWideBufferBytesAligned = AlignUp(rawWideBufferBytes, p.blockSize);
        if (outWideBufferBytesAligned <= 0 || rawWideBufferBytesAligned <= 0) {
            return false;
        }
        td.outWideBufferBytes = static_cast<uint32_t>(outWideBufferBytesAligned);
        td.rawWideBufferBytes = static_cast<uint32_t>(rawWideBufferBytesAligned);
    }
    const int64_t gatherElementBytes = boolGather ? static_cast<int64_t>(sizeof(uint16_t)) : p.typeSize;
    if (flatGather &&
        outputChannelStrideBytes / p.typeSize <= static_cast<int64_t>(IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS)) {
        if (!contiguousRaw) {
            const int64_t spatialTemplateElements = CHANNEL_TEMPLATE_BASE_KERNEL_ROWS * p.kernelW * outputGroupElements;
            if (spatialTemplateElements > static_cast<int64_t>(IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS) ||
                spatialTemplateElements * static_cast<int64_t>(sizeof(uint32_t)) % p.blockSize != 0) {
                return false;
            }
            std::fill_n(td.channelIndexTemplate, spatialTemplateElements, 0U);
            int64_t index = 0;
            for (int64_t kh = 0; kh < CHANNEL_TEMPLATE_BASE_KERNEL_ROWS; ++kh) {
                for (int64_t kw = 0; kw < p.kernelW; ++kw) {
                    for (int64_t oh = 0; oh < p.outH; ++oh) {
                        for (int64_t ow = 0; ow < p.outW; ++ow) {
                            const int64_t elementOffset = (oh * p.strideH + kh * p.dilationH) *
                                                              td.rawRowStrideElements +
                                                          ow * p.strideW + kw * p.dilationW;
                            td.channelIndexTemplate[index++] = static_cast<uint32_t>(elementOffset * p.typeSize);
                        }
                    }
                }
            }
            td.channelIndexTemplateValid = IM2COL_CHANNEL_INDEX_TEMPLATE_UINT32;
            td.channelIndexTemplateElements = static_cast<uint32_t>(spatialTemplateElements);
            return true;
        }
        int64_t invalidMagnitude = 0;
        int64_t inputPlaneWithBaseElements = 0;
        int64_t maxIndex = 0;
        if (!SafeAdd(channelBatch, INVALID_INDEX_GUARD_CHANNEL_COUNT, invalidMagnitude) ||
            !SafeMul(invalidMagnitude, inputPlaneElements, invalidMagnitude) ||
            !SafeMul(invalidMagnitude, gatherElementBytes, invalidMagnitude) ||
            !SafeAdd(rawInputBaseBytes / p.typeSize, inputPlaneElements, inputPlaneWithBaseElements) ||
            !SafeMul(inputPlaneWithBaseElements - 1, gatherElementBytes, maxIndex)) {
            return false;
        }
        const int64_t invalidIndex = -invalidMagnitude;
        const int64_t templateElements = outputChannelStrideBytes / p.typeSize;
        int64_t inputChannelBytes = 0;
        if (!SafeMul(inputPlaneElements, p.typeSize, inputChannelBytes)) {
            return false;
        }
        if (!boolGather && channelBatch >= COMPACT_TEMPLATE_MIN_CHANNEL_BATCH && rawChannelBytes > inputChannelBytes &&
            inputPlaneWithBaseElements <= std::numeric_limits<uint8_t>::max()) {
            const int64_t compactTemplateElements = AlignUp(templateElements, p.blockSize);
            if (compactTemplateElements > static_cast<int64_t>(IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS)) {
                return false;
            }
            auto* compactTemplate = reinterpret_cast<uint8_t*>(td.channelIndexTemplate);
            std::fill_n(compactTemplate, compactTemplateElements, static_cast<uint8_t>(0));
            int64_t index = 0;
            for (int64_t kh = 0; kh < p.kernelH; ++kh) {
                for (int64_t kw = 0; kw < p.kernelW; ++kw) {
                    for (int64_t oh = 0; oh < p.outH; ++oh) {
                        const int64_t inputH = oh * p.strideH + kh * p.dilationH - p.padTop;
                        for (int64_t ow = 0; ow < p.outW; ++ow) {
                            const int64_t inputW = ow * p.strideW + kw * p.dilationW - p.padLeft;
                            if (inputH >= 0 && inputH < p.h && inputW >= 0 && inputW < p.w) {
                                const int64_t elementOffset = rawInputBaseBytes / p.typeSize + inputH * p.w + inputW;
                                compactTemplate[index] = static_cast<uint8_t>(elementOffset);
                            }
                            ++index;
                        }
                    }
                }
            }
            td.channelIndexTemplateValid = IM2COL_CHANNEL_INDEX_TEMPLATE_UINT8;
            td.channelIndexTemplateElements = static_cast<uint32_t>(compactTemplateElements);
        } else if (!boolGather && channelBatch >= COMPACT_TEMPLATE_MIN_CHANNEL_BATCH &&
                   invalidIndex >= std::numeric_limits<int16_t>::min() &&
                   maxIndex <= std::numeric_limits<int16_t>::max()) {
            const int64_t compactTemplateElements = AlignUp(templateElements,
                                                            p.blockSize / static_cast<int64_t>(sizeof(uint16_t)));
            if (compactTemplateElements <= 0 ||
                compactTemplateElements > static_cast<int64_t>(IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS)) {
                return false;
            }
            auto* compactTemplate = reinterpret_cast<uint16_t*>(td.channelIndexTemplate);
            const uint16_t invalidBits = static_cast<uint16_t>(static_cast<int16_t>(invalidIndex));
            std::fill_n(compactTemplate, compactTemplateElements, invalidBits);
            int64_t index = 0;
            for (int64_t kh = 0; kh < p.kernelH; ++kh) {
                for (int64_t kw = 0; kw < p.kernelW; ++kw) {
                    for (int64_t oh = 0; oh < p.outH; ++oh) {
                        const int64_t inputH = oh * p.strideH + kh * p.dilationH - p.padTop;
                        for (int64_t ow = 0; ow < p.outW; ++ow) {
                            const int64_t inputW = ow * p.strideW + kw * p.dilationW - p.padLeft;
                            if (inputH >= 0 && inputH < p.h && inputW >= 0 && inputW < p.w) {
                                const int64_t elementOffset = rawInputBaseBytes / p.typeSize + inputH * p.w + inputW;
                                compactTemplate[index] = static_cast<uint16_t>(
                                    static_cast<int16_t>(elementOffset * gatherElementBytes));
                            }
                            ++index;
                        }
                    }
                }
            }
            td.channelIndexTemplateValid = IM2COL_CHANNEL_INDEX_TEMPLATE_INT16;
            td.channelIndexTemplateElements = static_cast<uint32_t>(compactTemplateElements);
        } else if (invalidIndex >= std::numeric_limits<int32_t>::min() &&
                   maxIndex <= std::numeric_limits<int32_t>::max()) {
            // BOOL has a private zero prefix for every raw channel and reuses
            // this vector with a moving Gather base, so invalid locations can
            // be clamped on the host.  Other dtypes retain negative sentinels
            // that the kernel adjusts for each channel before clamping.
            const uint32_t invalidBits = boolGather ? 0U : static_cast<uint32_t>(static_cast<int32_t>(invalidIndex));
            std::fill_n(td.channelIndexTemplate, templateElements, invalidBits);
            int64_t index = 0;
            for (int64_t kh = 0; kh < p.kernelH; ++kh) {
                for (int64_t kw = 0; kw < p.kernelW; ++kw) {
                    for (int64_t oh = 0; oh < p.outH; ++oh) {
                        const int64_t inputH = oh * p.strideH + kh * p.dilationH - p.padTop;
                        for (int64_t ow = 0; ow < p.outW; ++ow) {
                            const int64_t inputW = ow * p.strideW + kw * p.dilationW - p.padLeft;
                            if (inputH >= 0 && inputH < p.h && inputW >= 0 && inputW < p.w) {
                                const int64_t elementOffset = rawInputBaseBytes / p.typeSize + inputH * p.w + inputW;
                                td.channelIndexTemplate[index] = static_cast<uint32_t>(elementOffset *
                                                                                       gatherElementBytes);
                            }
                            ++index;
                        }
                    }
                }
            }
            td.channelIndexTemplateValid = IM2COL_CHANNEL_INDEX_TEMPLATE_UINT32;
            td.channelIndexTemplateElements = static_cast<uint32_t>(templateElements);
        }
    }
    return true;
}

static bool TryResolveGroupBatch(const Im2colParams& p, uint64_t ubSize, uint32_t path, Im2colTilingData& td)
{
    if (ubSize <= static_cast<uint64_t>(UB_RESERVE_BYTES)) {
        return false;
    }
    if (path == IM2COL_PATH_CONTIGUOUS_W) {
        const __int128 maxHorizontalPad = static_cast<__int128>(p.kernelW - 1) * p.dilationW +
                                          std::max(p.padLeft, p.padRight);
        if (maxHorizontalPad > std::numeric_limits<uint8_t>::max()) {
            return false;
        }
    }
    int64_t outRowBytes = 0;
    if (!SafeMul(p.outW, p.typeSize, outRowBytes) ||
        outRowBytes > std::numeric_limits<int64_t>::max() - (p.blockSize - 1)) {
        return false;
    }
    outRowBytes = AlignUp(outRowBytes, p.blockSize);

    int64_t rawElements = 0;
    int64_t rawRowBytes = 0;
    int64_t indexBytes = 0;
    if (path != IM2COL_PATH_CONTIGUOUS_W) {
        const __int128 raw = static_cast<__int128>(p.outW - 1) * p.strideW + 1;
        if (raw <= 0 || raw > std::numeric_limits<int64_t>::max()) {
            return false;
        }
        rawElements = static_cast<int64_t>(raw);
        if (!SafeMul(rawElements, p.typeSize, rawRowBytes) ||
            rawRowBytes > std::numeric_limits<int64_t>::max() - (p.blockSize - 1)) {
            return false;
        }
        rawRowBytes = AlignUp(rawRowBytes, p.blockSize);
        int64_t indexRawBytes = 0;
        if (!SafeMul(p.outW, static_cast<int64_t>(sizeof(uint32_t)), indexRawBytes) ||
            indexRawBytes > std::numeric_limits<int64_t>::max() - (p.blockSize - 1)) {
            return false;
        }
        indexBytes = AlignUp(indexRawBytes, p.blockSize);
    }

    int64_t sourceRowBytes = 0;
    if (!SafeMul(p.strideH, p.w, sourceRowBytes) || !SafeMul(sourceRowBytes, p.typeSize, sourceRowBytes) ||
        sourceRowBytes > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    int64_t perRowBytes = 0;
    if (!SafeAdd(outRowBytes, rawRowBytes, perRowBytes)) {
        return false;
    }
    if (path == IM2COL_PATH_GATHER_BOOL) {
        const int64_t outRowElements = outRowBytes / p.typeSize;
        const int64_t rawRowElements = rawRowBytes / p.typeSize;
        int64_t outWideBytes = 0;
        int64_t rawWideBytes = 0;
        if (!SafeMul(outRowElements, static_cast<int64_t>(sizeof(uint16_t)), outWideBytes) ||
            !SafeMul(rawRowElements, static_cast<int64_t>(sizeof(uint16_t)), rawWideBytes)) {
            return false;
        }
        int64_t wideBytes = 0;
        if (!SafeAdd(outWideBytes, rawWideBytes, wideBytes) || !SafeAdd(perRowBytes, wideBytes, perRowBytes)) {
            return false;
        }
    }
    const int64_t usable = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    if (perRowBytes <= 0 || indexBytes > usable || perRowBytes > usable - indexBytes) {
        return false;
    }
    const int64_t batchRows = std::min<int64_t>(
        p.outH, std::min<int64_t>(DATA_COPY_MAX_BLOCK_COUNT, (usable - indexBytes) / perRowBytes));
    if (batchRows <= 0) {
        return false;
    }

    int64_t groupBatch = MIN_GROUP_BATCH;
    if (path == IM2COL_PATH_CONTIGUOUS_W && batchRows == p.outH) {
        const int64_t groupBytes = batchRows * outRowBytes;
        const int64_t maxByBlockCount = DATA_COPY_MAX_BLOCK_COUNT / batchRows;
        const int64_t maxByUb = usable / groupBytes;
        groupBatch = std::min<int64_t>(p.totalGroups, std::min(maxByBlockCount, maxByUb));
        if (groupBatch <= 0) {
            return false;
        }
    }
    const int64_t outBufferBytes = groupBatch * batchRows * outRowBytes;
    const int64_t rawBufferBytes = batchRows * rawRowBytes;
    const int64_t outRowElements = outRowBytes / p.typeSize;
    const int64_t rawRowElements = path == IM2COL_PATH_CONTIGUOUS_W ? 0 : rawRowBytes / p.typeSize;
    const int64_t outWideBufferBytes = path == IM2COL_PATH_GATHER_BOOL ?
                                           batchRows * outRowElements * static_cast<int64_t>(sizeof(uint16_t)) :
                                           0;
    const int64_t rawWideBufferBytes = path == IM2COL_PATH_GATHER_BOOL ?
                                           batchRows * rawRowElements * static_cast<int64_t>(sizeof(uint16_t)) :
                                           0;
    const int64_t buffers[] = {outBufferBytes, rawBufferBytes, indexBytes, outWideBufferBytes, rawWideBufferBytes};
    for (int64_t bytes : buffers) {
        if (bytes < 0 || bytes > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }

    td.fastGroup = IM2COL_TILING_FLAG_ENABLED;
    td.tileElements = p.outW;
    td.rawElements = rawElements;
    td.batchRows = batchRows;
    td.outRowStrideElements = outRowElements;
    td.rawRowStrideElements = rawRowElements;
    td.groupBatch = groupBatch;
    td.outBufferBytes = static_cast<uint32_t>(outBufferBytes);
    td.rawBufferBytes = static_cast<uint32_t>(rawBufferBytes);
    td.indexBufferBytes = static_cast<uint32_t>(indexBytes);
    td.outWideBufferBytes = static_cast<uint32_t>(outWideBufferBytes);
    td.rawWideBufferBytes = static_cast<uint32_t>(rawWideBufferBytes);
    return true;
}

static int64_t BufferBytesForTile(int64_t tile, const Im2colParams& p, uint32_t path, Im2colTilingData& td)
{
    int64_t rawElements = 0;
    int64_t value = 0;
    if (path != IM2COL_PATH_CONTIGUOUS_W &&
        (!SafeMul(tile - 1, p.strideW, rawElements) || !SafeAdd(rawElements, 1, rawElements))) {
        return std::numeric_limits<int64_t>::max();
    }
    if (!SafeMul(tile, p.typeSize, value)) {
        return std::numeric_limits<int64_t>::max();
    }
    const int64_t outBytes = AlignUp(value, p.blockSize);
    int64_t rawBytes = 0;
    int64_t indexBytes = 0;
    int64_t outWideBytes = 0;
    int64_t rawWideBytes = 0;
    if (path != IM2COL_PATH_CONTIGUOUS_W) {
        if (!SafeMul(rawElements, p.typeSize, value)) {
            return std::numeric_limits<int64_t>::max();
        }
        rawBytes = AlignUp(value, p.blockSize);
        if (!SafeMul(tile, static_cast<int64_t>(sizeof(uint32_t)), value)) {
            return std::numeric_limits<int64_t>::max();
        }
        indexBytes = AlignUp(value, p.blockSize);
    }
    if (path == IM2COL_PATH_GATHER_BOOL) {
        if (!SafeMul(tile, static_cast<int64_t>(sizeof(uint16_t)), value)) {
            return std::numeric_limits<int64_t>::max();
        }
        outWideBytes = AlignUp(value, p.blockSize);
        if (!SafeMul(rawElements, static_cast<int64_t>(sizeof(uint16_t)), value)) {
            return std::numeric_limits<int64_t>::max();
        }
        rawWideBytes = AlignUp(value, p.blockSize);
    }
    int64_t total = 0;
    if (outBytes < 0 || rawBytes < 0 || indexBytes < 0 || outWideBytes < 0 || rawWideBytes < 0 ||
        !SafeAdd(outBytes, rawBytes, total) || !SafeAdd(total, indexBytes, total) ||
        !SafeAdd(total, outWideBytes, total) || !SafeAdd(total, rawWideBytes, total) ||
        total > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<int64_t>::max();
    }
    td.tileElements = tile;
    td.rawElements = rawElements;
    td.outBufferBytes = static_cast<uint32_t>(outBytes);
    td.rawBufferBytes = static_cast<uint32_t>(rawBytes);
    td.indexBufferBytes = static_cast<uint32_t>(indexBytes);
    td.outWideBufferBytes = static_cast<uint32_t>(outWideBytes);
    td.rawWideBufferBytes = static_cast<uint32_t>(rawWideBytes);
    return total;
}

static ge::graphStatus ResolveTile(gert::TilingContext* context, const Im2colParams& p, uint64_t ubSize, uint32_t path,
                                   Im2colTilingData& td)
{
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(UB_RESERVE_BYTES), OP_LOGE(context, "UB is too small"),
                return ge::GRAPH_FAILED);
    const int64_t usable = static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES;
    int64_t divisor = p.typeSize;
    if (path == IM2COL_PATH_GATHER_W) {
        int64_t strideWidth = 0;
        OP_CHECK_IF(!SafeAdd(p.strideW, 1, strideWidth) || !SafeMul(p.typeSize, strideWidth, divisor) ||
                        !SafeAdd(divisor, static_cast<int64_t>(sizeof(uint32_t)), divisor),
                    OP_LOGE(context, "tile divisor overflows"), return ge::GRAPH_FAILED);
    } else if (path == IM2COL_PATH_GATHER_BOOL) {
        int64_t strideWidth = 0;
        OP_CHECK_IF(!SafeAdd(p.strideW, 1, strideWidth) ||
                        !SafeMul(BOOL_GATHER_TILE_BYTES_PER_ELEMENT, strideWidth, divisor) ||
                        !SafeAdd(divisor, static_cast<int64_t>(sizeof(uint32_t)), divisor),
                    OP_LOGE(context, "tile divisor overflows"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(divisor <= 0, OP_LOGE(context, "tile divisor overflow"), return ge::GRAPH_FAILED);
    int64_t tile = std::min<int64_t>(p.outW, std::min<int64_t>(MAX_TILE_ELEMENTS, usable / divisor));
    tile = std::max<int64_t>(tile, MIN_TILE_ELEMENTS);
    while (tile > MIN_TILE_ELEMENTS && BufferBytesForTile(tile, p, path, td) > usable) {
        --tile;
    }
    OP_CHECK_IF(BufferBytesForTile(tile, p, path, td) > usable, OP_LOGE(context, "UB cannot hold one output tile"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void FillTilingData(const Im2colParams& p, int64_t baseRowsPerCore, int64_t extraRows, Im2colTilingData& td)
{
    td.n = p.n;
    td.c = p.c;
    td.h = p.h;
    td.w = p.w;
    td.kernelH = p.kernelH;
    td.kernelW = p.kernelW;
    td.strideH = p.strideH;
    td.strideW = p.strideW;
    td.dilationH = p.dilationH;
    td.dilationW = p.dilationW;
    td.padTop = p.padTop;
    td.padBottom = p.padBottom;
    td.padLeft = p.padLeft;
    td.padRight = p.padRight;
    td.outH = p.outH;
    td.outW = p.outW;
    td.totalRows = p.totalRows;
    td.totalGroups = p.totalGroups;
    td.totalInputElements = p.totalInputElements;
    td.totalOutputElements = p.totalOutputElements;
    td.baseRowsPerCore = baseRowsPerCore;
    td.extraRows = extraRows;
}

static ge::graphStatus Im2colTilingFunc(gert::TilingContext* context)
{
    Im2colParams p;
    auto status = ReadParams(context, p);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "invalid AIV core number %ld", coreNum), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "invalid UB size 0"), return ge::GRAPH_FAILED);
    p.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    OP_CHECK_IF(p.blockSize <= 0, OP_LOGE(context, "invalid UB block size %ld", p.blockSize), return ge::GRAPH_FAILED);

    const uint32_t path = p.strideW == UNIT_SPATIAL_STEP ?
                              IM2COL_PATH_CONTIGUOUS_W :
                              (p.dtype == ge::DT_BOOL ? IM2COL_PATH_GATHER_BOOL : IM2COL_PATH_GATHER_W);
    int64_t schedulerChannels = 0;
    int64_t schedulerOutputChannelBytes = 0;
    const bool schedulerShapeValid = SafeMul(p.n, p.c, schedulerChannels) && schedulerChannels > 0 &&
                                     p.totalOutputElements % schedulerChannels == 0 &&
                                     SafeMul(p.totalOutputElements / schedulerChannels, p.typeSize,
                                             schedulerOutputChannelBytes);
    // For these BOOL shapes strideW==1 makes every output row an MTE copy.
    // The group kernel can then batch many (N,C,kH,kW) planes in one UB
    // buffer without the byte->half->byte conversion required by Gather.
    const bool preferContiguousBoolGroup = schedulerShapeValid && p.dtype == ge::DT_BOOL &&
                                           p.strideW == UNIT_SPATIAL_STEP &&
                                           schedulerChannels == SINGLE_CHANNEL_COUNT &&
                                           schedulerOutputChannelBytes <= CONTIGUOUS_BOOL_GROUP_MAX_OUTPUT_BYTES;
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < sizeof(Im2colTilingData),
                OP_LOGE(context, "raw tiling capacity %zu is smaller than required %zu", rawTilingData->GetCapacity(),
                        sizeof(Im2colTilingData)),
                return ge::GRAPH_FAILED);
    auto* td = reinterpret_cast<Im2colTilingData*>(rawTilingData->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(Im2colTilingData), 0, sizeof(Im2colTilingData)) != EOK,
                OP_LOGE(context, "failed to clear tiling data"), return ge::GRAPH_FAILED);
    const bool fastTranspose = TryResolveChannelTranspose(p, ubSize, *td);
    const bool fastChannel = fastTranspose || (!preferContiguousBoolGroup && TryResolveChannelBatch(p, ubSize, *td));
    const bool fastGroup = !fastChannel && TryResolveGroupBatch(p, ubSize, path, *td);
    const int64_t workItems = fastChannel ? td->totalChannels : (fastGroup ? p.totalGroups : p.totalRows);
    int64_t usedCores = std::min<int64_t>(coreNum, workItems);
    if (preferContiguousBoolGroup && fastGroup) {
        usedCores = std::min<int64_t>(usedCores, CONTIGUOUS_BOOL_GROUP_MAX_CORES);
    }
    if (fastChannel && p.dtype == ge::DT_BOOL && td->channelIdentity == IM2COL_TILING_FLAG_DISABLED &&
        td->channelIndexTemplateValid != IM2COL_CHANNEL_INDEX_TEMPLATE_NONE) {
        // A compact template is read by every active core.  Give each core a
        // substantial channel batch to reduce duplicated GM reads, while
        // keeping the longest core's Gather/store payload near 24 KiB.
        const int64_t outputChannelBytes = td->outputChannelStrideElements * p.typeSize;
        OP_CHECK_IF(outputChannelBytes <= 0, OP_LOGE(context, "invalid output channel byte count"),
                    return ge::GRAPH_FAILED);
        const int64_t workloadChannels = std::max<int64_t>(SINGLE_CHANNEL_COUNT,
                                                           BOOL_OUTPUT_BYTES_PER_CORE / outputChannelBytes);
        // Tiny BOOL templates are launch/template-fetch bound above 32 AIVs.
        // Keep the same ceiling used by the reference small-shape schedule.
        // A very small batched 1x1 transform is better split by N so one core
        // consumes all channels of a batch item without excess core startup.
        const bool tinyBatchedPointwise = p.kernelH == POINTWISE_KERNEL_EXTENT &&
                                          p.kernelW == POINTWISE_KERNEL_EXTENT && p.n > SINGLE_BATCH_COUNT &&
                                          p.c <= BOOL_TINY_POINTWISE_MAX_CHANNELS &&
                                          p.totalOutputElements < BOOL_TINY_POINTWISE_MAX_ELEMENTS;
        const int64_t tinyCoreLimit = tinyBatchedPointwise ? p.n : BOOL_TINY_TEMPLATE_MAX_CORES;
        if (outputChannelBytes <= BOOL_TINY_CHANNEL_MAX_OUTPUT_BYTES) {
            usedCores = std::min<int64_t>(usedCores, tinyCoreLimit);
        }
        // Larger outputs normally amortize the template with up to 8 channels
        // per core (the measured optimum for Test053).  A narrow batched
        // medium-plane class benefits from much larger batches; conversely,
        // a few 1--4 KiB channels need about three cores to hide V/MTE waits.
        int64_t maxChannelsPerCore = outputChannelBytes <= BOOL_TINY_CHANNEL_MAX_OUTPUT_BYTES ?
                                         SINGLE_CHANNEL_COUNT :
                                         BOOL_DEFAULT_MAX_CHANNELS_PER_CORE;
        const bool batchedMediumPlane = p.n > SINGLE_BATCH_COUNT && p.c > SINGLE_CHANNEL_COUNT &&
                                        td->totalChannels >= BOOL_MEDIUM_BATCH_MIN_CHANNELS &&
                                        outputChannelBytes > BOOL_TINY_CHANNEL_MAX_OUTPUT_BYTES &&
                                        outputChannelBytes <= BOOL_MEDIUM_BATCH_MAX_OUTPUT_BYTES;
        if (batchedMediumPlane) {
            maxChannelsPerCore = td->channelBatch;
        }
        int64_t targetChannelsPerCore = std::min<int64_t>(td->channelBatch,
                                                          std::min(workloadChannels, maxChannelsPerCore));
        if (batchedMediumPlane) {
            // Keep eight AIVs busy for this medium-plane batch while still
            // amortizing the shared template across 32 channels per core.
            // Test081 is the representative 256-channel workload.
            targetChannelsPerCore = std::min<int64_t>(td->channelBatch,
                                                      CeilDiv(td->totalChannels, BOOL_MEDIUM_BATCH_TARGET_CORES));
        }
        if (td->totalChannels >= BOOL_SMALL_BATCH_MIN_CHANNELS && td->totalChannels <= BOOL_SMALL_BATCH_MAX_CHANNELS &&
            p.kernelH * p.kernelW > POINTWISE_KERNEL_EXTENT &&
            outputChannelBytes >= BOOL_SMALL_BATCH_MIN_OUTPUT_BYTES &&
            outputChannelBytes <= BOOL_TINY_CHANNEL_MAX_OUTPUT_BYTES) {
            // Tiny per-channel outputs are launch/template-fetch bound with
            // one channel per AIV.  Four AIVs amortize the shared template
            // while retaining enough parallelism (Test007/061/069 class).
            targetChannelsPerCore = std::min<int64_t>(td->channelBatch,
                                                      CeilDiv(td->totalChannels, BOOL_SMALL_BATCH_TARGET_CORES));
        }
        if (td->totalChannels >= BOOL_LARGE_PLANE_MIN_CHANNELS && td->totalChannels <= BOOL_SMALL_BATCH_MAX_CHANNELS &&
            outputChannelBytes > BOOL_LARGE_PLANE_MIN_OUTPUT_BYTES &&
            outputChannelBytes <= BOOL_LARGE_PLANE_MAX_OUTPUT_BYTES) {
            targetChannelsPerCore = std::min<int64_t>(td->channelBatch,
                                                      CeilDiv(td->totalChannels, BOOL_LARGE_PLANE_TARGET_CORES));
        }
        OP_CHECK_IF(targetChannelsPerCore <= 0, OP_LOGE(context, "invalid target channels per core"),
                    return ge::GRAPH_FAILED);
        const int64_t channelCores = (td->totalChannels - 1) / targetChannelsPerCore + 1;
        usedCores = std::min<int64_t>(usedCores, channelCores);
    }
    if (fastChannel && p.dtype != ge::DT_BOOL && td->channelIdentity == IM2COL_TILING_FLAG_DISABLED &&
        td->channelIndexTemplateValid != IM2COL_CHANNEL_INDEX_TEMPLATE_NONE) {
        // Very small FP template workloads become MTE2-latency bound when all
        // 40 cores fetch the same template.  Keep roughly 1.8 KiB of logical
        // output per core to reduce duplicate template traffic.
        int64_t totalOutputBytes = 0;
        if (SafeMul(p.totalOutputElements, p.typeSize, totalOutputBytes)) {
            const int64_t templateCores = std::max<int64_t>(MIN_ACTIVE_CORE_COUNT,
                                                            (totalOutputBytes - 1) / FP_OUTPUT_BYTES_PER_CORE + 1);
            usedCores = std::min<int64_t>(usedCores, templateCores);
        }
    }
    if (fastChannel && td->channelIdentity != IM2COL_TILING_FLAG_DISABLED) {
        int64_t totalOutputBytes = 0;
        if (SafeMul(p.totalOutputElements, p.typeSize, totalOutputBytes)) {
            const int64_t copyCores = std::max<int64_t>(MIN_ACTIVE_CORE_COUNT,
                                                        (totalOutputBytes - 1) / IDENTITY_BYTES_PER_CORE + 1);
            usedCores = std::min<int64_t>(usedCores, copyCores);
        }
    }
    OP_CHECK_IF(usedCores <= 0, OP_LOGE(context, "no work items to process"), return ge::GRAPH_FAILED);
    const int64_t baseWorkPerCore = workItems / usedCores;
    const int64_t extraWork = workItems % usedCores;

    FillTilingData(p, (fastChannel || fastGroup) ? 0 : baseWorkPerCore, (fastChannel || fastGroup) ? 0 : extraWork,
                   *td);
    if (fastChannel) {
        td->baseChannelsPerCore = baseWorkPerCore;
        td->extraChannels = extraWork;
    } else if (fastGroup) {
        td->baseGroupsPerCore = baseWorkPerCore;
        td->extraGroups = extraWork;
    } else {
        status = ResolveTile(context, p, ubSize, path, *td);
        if (status != ge::GRAPH_SUCCESS) {
            return status;
        }
    }

    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[WORKSPACE_INDEX] = 0;
    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    size_t tilingDataSize = sizeof(Im2colTilingHeader);
    if (td->channelIndexTemplateValid != IM2COL_CHANNEL_INDEX_TEMPLATE_NONE) {
        tilingDataSize = offsetof(Im2colTilingData, channelIndexTemplate) +
                         static_cast<size_t>(td->channelIndexTemplateElements) *
                             (td->channelIndexTemplateValid == IM2COL_CHANNEL_INDEX_TEMPLATE_UINT8 ? sizeof(uint8_t) :
                              td->channelIndexTemplateValid == IM2COL_CHANNEL_INDEX_TEMPLATE_INT16 ? sizeof(uint16_t) :
                                                                                                     sizeof(uint32_t));
    }
    OP_CHECK_IF(tilingDataSize > sizeof(Im2colTilingData),
                OP_LOGE(context, "tiling data size %zu exceeds capacity %zu", tilingDataSize, sizeof(Im2colTilingData)),
                return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(tilingDataSize);
    const bool boolChannelGather = fastChannel && p.dtype == ge::DT_BOOL &&
                                   td->channelIdentity == IM2COL_TILING_FLAG_DISABLED;
    const uint32_t selectedPath = fastTranspose ?
                                      IM2COL_PATH_CHANNEL_TRANSPOSE :
                                      (boolChannelGather ?
                                           IM2COL_PATH_GATHER_BOOL :
                                           (td->channelIndexTemplateValid != IM2COL_CHANNEL_INDEX_TEMPLATE_NONE ?
                                                IM2COL_PATH_CHANNEL_TEMPLATE :
                                                path));
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(p.dtype), selectedPath);
    OP_LOGI(context,
            "Im2col DAV_2201: path=%u dtype=%d rows=%ld groups=%ld channels=%ld cores=%ld "
            "fastTranspose=%u fastChannel=%u identity=%u flatGather=%u channelBatch=%ld fastGroup=%u tile=%ld "
            "batchRows=%ld groupBatch=%ld raw=%ld out=(%ld,%ld)",
            selectedPath, static_cast<int>(p.dtype), p.totalRows, p.totalGroups, td->totalChannels, usedCores,
            fastTranspose ? IM2COL_TILING_FLAG_ENABLED : IM2COL_TILING_FLAG_DISABLED, td->fastChannel,
            td->channelIdentity, td->channelFlatGather, td->channelBatch, td->fastGroup, td->tileElements,
            td->batchRows, td->groupBatch, td->rawElements, p.outH, p.outW);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Im2colTilingParse([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(Im2col).Tiling(Im2colTilingFunc).TilingParse<Im2colCompileInfo>(Im2colTilingParse);

} // namespace optiling
