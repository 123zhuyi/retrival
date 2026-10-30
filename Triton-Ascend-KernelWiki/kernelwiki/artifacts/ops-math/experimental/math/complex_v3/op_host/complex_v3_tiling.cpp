/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file complex_v3_tiling.cpp
 * \brief ComplexV3 operator tiling implementation (arch32)
 *
 * Computes tiling parameters for the ComplexV3 operator:
 * - Multi-core split: totalLength / coreNum
 * - UB split: considering realBuf + imagBuf + outBuf (outBuf is 2x)
 * - Broadcast detection and stride computation
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/complex_v3_tiling_data.h"
#include "../op_kernel/complex_v3_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Double buffer threshold: enable double buffer when data exceeds this
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// Get platform info: ubSize and coreNum
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Check if two shapes are equal
static bool ShapesEqual(const gert::Shape& a, const gert::Shape& b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < a.GetDimNum(); i++) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// Main tiling function
static ge::graphStatus ComplexV3TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get workspace info
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 3. Get input shapes
    auto inputReal = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputReal);
    auto realShape = EnsureNotScalar(inputReal->GetStorageShape());

    auto inputImag = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputImag);
    auto imagShape = EnsureNotScalar(inputImag->GetStorageShape());

    // 4. Get input dtype and compute type size
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = (dtype == ge::DT_FLOAT) ? 4 : 2;

    // 5. Determine broadcast mode
    bool needBroadcast = !ShapesEqual(realShape, imagShape);
    uint32_t broadcastMode = needBroadcast ? 1 : 0;

    // 6. Initialize tiling data
    ComplexV3TilingData* tiling = context->GetTilingData<ComplexV3TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ComplexV3TilingData), 0, sizeof(ComplexV3TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->broadcastMode = broadcastMode;

    // 7. Compute broadcast shape and total length
    int64_t totalLength = 0;
    if (!needBroadcast) {
        // Same shape: totalLength = product of shape dims
        totalLength = realShape.GetShapeSize();
    } else {
        // Broadcast: compute broadcast shape using NumPy rules
        uint32_t maxDim =
            std::max(static_cast<uint32_t>(realShape.GetDimNum()), static_cast<uint32_t>(imagShape.GetDimNum()));
        tiling->dimNum = maxDim;

        totalLength = 1;
        uint32_t realDims = static_cast<uint32_t>(realShape.GetDimNum());
        uint32_t imagDims = static_cast<uint32_t>(imagShape.GetDimNum());

        for (uint32_t d = 0; d < maxDim; d++) {
            int64_t realDim = (d < maxDim - realDims) ? 1 : realShape.GetDim(d - (maxDim - realDims));
            int64_t imagDim = (d < maxDim - imagDims) ? 1 : imagShape.GetDim(d - (maxDim - imagDims));
            tiling->outShape[d] = std::max(realDim, imagDim);
            totalLength *= tiling->outShape[d];
        }

        // Compute broadcast strides
        int64_t realAccStride = 1;
        int64_t imagAccStride = 1;
        for (int d = static_cast<int>(maxDim) - 1; d >= 0; d--) {
            int64_t realDim = (static_cast<uint32_t>(d) < maxDim - realDims) ?
                                  1 :
                                  realShape.GetDim(d - static_cast<int>(maxDim - realDims));
            int64_t imagDim = (static_cast<uint32_t>(d) < maxDim - imagDims) ?
                                  1 :
                                  imagShape.GetDim(d - static_cast<int>(maxDim - imagDims));

            tiling->realStride[d] = (realDim == 1) ? 0 : realAccStride;
            tiling->imagStride[d] = (imagDim == 1) ? 0 : imagAccStride;
            realAccStride *= realDim;
            imagAccStride *= imagDim;
        }

        // Record actual input sizes for UB preload in kernel
        tiling->realInputSize = realShape.GetShapeSize();
        tiling->imagInputSize = imagShape.GetShapeSize();
    }

    tiling->totalLength = totalLength;

    // 7.5. Empty tensor guard: if totalLength is 0, set minimal tiling and return early
    if (totalLength == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1); // Launch at least 1 core (kernel will exit immediately)
        uint32_t dType = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, dType, broadcastMode);
        return ge::GRAPH_SUCCESS;
    }

    // 8. Multi-core split
    tiling->blockFactor = CeilDiv(totalLength, coreNum);
    int64_t usedCoreNum = CeilDiv(totalLength, tiling->blockFactor);

    // 9. UB split
    // No-broadcast mode: realBuf(1) + imagBuf(1) + outBuf(2) = 4 parts per buffer
    // Broadcast mode: realBuf + imagBuf preloaded once, outBuf(2) for double buffer
    // Double buffer: multiply by 2
    int64_t ubBlockSize = GetUbBlockSize(context);
    bool useDoubleBuffer = (totalLength > MIN_SPLIT_THRESHOLD);
    if (!needBroadcast) {
        // UB layout for no-broadcast:
        //   realBuf(ubFactor) + imagBuf(ubFactor) + outBuf(2*ubFactor)
        // Double buffer: realBuf/imagBuf/outBuf are doubled (x2)
        // Total coefficient: 4 (single buffer) or 8 (double buffer)
        int64_t bufferCoeff = useDoubleBuffer ? 8 : 4;
        tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, bufferCoeff), ubBlockSize);
    } else {
        // Broadcast: check if inputs can be fully preloaded into UB
        int64_t realInputElems = tiling->realInputSize;
        int64_t imagInputElems = tiling->imagInputSize;
        // Align input buffer sizes to 32 bytes
        int64_t realBufElems = CeilDiv(realInputElems * typeSize, static_cast<int64_t>(32)) * 32 / typeSize;
        int64_t imagBufElems = CeilDiv(imagInputElems * typeSize, static_cast<int64_t>(32)) * 32 / typeSize;
        int64_t preloadBytes = (realBufElems + imagBufElems) * typeSize;

        // outBuf: ubFactor*2 elements (interleaved) x BUFFER_NUM(2) queue slots = 4
        int64_t outBufCoeff = 4; // 2 (interleave) * 2 (BUFFER_NUM queue depth)

        if (preloadBytes < static_cast<int64_t>(ubSize) / 2) {
            // Full preload mode: inputs fit in <50% of UB
            tiling->preloadMode = 1;
            int64_t remainBytes = static_cast<int64_t>(ubSize) - preloadBytes;
            tiling->ubFactor = FloorAlign(FloorDiv(remainBytes / typeSize, outBufCoeff), ubBlockSize);
        } else {
            // On-demand mode: inputs too large, use UB only for output + temp buffers
            // Layout: outBuf(ubFactor*2) x2 slots + realTmp(ubFactor) + imagTmp(ubFactor)
            // Total = ubFactor * (4 + 1 + 1) = ubFactor * 6
            tiling->preloadMode = 0;
            int64_t onDemandCoeff = outBufCoeff + 2; // +2 for realTmp + imagTmp (each ubFactor elems)
            tiling->ubFactor =
                FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, onDemandCoeff), ubBlockSize);
        }
    }

    // 10. Set BlockDim and TilingKey
    context->SetBlockDim(usedCoreNum);

    // TilingKey parameters match ASCENDC_TPL_ARGS_DECL order: D_T, BROADCAST_MODE
    uint32_t dType = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, dType, broadcastMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForComplexV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ComplexV3CompileInfo {}; // Required for graph mode dependency

IMPL_OP_OPTILING(ComplexV3).Tiling(ComplexV3TilingFunc).TilingParse<ComplexV3CompileInfo>(TilingParseForComplexV3);

} // namespace optiling
