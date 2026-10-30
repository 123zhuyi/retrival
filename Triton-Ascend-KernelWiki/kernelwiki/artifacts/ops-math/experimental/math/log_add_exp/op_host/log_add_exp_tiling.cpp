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

/**
 * \file log_add_exp_tiling.cpp
 * \brief LogAddExp tiling implementation for arch32 (Ascend910B)
 *
 * Iteration 1: fp32 + non-broadcast path only.
 * Pre-embedded: fp16/bf16 TilingKey selection and broadcast TilingData fields.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/log_add_exp_tiling_data.h"
#include "../op_kernel/log_add_exp_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

// fp32: 2 input(x2 double buffer) + 1 output(x2 double buffer) + 2 tmp = 8 buffers of sizeof(float)
constexpr int64_t BUFFER_NUM_FP32 = 8;
// fp16/bf16: reserved for iteration 2
constexpr int64_t BUFFER_NUM_FP16 = 10;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context, bool useBinaryDoubling, int64_t expandRows, int64_t innerSize, ge::DataType dataType)
{
    (void)expandRows;
    (void)innerSize;
    (void)dataType;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    if (useBinaryDoubling) {
        // Modular broadcast: no workspace needed (broadcast done in CopyIn)
        currentWorkspace[0] = WS_SYS_SIZE;
    } else {
        currentWorkspace[0] = WS_SYS_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LogAddExpTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input/output shapes and dtype
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    auto inputY = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());

    auto outputZ = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputZ);
    (void)outputZ;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // Validate dtype
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "LogAddExp: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    // 3. Compute broadcast output shape (don't rely on GetOutputShape, recompute it)
    int64_t dimX = inputShapeX.GetDimNum();
    int64_t dimY = inputShapeY.GetDimNum();
    int64_t maxDim = (dimX > dimY) ? dimX : dimY;

    // Compute broadcast output shape using NumPy rules
    std::vector<int64_t> outShape(maxDim);
    bool needBroadcast = false;

    for (int64_t i = 0; i < maxDim; i++) {
        int64_t dimIdxX = dimX - 1 - i;
        int64_t dimIdxY = dimY - 1 - i;
        int64_t outIdx = maxDim - 1 - i;

        int64_t sizeX = (dimIdxX >= 0) ? inputShapeX.GetDim(dimIdxX) : 1;
        int64_t sizeY = (dimIdxY >= 0) ? inputShapeY.GetDim(dimIdxY) : 1;

        if (sizeX == sizeY) {
            outShape[outIdx] = sizeX;
        } else if (sizeX == 1) {
            outShape[outIdx] = sizeY;
            needBroadcast = true;
        } else if (sizeY == 1) {
            outShape[outIdx] = sizeX;
            needBroadcast = true;
        } else {
            OP_LOGE(context, "LogAddExp: incompatible shapes for broadcast");
            return ge::GRAPH_FAILED;
        }
    }

    // Compute total element count
    int64_t totalLength = 1;
    for (int64_t i = 0; i < maxDim; i++) {
        totalLength *= outShape[i];
    }

    // 6. Fill TilingData
    LogAddExpTilingData* tiling = context->GetTilingData<LogAddExpTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LogAddExpTilingData), 0, sizeof(LogAddExpTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // UB tiling
    // For FP16/BF16: 2 input queues(x2 double buf) * 2B + 1 output queue(x2 double buf) * 2B + 3 tmp * 4B = 24B/elem
    // For FP32:      2 input queues(x2 double buf) * 4B + 1 output queue(x2 double buf) * 4B + 2 tmp * 4B = 32B/elem
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t bytesPerElem = sizeof(float); // FP32 default
    int64_t bufferNum = BUFFER_NUM_FP32;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        bufferNum = BUFFER_NUM_FP16;
        bytesPerElem = sizeof(float); // dominant cost is fp32 tmp buffers
    }
    // ubFactor must be aligned to 32 bytes for both T_IN and T_COMPUTE
    // For FP16: 32/2=16 elems; for FP32: 32/4=8 elems. Use 16 as safe alignment.
    int64_t alignFactor = (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) ? 16 : 8;
    tiling->ubFactor = (FloorDiv((ubCanUse / bytesPerElem), bufferNum) / alignFactor) * alignFactor;
    if (tiling->ubFactor <= 0) {
        tiling->ubFactor = alignFactor;
    }

    // Broadcast info
    tiling->needBroadcast = needBroadcast ? 1 : 0;
    tiling->dimNum = maxDim;

    // -----------------------------------------------------------------------
    // Detect "2D leading-dim broadcast" for binary doubling optimization:
    //   x=(1,N) + y=(M,N)  OR  x=(M,N) + y=(1,N)
    // -----------------------------------------------------------------------
    bool useBinaryDoubling = false;
    int64_t innerSize = 0, expandRows = 0, expandSrcIsX = 1;

    if (needBroadcast && maxDim == 2) {
        std::vector<int64_t> xShapeAlignedTmp(2, 1), yShapeAlignedTmp(2, 1);
        for (int64_t i = 0; i < dimX; i++)
            xShapeAlignedTmp[2 - dimX + i] = inputShapeX.GetDim(i);
        for (int64_t i = 0; i < dimY; i++)
            yShapeAlignedTmp[2 - dimY + i] = inputShapeY.GetDim(i);

        if (xShapeAlignedTmp[0] == 1 && yShapeAlignedTmp[0] > 1 && xShapeAlignedTmp[1] == yShapeAlignedTmp[1]) {
            // x = (1, N), y = (M, N)
            useBinaryDoubling = true;
            innerSize = outShape[1];
            expandRows = outShape[0];
            expandSrcIsX = 1;
        } else if (yShapeAlignedTmp[0] == 1 && xShapeAlignedTmp[0] > 1 && xShapeAlignedTmp[1] == yShapeAlignedTmp[1]) {
            // x = (M, N), y = (1, N)
            useBinaryDoubling = true;
            innerSize = outShape[1];
            expandRows = outShape[0];
            expandSrcIsX = 0;
        }
    }

    // 5. Get workspace + multi-core blockFactor
    OP_CHECK_IF(
        GetWorkspaceSize(context, useBinaryDoubling, expandRows, innerSize, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    // Multi-core tiling
    tiling->totalLength = totalLength;
    tiling->blockFactor = CeilDiv(totalLength, coreNum);

    // For binary doubling, round blockFactor up to multiple of innerSize
    // so each core processes whole rows (simplifies per-core expansion).
    if (useBinaryDoubling && innerSize > 0) {
        tiling->blockFactor = CeilDiv(tiling->blockFactor, innerSize) * innerSize;
    }
    int64_t usedCoreNum = CeilDiv(totalLength, tiling->blockFactor);

    // Fill binary doubling fields
    // EXPERIMENT 11: enable binary doubling for ALL dtypes (tiling only)
    tiling->useBinaryDoubling = useBinaryDoubling ? 1 : 0;
    tiling->expandSrcIsX = expandSrcIsX;
    tiling->innerSize = innerSize;
    tiling->expandRows = expandRows;

    if (needBroadcast) {
        // Compute strides (right-aligned)
        std::vector<int64_t> xShapeAligned(maxDim, 1);
        std::vector<int64_t> yShapeAligned(maxDim, 1);

        // Right-align shapes
        for (int64_t i = 0; i < dimX; i++) {
            xShapeAligned[maxDim - dimX + i] = inputShapeX.GetDim(i);
        }
        for (int64_t i = 0; i < dimY; i++) {
            yShapeAligned[maxDim - dimY + i] = inputShapeY.GetDim(i);
        }

        // Compute strides (broadcast dim stride = 0)
        std::vector<int64_t> xStrides(maxDim);
        std::vector<int64_t> yStrides(maxDim);
        std::vector<int64_t> outStrides(maxDim);

        int64_t xStride = 1;
        int64_t yStride = 1;
        int64_t outStride = 1;

        for (int64_t i = maxDim - 1; i >= 0; i--) {
            // Output stride
            outStrides[i] = outStride;
            outStride *= outShape[i];

            // X stride (0 if broadcast dim)
            if (xShapeAligned[i] == 1 && outShape[i] > 1) {
                xStrides[i] = 0;
            } else {
                xStrides[i] = xStride;
                xStride *= xShapeAligned[i];
            }

            // Y stride (0 if broadcast dim)
            if (yShapeAligned[i] == 1 && outShape[i] > 1) {
                yStrides[i] = 0;
            } else {
                yStrides[i] = yStride;
                yStride *= yShapeAligned[i];
            }
        }

        // Fill TilingData
        for (int64_t i = 0; i < maxDim && i < 8; i++) {
            tiling->xShape[i] = xShapeAligned[i];
            tiling->yShape[i] = yShapeAligned[i];
            tiling->outShape[i] = outShape[i];
            tiling->xStrides[i] = xStrides[i];
            tiling->yStrides[i] = yStrides[i];
            tiling->outStrides[i] = outStrides[i];
        }
    }

    // 7. Set BlockDim and TilingKey
    context->SetBlockDim(usedCoreNum);

    if (dataType == ge::DT_FLOAT) {
        context->SetTilingKey(GET_TPL_TILING_KEY(LOG_ADD_EXP_TPL_SCH_MODE_FP32));
    } else if (dataType == ge::DT_FLOAT16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(LOG_ADD_EXP_TPL_SCH_MODE_FP16));
    } else if (dataType == ge::DT_BF16) {
        context->SetTilingKey(GET_TPL_TILING_KEY(LOG_ADD_EXP_TPL_SCH_MODE_BF16));
    } else {
        OP_LOGE(context, "LogAddExp: unsupported dtype for TilingKey");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForLogAddExp([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct LogAddExpCompileInfo {};

IMPL_OP_OPTILING(LogAddExp).Tiling(LogAddExpTilingFunc).TilingParse<LogAddExpCompileInfo>(TilingParseForLogAddExp);

} // namespace optiling
