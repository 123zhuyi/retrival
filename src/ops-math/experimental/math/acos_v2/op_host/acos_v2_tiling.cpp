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
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/acos_v2_tiling_data.h"
#include "../op_kernel/acos_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t DOUBLE_BUFFER = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
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

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalLength, ge::DataType& dataType)
{
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShape = EnsureNotScalar(inputSelf->GetStorageShape());

    totalLength = inputShape.GetShapeSize();

    // Get dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Acos: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AcosV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape and dtype info
    int64_t totalLength;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalLength, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    AcosV2TilingData* tiling = context->GetTilingData<AcosV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AcosV2TilingData), 0, sizeof(AcosV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalLength = static_cast<uint32_t>(totalLength);

    // --- Empty tensor fast path ---
    // When totalLength == 0 (e.g. shape=[0] or shape=[3,0,4]),
    // set safe tiling parameters and return immediately.
    // blockDim must be >= 1 (at least one core launched).
    // tileLength must be > 0 to avoid division-by-zero in kernel loop calculation.
    // The kernel will see blockLength_ == 0 and skip all processing.
    if (totalLength <= 0) {
        tiling->blockFactor = 0;
        tiling->blockTailFactor = 0;
        tiling->tileLength = 1; // safe non-zero value for kernel loop guard

        context->SetBlockDim(1);

        uint64_t tilingKey = 0;
        if (dataType == ge::DT_FLOAT) {
            tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_FP32);
        } else if (dataType == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_FP16);
        } else {
            tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_BF16);
        }
        context->SetTilingKey(tilingKey);

        return ge::GRAPH_SUCCESS;
    }

    // Multi-core split
    int64_t blockFactor = CeilDiv(totalLength, coreNum);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    int64_t usedCoreNum = CeilDiv(totalLength, blockFactor);
    tiling->blockTailFactor = static_cast<uint32_t>(totalLength - blockFactor * (usedCoreNum - 1));

    // UB split: calculate tile size
    // The Acos advanced API (adv_api) internally uses PopStackBuffer to allocate
    // temporary buffer from remaining UB stack space. We must reserve enough space
    // for both the I/O queues AND the Acos API's internal temp buffers.
    //
    // Acos API temp buffer requirements (operates on fp32 data):
    //   ASIN_FLOAT_CALC_PROCEDURE = 4 -> needs 4 * tileLength * sizeof(float) bytes
    //
    // Total UB budget per element:
    //   fp32: inQueue(2*4) + outQueue(2*4) + AcosTemp(4*4) = 32 bytes/elem
    //   fp16: inQueue(2*2) + outQueue(2*2) + castBuf(4) + acosBuf(4) + AcosTemp(4*4) = 32 bytes/elem
    //   bf16: same as fp16 = 32 bytes/elem
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = GetUbBlockSize(context);

    int64_t bytesPerElement;

    if (dataType == ge::DT_FLOAT) {
        // fp32: inQueue double-buf (2*4) + outQueue double-buf (2*4) + Acos API temp (4*4) = 32
        bytesPerElement = static_cast<int64_t>(2 * DOUBLE_BUFFER * 4 + 4 * 4); // 32
    } else if (dataType == ge::DT_FLOAT16) {
        // fp16: inQueue double-buf (2*2) + outQueue double-buf (2*2) +
        //       castBuf(4) + acosBuf(4) + Acos API temp on fp32 (4*4) = 32
        bytesPerElement = static_cast<int64_t>(2 * DOUBLE_BUFFER * 2 + 2 * 4 + 4 * 4); // 32
    } else {
        // bf16: same memory layout as fp16 (cast to fp32 for Acos)
        bytesPerElement = static_cast<int64_t>(2 * DOUBLE_BUFFER * 2 + 2 * 4 + 4 * 4); // 32
    }

    int64_t tileLength = FloorAlign(FloorDiv(ubCanUse, bytesPerElement), ubBlockSize);

    tiling->tileLength = static_cast<uint32_t>(tileLength);

    // 5. Set block dim and tiling key
    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_FP32);
    } else if (dataType == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_FP16);
    } else if (dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(ACOS_MODE_BF16);
    } else {
        OP_LOGE(context, "Acos: unsupported dtype for tiling key");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAcosV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AcosCompileInfo {};

IMPL_OP_OPTILING(AcosV2).Tiling(AcosV2TilingFunc).TilingParse<AcosCompileInfo>(TilingParseForAcosV2);

} // namespace optiling
