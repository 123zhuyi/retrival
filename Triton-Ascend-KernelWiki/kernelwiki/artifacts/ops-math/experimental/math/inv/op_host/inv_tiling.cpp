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
 * \file inv_tiling.cpp
 * \brief Inv tiling implementation (arch35)
 *
 * Tiling strategy:
 *   1. Multi-core: divide total elements evenly across AI Cores
 *   2. UB: divide per-core elements into UB-sized chunks
 *   3. Buffer layout: inputQueue(1 buf) + outputQueue(1 buf) + tmpBuf1 + tmpBuf2
 *
 * Unified formula for all dtypes:
 *   bytesPerElem = 2 * sizeof(T) + 2 * sizeof(float)
 *   ubFactor = FloorAlign(ubSize / bytesPerElem, ubBlockSize)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/inv_tiling_data.h"
#include "../op_kernel/inv_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus GetShapeInfo(gert::TilingContext* context, int64_t& totalElements, ge::DataType& dataType)
{
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto inputShape = EnsureNotScalar(inputSelf->GetStorageShape());
    totalElements = inputShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "Inv: unsupported dtype %d", static_cast<int>(dataType));
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

static ge::graphStatus InvTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get shape info
    int64_t totalElements = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeInfo(context, totalElements, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeInfo error"),
        return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    InvTilingData* tiling = context->GetTilingData<InvTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(InvTilingData), 0, sizeof(InvTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Determine typeSize based on dtype
    int64_t typeSize = 4; // default: sizeof(float)
    switch (dataType) {
        case ge::DT_FLOAT:
            typeSize = 4;
            break;
        case ge::DT_FLOAT16:
            typeSize = 2;
            break;
        case ge::DT_BF16:
            typeSize = 2;
            break;
        default:
            OP_LOGE(context, "Inv: unexpected dtype %d", static_cast<int>(dataType));
            return ge::GRAPH_FAILED;
    }

    int64_t ubBlockSize = 32 / typeSize; // 32-byte alignment in elements

    // Handle empty tensor: set blockDim=1, kernel will early-return
    if (totalElements == 0) {
        tiling->totalElements = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        uint32_t dTypeSelf = static_cast<uint32_t>(dataType);
        ASCENDC_TPL_SEL_PARAM(context, dTypeSelf);
        return ge::GRAPH_SUCCESS;
    }

    // Multi-core split
    int64_t blockFactor = CeilDiv(totalElements, coreNum);
    blockFactor = ((blockFactor + ubBlockSize - 1) / ubBlockSize) * ubBlockSize;
    int64_t usedCoreNum = CeilDiv(totalElements, blockFactor);

    // UB split
    // Unified formula: bytesPerElem = 2 * typeSize + 2 * sizeof(float)
    // ubDivisor = bytesPerElem / typeSize = (2*typeSize + 2*4) / typeSize
    int64_t bytesPerElem = 2 * typeSize + 2 * static_cast<int64_t>(sizeof(float));
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / bytesPerElem, ubBlockSize);
    OP_CHECK_IF(
        ubFactor <= 0, OP_LOGE(context, "Inv: ubFactor is %ld, UB too small", ubFactor), return ge::GRAPH_FAILED);

    tiling->totalElements = totalElements;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. Set TilingKey using ASCENDC_TPL_SEL_PARAM
    uint32_t dTypeSelf = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeSelf);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForInv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct InvCompileInfo {};

IMPL_OP_OPTILING(Inv).Tiling(InvTilingFunc).TilingParse<InvCompileInfo>(TilingParseForInv);

} // namespace optiling
