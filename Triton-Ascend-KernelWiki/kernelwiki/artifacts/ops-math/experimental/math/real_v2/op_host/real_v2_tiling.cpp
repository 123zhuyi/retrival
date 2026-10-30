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
 * \file real_v2_tiling.cpp
 * \brief RealV2 operator Tiling implementation
 *
 * Full dtype coverage (FLOAT/FLOAT16/COMPLEX64/COMPLEX32),
 * IS_COMPLEX=0 real passthrough and IS_COMPLEX=1 Gather extraction.
 * Added empty tensor (0 elements) defense-in-depth and scalar tensor
 * normalization via EnsureNotScalar().
 */

#include <algorithm>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/real_v2_tiling_data.h"
#include "../op_kernel/real_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

struct DtypeInfo {
    uint64_t isComplex;
    int64_t typeSize;
    ge::DataType outputDtype;
};

static ge::graphStatus GetDtypeInfo(gert::TilingContext* context, ge::DataType inputDtype, DtypeInfo& info)
{
    switch (inputDtype) {
        case ge::DT_FLOAT:
            info = {0, 4, ge::DT_FLOAT};
            break;
        case ge::DT_FLOAT16:
            info = {0, 2, ge::DT_FLOAT16};
            break;
        case ge::DT_COMPLEX64:
            info = {1, 4, ge::DT_FLOAT};
            break;
        case ge::DT_COMPLEX32:
            info = {1, 2, ge::DT_FLOAT16};
            break;
        default:
            OP_LOGE(context, "Unsupported dtype: %d", static_cast<int>(inputDtype));
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static int64_t ComputeUbFactor(uint64_t isComplex, int64_t typeSize, int64_t ubSize, int64_t ubBlockSize)
{
    if (typeSize <= 0 || ubBlockSize <= 0) {
        return 0;
    }

    constexpr int64_t MAX_COPY_BYTES = 65504;

    if (isComplex == 0) {
        int64_t bufferNum = 1;
        return FloorAlign(FloorDiv(ubSize / typeSize, bufferNum), ubBlockSize);
    }
    int64_t bytesPerElement = 3 * typeSize + 4;
    int64_t ubCapacity = FloorAlign(ubSize / bytesPerElement, ubBlockSize);
    int64_t maxByBlockLen = FloorAlign(MAX_COPY_BYTES / (2 * typeSize), ubBlockSize);
    return std::min(ubCapacity, maxByBlockLen);
}

static ge::graphStatus RealV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape and dtype
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = EnsureNotScalar(inputShapePtr->GetStorageShape());

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    DtypeInfo dtypeInfo;
    OP_CHECK_IF(
        GetDtypeInfo(context, inputDesc->GetDataType(), dtypeInfo) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetDtypeInfo error"), return ge::GRAPH_FAILED);

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Compute tiling parameters
    RealV2TilingData* tiling = context->GetTilingData<RealV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RealV2TilingData), 0, sizeof(RealV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    int64_t totalOutputNum = inputShape.GetShapeSize();

    // Empty tensor: set minimal tiling and return early
    if (totalOutputNum <= 0) {
        context->SetBlockDim(0);
        uint32_t dType = static_cast<uint32_t>(dtypeInfo.outputDtype);
        ASCENDC_TPL_SEL_PARAM(context, dType, dtypeInfo.isComplex);
        return ge::GRAPH_SUCCESS;
    }

    tiling->totalOutputNum = totalOutputNum;
    tiling->blockFactor = CeilAlign(CeilDiv(totalOutputNum, coreNum), ubBlockSize);
    tiling->ubFactor =
        ComputeUbFactor(dtypeInfo.isComplex, dtypeInfo.typeSize, static_cast<int64_t>(ubSize), ubBlockSize);

    context->SetBlockDim(CeilDiv(totalOutputNum, tiling->blockFactor));

    // 5. Set TilingKey
    uint32_t dType = static_cast<uint32_t>(dtypeInfo.outputDtype);
    ASCENDC_TPL_SEL_PARAM(context, dType, dtypeInfo.isComplex);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRealV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RealV2CompileInfo {};

IMPL_OP_OPTILING(RealV2).Tiling(RealV2TilingFunc).TilingParse<RealV2CompileInfo>(TilingParseForRealV2);

} // namespace optiling
