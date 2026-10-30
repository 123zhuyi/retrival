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
 * \file tan_tiling.cpp
 * \brief Tan Tiling implementation (arch32 - Ascend910B)
 *
 * Computes tiling parameters for Tan operator:
 * - Multi-core splitting: totalNum divided evenly across AI Cores
 * - UB splitting: each core processes data in chunks of ubFactor elements
 * - TilingKey selection: based on input dtype (float32 vs float16)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/tan_tiling_data.h"
#include "../op_kernel/tan_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// float32: inputQueue(x2) + outputQueue(x2) + tmpBuf1(x1) + tmpBuf2(x1) = 6 float-sized buffers
// Total bytes = ubFactor * 6 * sizeof(float)
constexpr int64_t BUFFER_NUM_FP32 = 6;
// float16: inputQueue(x2,half) + outputQueue(x2,half) + tmpBuf1(x1,float) + tmpBuf2(x1,float)
// Total bytes = ubFactor * (2*2 + 2*2 + 4 + 4) = ubFactor * 16 = ubFactor * sizeof(float) * 4
constexpr int64_t BUFFER_NUM_FP16 = 4;

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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TanTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape and dtype
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = storageShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();

    // 3. Get workspace size
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. Set TilingData
    TanTilingData* tiling = context->GetTilingData<TanTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TanTilingData), 0, sizeof(TanTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // Handle empty tensor
    if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_0));
        return ge::GRAPH_SUCCESS;
    }

    // 5. Multi-core splitting
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // 6. UB splitting and TilingKey selection
    int64_t ubCanUse = static_cast<int64_t>(ubSize);
    int64_t ubBlockSize = GetUbBlockSize(context);
    // Both paths compute internally in float32, so typeSize = 4
    constexpr int64_t typeSize = 4;

    if (dtype == ge::DT_FLOAT) {
        tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP32), ubBlockSize);
        context->SetTilingKey(GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_0));
    } else if (dtype == ge::DT_FLOAT16) {
        tiling->ubFactor = FloorAlign(FloorDiv((ubCanUse / typeSize), BUFFER_NUM_FP16), ubBlockSize);
        context->SetTilingKey(GET_TPL_TILING_KEY(TAN_TPL_SCH_MODE_1));
    } else {
        OP_LOGE(context, "Tan: unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTan([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TanCompileInfo {};

IMPL_OP_OPTILING(Tan).Tiling(TanTilingFunc).TilingParse<TanCompileInfo>(TilingParseForTan);

} // namespace optiling
