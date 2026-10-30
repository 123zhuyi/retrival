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
 * \file cosh_v2_tiling.cpp
 * \brief CoshV2 operator tiling implementation (arch32 / Ascend910B)
 *
 * Tiling strategy:
 *   - Multi-core: totalNum / coreNum elements per core
 *   - UB split: ubSize / (computeTypeSize * bufferNum) elements per UB loop
 *   - Double buffer when totalNum > MIN_SPLIT_THRESHOLD
 *
 * Buffer count:
 *   fp16/fp32: inputQueue + outputQueue + tmpExpPos + tmpExpNeg = 4 buffers
 *   bf16:      inputQueue(bf16) + outputQueue(bf16) + castBuf(fp32) + tmpExpPos(fp32) + tmpExpNeg(fp32)
 *              simplified to 5 float-equivalent buffers
 *   Double buffer multiplies queue buffers by 2
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/cosh_v2_tiling_data.h"
#include "../op_kernel/cosh_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus CoshV2TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    int64_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. Get input info
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    int64_t totalNum = storageShape.GetShapeSize();
    if (totalNum == 0)
        totalNum = 1;

    // 3. Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 4. TilingData
    CoshV2TilingData* tiling = context->GetTilingData<CoshV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(CoshV2TilingData), 0, sizeof(CoshV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB split calculation
    // fp32: in + tmpExpPos + tmpExpNeg + out = 4 float buffers
    //   double buffer: (in + out) * 2 + (tmpExpPos + tmpExpNeg) = 2*2 + 2 = 6; simplified: 8 (4*2)
    // fp16/bf16: compute in fp32 for precision + numerical stability
    //   in(half/bf16) + out(half/bf16) + castBuf(fp32) + tmpExpPos(fp32) + tmpExpNeg(fp32)
    //   simplified to 5 float-equivalent buffers (conservative for half-size in/out)
    //   double buffer: 10 (5*2)
    int64_t ubBlockSize = GetUbBlockSize(context);
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    int64_t computeTypeSize;
    int64_t bufferNum;
    if (dataType == ge::DT_FLOAT) {
        computeTypeSize = 4;
        bufferNum = useDoubleBuffer ? 8 : 4;
    } else {
        // fp16 and bf16: compute in float, need castBuf
        computeTypeSize = 4; // compute in float
        bufferNum = useDoubleBuffer ? 10 : 5;
    }

    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / computeTypeSize, bufferNum), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5. TilingKey - parameter order matches ASCENDC_TPL_ARGS_DECL in cosh_v2_tiling_key.h
    uint32_t dTypeParam = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeParam, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCoshV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct CoshV2CompileInfo {};

IMPL_OP_OPTILING(CoshV2).Tiling(CoshV2TilingFunc).TilingParse<CoshV2CompileInfo>(TilingParseForCoshV2);

} // namespace optiling
