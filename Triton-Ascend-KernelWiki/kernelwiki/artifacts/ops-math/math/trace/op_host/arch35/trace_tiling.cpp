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
 * \file trace_tiling.cpp
 * \brief Tiling implementation for trace operator
 *
 * Single-core strategy: needCoreNum = 1
 * Computes diagSize = min(M, N) and diagStride = stride0 + stride1
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/arch35/trace_tiling_data.h"
#include "../../op_kernel/arch35/trace_tiling_key.h"

namespace optiling {

using namespace Ops::Math::OpTiling;

constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

struct TraceCompileInfo {};

static ge::graphStatus TraceTilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    // 2. Get input shape
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto storageShape = inputShape->GetStorageShape();

    // 3. Dimension check (must be 2D)
    if (storageShape.GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "x",
            std::to_string(storageShape.GetDimNum()).c_str(),
            "trace expects a 2D matrix");
        return ge::GRAPH_FAILED;
    }

    int64_t M = storageShape.GetDim(0);
    int64_t N = storageShape.GetDim(1);

    // 4. Compute tiling parameters
    int64_t diagSize = (M < N) ? M : N;
    int64_t stride0 = N;  // Row stride for contiguous tensor
    int64_t stride1 = 1;  // Column stride for contiguous tensor
    int64_t diagStride = stride0 + stride1;

    // 5. Set tiling data
    TraceTilingData* tiling = context->GetTilingData<TraceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TraceTilingData), 0, sizeof(TraceTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->diagSize = diagSize;
    tiling->diagStride = diagStride;

    // 6. Set block dim (single core)
    context->SetBlockDim(1);

    // 7. Set workspace (system workspace only)
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = static_cast<size_t>(sysWorkspaceSize);

    // 8. Set local memory size
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
        OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize),
        return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(
        static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context, "SetLocalMemorySize failed"),
        return ge::GRAPH_FAILED);

    // 9. Set tiling key (single mode, dtype handled by DTYPE_X macro)
    context->SetTilingKey(GET_TPL_TILING_KEY(TRACE_TPL_MODE_DEFAULT));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTrace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Trace)
    .Tiling(TraceTilingFunc)
    .TilingParse<TraceCompileInfo>(TilingParseForTrace);
}  // namespace optiling
