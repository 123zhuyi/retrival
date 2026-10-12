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

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/erf_inv_tiling_data.h"
#include "../op_kernel/erf_inv_tiling_key.h"

namespace optiling {

struct ErfInvCompileInfo {};

static ge::graphStatus ErfInvTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalElems = 1;
    for (size_t i = 0; i < inputShape->GetStorageShape().GetDimNum(); i++) {
        totalElems *= inputShape->GetStorageShape().GetDim(i);
    }
    if (totalElems > UINT32_MAX) {
        OP_LOGE(context, "[ErfInv] totalElems exceeds UINT32_MAX");
        return ge::GRAPH_FAILED;
    }

    // Dynamic core count: query hardware, never hard-code
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    uint32_t elems32 = static_cast<uint32_t>(totalElems);
    uint32_t usedCoreNum = (elems32 < coreNum) ? elems32 : coreNum;
    context->SetBlockDim(usedCoreNum);

    // Pivot distribution: first 'pivot' cores get (baseElems + 1) elements
    uint32_t baseElems = elems32 / usedCoreNum;
    uint32_t pivot = elems32 % usedCoreNum;

    // Tiling strategy: 4096 float elements per tile (16 KB per fp32 queue — matches DMA burst).
    // UB usage (fp32, depth=2 queues):
    //   2 × 2 × 4096 × 4 (queues, 64 KB) + 7 × 4096 × 4 (float scratch, 112 KB) + ~544 (mask)
    //   ≈ 176 KB, fits in 192 KB UB on ascend910b / 910_93 / 950 (16 KB margin).
    uint32_t tileSize = 4096;
    uint32_t maxElems = baseElems + (pivot > 0 ? 1 : 0);
    uint32_t innerLoops = (maxElems + tileSize - 1) / tileSize;

    // Fill TilingData via GetTilingData (C++ POD, direct assignment)
    ErfInvTilingData* tiling = context->GetTilingData<ErfInvTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->baseElems = baseElems;
    tiling->pivot = pivot;
    tiling->tileSize = tileSize;
    tiling->innerLoops = innerLoops;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForErfInv([[maybe_unused]] gert::TilingParseContext* context)
{
    // AscendC operators can directly return SUCCESS; hardware info is obtained at runtime
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ErfInv).Tiling(ErfInvTilingFunc).TilingParse<ErfInvCompileInfo>(TilingParseForErfInv);
} // namespace optiling
