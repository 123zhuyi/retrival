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

#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/bernoulli_mask_tiling_data.h"
#include "../op_kernel/bernoulli_mask_tiling_key.h"

namespace optiling {
namespace {
struct BernoulliMaskCompileInfo {};

constexpr uint64_t MASK_ALIGN_ELEMENTS = 256;
constexpr uint64_t ASCENDC_RESERVED_UB_BYTES = 8 * 1024;
constexpr uint64_t MAX_TILE_ELEMENTS = 16 * 1024;
constexpr uint64_t WORK_BYTES_PER_ELEMENT = sizeof(float);

uint64_t CeilDiv(uint64_t value, uint64_t divisor) { return divisor == 0 ? 0 : (value + divisor - 1) / divisor; }

uint64_t AlignUp(uint64_t value, uint64_t alignment) { return CeilDiv(value, alignment) * alignment; }

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

ge::graphStatus GetTilingKey(ge::DataType dtype, uint64_t& key)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            key = BernoulliMaskKey::FLOAT16;
            break;
        case ge::DT_FLOAT:
            key = BernoulliMaskKey::FLOAT;
            break;
        case ge::DT_DOUBLE:
            key = BernoulliMaskKey::DOUBLE;
            break;
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            key = BernoulliMaskKey::UINT8_OR_BOOL;
            break;
        case ge::DT_INT8:
            key = BernoulliMaskKey::INT8;
            break;
        case ge::DT_INT16:
            key = BernoulliMaskKey::INT16;
            break;
        case ge::DT_INT32:
            key = BernoulliMaskKey::INT32;
            break;
        case ge::DT_INT64:
            key = BernoulliMaskKey::INT64;
            break;
        case ge::DT_BF16:
            key = BernoulliMaskKey::BFLOAT16;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BernoulliTiling(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto outputShape = context->GetOutputShape(0);
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    const int64_t signedElements = outputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(signedElements < 0, OP_LOGE(context, "Output shape size must be non-negative."),
                return ge::GRAPH_FAILED);
    const uint64_t totalElements = static_cast<uint64_t>(signedElements);
    bool maskAliasesOut = false;
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* maskAliasesOutAttr = attrs->GetInt(1);
    if (maskAliasesOutAttr != nullptr) {
        maskAliasesOut = *maskAliasesOutAttr != 0;
    }

    uint64_t tilingKey = 0;
    OP_CHECK_IF(GetTilingKey(outputDesc->GetDataType(), tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "Unsupported output dtype."), return ge::GRAPH_FAILED);

    uint32_t outputTypeBytes = 0;
    OP_CHECK_IF(!ge::TypeUtils::GetDataTypeLength(outputDesc->GetDataType(), outputTypeBytes) || outputTypeBytes == 0,
                OP_LOGE(context, "Failed to get output dtype length."), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    uint32_t coreNum = platform.GetCoreNum();
    OP_CHECK_IF(ubBytes <= ASCENDC_RESERVED_UB_BYTES || coreNum == 0,
                OP_LOGE(context, "Invalid platform UB/core count."), return ge::GRAPH_FAILED);

    // One packed-mask byte represents eight outputs. Charging one byte per element
    // is deliberately conservative and leaves room for queue alignment.
    const uint64_t bytesPerElement = WORK_BYTES_PER_ELEMENT + outputTypeBytes + 1;
    uint64_t tileElements = AlignDown((ubBytes - ASCENDC_RESERVED_UB_BYTES) / bytesPerElement, MASK_ALIGN_ELEMENTS);
    tileElements = std::min(tileElements, MAX_TILE_ELEMENTS);
    OP_CHECK_IF(tileElements == 0, OP_LOGE(context, "UB is too small for one aligned tile."), return ge::GRAPH_FAILED);

    uint64_t blockDim = std::min<uint64_t>(coreNum, std::max<uint64_t>(1, CeilDiv(totalElements, tileElements)));
    uint64_t elementsPerCore = AlignUp(CeilDiv(totalElements, blockDim), MASK_ALIGN_ELEMENTS);
    blockDim = std::max<uint64_t>(1, CeilDiv(totalElements, elementsPerCore));

    auto tilingData = context->GetTilingData<BernoulliMaskTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->totalElements = totalElements;
    tilingData->elementsPerCore = elementsPerCore;
    tilingData->tileElements = tileElements;
    tilingData->maskAliasesOut = maskAliasesOut ? 1 : 0;

    // The generated kernel ABI contains one workspace pointer even though the
    // kernel does not consume global workspace. Publish a zero-byte entry so
    // the standard L0 launcher preserves that ABI without charging this
    // kernel for additional device storage.
    auto workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    if (maskAliasesOut) {
        // The in-place expansion uses identical cross-core barrier counts
        // between safe output waves.
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context, "Failed to enable the deterministic multi-core schedule."),
                    return ge::GRAPH_FAILED);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BernoulliMaskTilingParse([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(BernoulliMask).Tiling(BernoulliTiling).TilingParse<BernoulliMaskCompileInfo>(BernoulliMaskTilingParse);
} // namespace optiling
