/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/log/log.h"
#include "sign_bits_pack_tiling_arch35.h"
#include "sign_bits_pack_tiling_compute.h"

namespace optiling {

constexpr const char* kOpName = "SignBitsPack";

static bool GetPlatformAndInputSignBitsPack(gert::TilingContext* context, int64_t& n, int64_t& sizeAttr, int64_t& rank,
                                            ge::DataType& dtype, uint64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(kOpName, "GetPlatformInfo returned null");
        return false;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aivCores = platform.GetCoreNumAiv();
    coreNum = (aivCores == 0) ? 1u : static_cast<uint64_t>(aivCores);

    const gert::StorageShape* inShape = context->GetInputShape(0);
    if (inShape == nullptr) {
        OP_LOGE(kOpName, "GetInputShape(0) returned null");
        return false;
    }
    const gert::Shape& storageShape = inShape->GetStorageShape();
    rank = static_cast<int64_t>(storageShape.GetDimNum());
    n = storageShape.GetShapeSize();

    const gert::CompileTimeTensorDesc* inDesc = context->GetInputDesc(0);
    if (inDesc == nullptr) {
        OP_LOGE(kOpName, "GetInputDesc(0) returned null");
        return false;
    }
    dtype = inDesc->GetDataType();

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "GetAttrs returned null");
        return false;
    }
    const int64_t* sizePtr = attrs->GetInt(0);
    if (sizePtr == nullptr) {
        OP_LOGE(kOpName, "attribute 'size' returned null");
        return false;
    }
    sizeAttr = *sizePtr;

    return true;
}

static void ReportTilingError(SignBitsPackTilingStatus status, int64_t n, int64_t rank, ge::DataType dtype,
                              int64_t sizeAttr)
{
    switch (status) {
        case SignBitsPackTilingStatus::kDtypeNotSupported:
            OP_LOGE(kOpName, "dtype_not_supported: x.dtype=%d", static_cast<int>(dtype));
            break;
        case SignBitsPackTilingStatus::kShapeMismatch:
            OP_LOGE(kOpName, "shape_mismatch: x.rank=%ld, expected rank=1", static_cast<long>(rank));
            break;
        case SignBitsPackTilingStatus::kAttrOutOfRange:
            OP_LOGE(kOpName, "attr_out_of_range: size=%ld, n=%ld", static_cast<long>(sizeAttr), static_cast<long>(n));
            break;
        default:
            OP_LOGE(kOpName, "tiling failed, status=%d", static_cast<int>(status));
            break;
    }
}

static ge::graphStatus TilingFuncSignBitsPack(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Enter SignBitsPackTilingFunc");

    int64_t n = 0;
    int64_t sizeAttr = 0;
    int64_t rank = 0;
    ge::DataType dtype = ge::DT_UNDEFINED;
    uint64_t coreNum = 1;

    if (!GetPlatformAndInputSignBitsPack(context, n, sizeAttr, rank, dtype, coreNum)) {
        return ge::GRAPH_FAILED;
    }

    if (n <= 0) {
        context->SetTilingKey(0);
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    SignBitsPackTilingInputs inputs{};
    inputs.n = n;
    inputs.sizeAttr = sizeAttr;
    inputs.rank = rank;
    inputs.dtype = dtype;
    inputs.coreNum = coreNum;

    SignBitsPackTilingData* td = context->GetTilingData<SignBitsPackTilingData>();
    if (td == nullptr) {
        OP_LOGE(kOpName, "GetTilingData returned null");
        return ge::GRAPH_FAILED;
    }

    SignBitsPackTilingStatus status = ComputeTilingSignBitsPack(inputs, *td);

    if (status != SignBitsPackTilingStatus::kSuccess) {
        ReportTilingError(status, n, rank, dtype, sizeAttr);
        return ge::GRAPH_FAILED;
    }

    if (context->SetTilingKey(1) != ge::GRAPH_SUCCESS) {
        OP_LOGE(kOpName, "SetTilingKey(1) failed");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(static_cast<uint32_t>(td->realCoreNum));

    OP_LOGI(kOpName, "n=%ld, size=%ld, dtype=%d, coreNum=%lu, packedLen=%ld, realCoreNum=%u", static_cast<long>(n),
            static_cast<long>(sizeAttr), static_cast<int>(dtype), static_cast<unsigned long>(coreNum),
            static_cast<long>(td->packedLen), static_cast<unsigned>(td->realCoreNum));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSignBitsPack(gert::TilingParseContext* context)
{
    SignBitsPackCompileInfo* ci = context->GetCompiledInfo<SignBitsPackCompileInfo>();
    if (ci == nullptr) {
        OP_LOGE(kOpName, "GetCompiledInfo returned null");
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(kOpName, "GetPlatformInfo returned null");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    ci->coreNum = static_cast<uint64_t>(platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = ubSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SignBitsPack)
    .Tiling(TilingFuncSignBitsPack)
    .TilingParse<SignBitsPackCompileInfo>(TilingPrepareForSignBitsPack);

} // namespace optiling
