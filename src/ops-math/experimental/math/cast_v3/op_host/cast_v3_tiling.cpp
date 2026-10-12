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
 * \file cast_v3_tiling.cpp
 * \brief CastV3 tiling implementation (experimental, ascend310p)
 */
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/cast_tiling_data.h"

namespace optiling {

constexpr uint32_t RESERVED_UB_SIZE = 0;
constexpr uint64_t BUFFER_NUM = 2;
constexpr uint32_t PROCESS_SIZE = 256;

struct CastV3CompileInfo {};

class CastV3Tiling {
public:
    explicit CastV3Tiling(gert::TilingContext* ctx) : context(ctx) {}

    ge::graphStatus Init()
    {
        OP_CHECK_IF(InitFromInput() != ge::GRAPH_SUCCESS, OP_LOGE(context, "InitFromInput failed"),
                    return ge::GRAPH_FAILED);

        fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
        auto ascendCPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = static_cast<uint32_t>(ascendCPlatform.GetCoreNumAiv());

        size_t sysWorkspaceSize = ascendCPlatform.GetLibApiWorkSpaceSize();
        size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkSpace);
        currentWorkSpace[0] = sysWorkspaceSize + 2 * 1024;

        BaseTiling();
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetKernelTiling()
    {
        CastTilingData* tiling = context->GetTilingData<CastTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(memset_s(tiling, sizeof(CastTilingData), 0, sizeof(CastTilingData)) != EOK,
                    OP_LOGE(context, "memset tiling data failed"), return ge::GRAPH_FAILED);

        tiling->batchSize = batchSize;
        tiling->formerCoreNum = formerCoreNum;
        tiling->formerBatchSize = formerBatchSize;
        tiling->tailBatchSize = tailBatchSize;
        tiling->ubProcessNum = ubProcessNum;
        tiling->tilingKey = tilingKey;

        context->SetBlockDim(coreNum);
        return ge::GRAPH_SUCCESS;
    }

private:
    gert::TilingContext* context;

    uint64_t ubSize = 0;
    uint32_t coreNum = 1;
    uint32_t tilingKey = 1;
    int32_t ubProcessNum = 0;
    int64_t batchSize = 1;
    uint32_t inputTypeLength = 0;
    uint32_t outputTypeLength = 0;
    uint64_t formerBatchSize = 0;
    uint64_t tailBatchSize = 0;
    uint64_t formerCoreNum = 0;

    ge::graphStatus InitFromInput()
    {
        auto inputShapePtr = context->GetInputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
        auto xShape = inputShapePtr->GetStorageShape();
        int64_t totalElements = 1;
        for (size_t i = 0; i < xShape.GetDimNum(); i++) {
            totalElements *= xShape.GetDim(i);
        }
        OP_CHECK_IF(totalElements == 0, OP_LOGE(context, "CastV3 input shape has zero dimension, product is 0"),
                    return ge::GRAPH_FAILED);
        batchSize = totalElements;
        auto inputDescPtr = context->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
        auto outputDescPtr = context->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputDescPtr);
        ge::TypeUtils::GetDataTypeLength(inputDescPtr->GetDataType(), inputTypeLength);
        ge::TypeUtils::GetDataTypeLength(outputDescPtr->GetDataType(), outputTypeLength);
        OP_CHECK_IF(inputTypeLength == 0, OP_LOGE(context, "CastV3 input type length is 0"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(outputTypeLength == 0, OP_LOGE(context, "CastV3 output type length is 0"), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    void BaseTiling()
    {
        auto processNum = static_cast<uint64_t>(batchSize) / (PROCESS_SIZE / inputTypeLength);
        coreNum = std::min(static_cast<uint64_t>(coreNum), processNum);
        coreNum = std::max(coreNum, 1u);
        formerBatchSize = processNum / coreNum * (PROCESS_SIZE / inputTypeLength);
        formerCoreNum = (static_cast<uint64_t>(batchSize) == formerBatchSize * coreNum) ? coreNum : coreNum - 1;
        tailBatchSize = formerBatchSize + static_cast<uint64_t>(batchSize) - formerBatchSize * coreNum;
        SetTilingKey();
    }

    inline bool is1ByteType(ge::DataType dtype)
    {
        return dtype == ge::DT_BOOL || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8;
    }

    // Tiling key:
    //   1: int16 / int64 / half->int16  (CastBf16 kernel class)
    //   2: bf16 input / output          (CastBf16 kernel class, handled by compile-time dispatch in kernel)
    //   4: both 1-byte types            (CastCopy kernel class)
    //   5: 1-byte -> wider              (CastExpand kernel class)
    //   6: generic dtype cast           (CastGeneric kernel class)
    void SetTilingKey()
    {
        int64_t availableUbSize = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
        auto inputDtype = context->GetInputDesc(0)->GetDataType();
        auto outputDtype = context->GetOutputDesc(0)->GetDataType();

        if (inputDtype == ge::DT_BF16 || outputDtype == ge::DT_BF16) {
            // BF16 cases go through CastBf16 via compile-time type dispatch in kernel entry
            tilingKey = 2;
            ubProcessNum = static_cast<int32_t>(
                availableUbSize /
                (BUFFER_NUM * (inputTypeLength + outputTypeLength) + sizeof(float) + sizeof(int32_t)) / 256 * 256);
        } else if (inputDtype == ge::DT_INT64) {
            tilingKey = 1;
            ubProcessNum = static_cast<int32_t>(
                availableUbSize / (BUFFER_NUM * (inputTypeLength + outputTypeLength) + 2 * sizeof(int64_t)) / 256 *
                256);
        } else if (inputDtype == ge::DT_INT16 || (inputDtype == ge::DT_FLOAT16 && outputDtype == ge::DT_INT16)) {
            tilingKey = 1;
            ubProcessNum = static_cast<int32_t>(
                availableUbSize /
                (BUFFER_NUM * (inputTypeLength + outputTypeLength) + sizeof(float) + sizeof(int32_t)) / 256 * 256);
        } else if (is1ByteType(inputDtype) && is1ByteType(outputDtype)) {
            tilingKey = 4;
            ubProcessNum = static_cast<int32_t>(availableUbSize / (BUFFER_NUM * inputTypeLength) / 256 * 256);
        } else if (is1ByteType(inputDtype) && outputDtype == ge::DT_FLOAT16) {
            tilingKey = 5;
            ubProcessNum = static_cast<int32_t>(availableUbSize / (BUFFER_NUM * (inputTypeLength + outputTypeLength)) /
                                                256 * 256);
        } else if (is1ByteType(inputDtype) && !is1ByteType(outputDtype)) {
            tilingKey = 5;
            ubProcessNum = static_cast<int32_t>(
                availableUbSize / (BUFFER_NUM * (inputTypeLength + outputTypeLength) + sizeof(int16_t)) / 256 * 256);
        } else if (inputDtype == ge::DT_FLOAT16 && is1ByteType(outputDtype)) {
            tilingKey = 6;
            ubProcessNum = static_cast<int32_t>(
                availableUbSize / (BUFFER_NUM * (inputTypeLength + outputTypeLength) + sizeof(int32_t)) / 256 * 256);
        } else {
            tilingKey = 6;
            ubProcessNum = static_cast<int32_t>(availableUbSize / (BUFFER_NUM * (inputTypeLength + outputTypeLength)) /
                                                256 * 256);
        }
    }
};

static ge::graphStatus CastV3TilingFunc(gert::TilingContext* context)
{
    CastV3Tiling tilingObject(context);
    OP_CHECK_IF(tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context, "CastV3Tiling Init failed"),
                return ge::GRAPH_FAILED);
    return tilingObject.SetKernelTiling();
}

static ge::graphStatus TilingParseForCastV3([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CastV3).Tiling(CastV3TilingFunc).TilingParse<CastV3CompileInfo>(TilingParseForCastV3);
} // namespace optiling
