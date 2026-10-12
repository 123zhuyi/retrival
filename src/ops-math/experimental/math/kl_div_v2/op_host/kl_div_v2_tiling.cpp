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
 * \file kl_div_v2_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/kl_div_v2_tiling_data.h"
#include "../op_kernel/kl_div_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_FACTOR = 8;
constexpr uint32_t UB_BLOCK_FACTOR_FP32 = 6;
constexpr uint32_t REDUCE_SINGLE_CORE_MAX = 8192;
constexpr size_t ATTR_REDUCTION_IDX = 0;
constexpr size_t ATTR_LOG_TARGET_IDX = 1;

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

static uint32_t ParseReduction(const char* reduction)
{
    if (reduction == nullptr) {
        return 2U;
    }
    if (reduction[0] == 'n') {
        return 0U;
    }
    if (reduction[0] == 's') {
        return 1U;
    }
    if (reduction[0] == 'b') {
        return 3U;
    }
    return 2U;
}

static ge::graphStatus KLDivV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto storageShape = inputX->GetStorageShape();
    int64_t totalNum = storageShape.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    const char* reductionStr = "mean";
    bool logTarget = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        if (attrs->GetAttrNum() > ATTR_REDUCTION_IDX) {
            const char* s = attrs->GetStr(ATTR_REDUCTION_IDX);
            if (s != nullptr) {
                reductionStr = s;
            }
        }
        if (attrs->GetAttrNum() > ATTR_LOG_TARGET_IDX) {
            const bool* b = attrs->GetAttrPointer<bool>(ATTR_LOG_TARGET_IDX);
            if (b != nullptr) {
                logTarget = *b;
            }
        }
    }
    uint32_t reductionMode = ParseReduction(reductionStr);

    KLDivV2TilingData* tiling = context->GetTilingData<KLDivV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(KLDivV2TilingData), 0, sizeof(KLDivV2TilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->reduction = (reductionMode == 0U) ? 0U : 1U;
    tiling->logTarget = logTarget ? 1U : 0U;

    float cof = 1.0f;
    if (reductionMode == 3U) {
        int64_t batch = (storageShape.GetDimNum() > 0) ? storageShape.GetDim(0) : 1;
        if (batch > 0) {
            cof = 1.0f / static_cast<float>(batch);
        }
    } else if (reductionMode == 2U) {
        if (totalNum > 0) {
            cof = 1.0f / static_cast<float>(totalNum);
        }
    }
    tiling->cof = cof;

    if (totalNum <= 0) {
        context->SetBlockDim(1);
        size_t* ws = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, ws);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        ws[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
        uint64_t tilingKey = (dataType == ge::DT_FLOAT16) ? KLDIVV2_TPL_SCH_MODE_0 :
                             (dataType == ge::DT_BF16)    ? KLDIVV2_TPL_SCH_MODE_2 :
                                                            KLDIVV2_TPL_SCH_MODE_1;
        context->SetTilingKey(GET_TPL_TILING_KEY(tilingKey));
        return ge::GRAPH_SUCCESS;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    uint32_t inputBytes = typeLength;

    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    uint32_t ubFactor = (dataType == ge::DT_FLOAT) ? UB_BLOCK_FACTOR_FP32 : UB_BLOCK_FACTOR;
    uint32_t tmp = static_cast<uint32_t>(ubSize / blockSize / BUFFER_NUM);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint32_t tb = tmp / ubFactor;
        tileBlockNum = (tb == 0) ? 1U : tb;
    }
    uint32_t tileDataNum = (tileBlockNum * blockSize) / inputBytes;
    if (tileDataNum == 0U) {
        tileDataNum = 1U;
    }

    OP_CHECK_IF(totalNum > static_cast<int64_t>(UINT32_MAX), OP_LOGE(context, "totalNum exceeds uint32 range"),
                return ge::GRAPH_FAILED);
    uint32_t total = static_cast<uint32_t>(totalNum);
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum);
    if (finalCoreNum > total) {
        finalCoreNum = total;
    }
    if (finalCoreNum == 0U) {
        finalCoreNum = 1U;
    }
    if (reductionMode != 0U && total <= REDUCE_SINGLE_CORE_MAX) {
        finalCoreNum = 1U;
    }

    uint32_t baseNum = total / finalCoreNum;
    uint32_t tailBlockNum = total % finalCoreNum;

    uint32_t smallCoreDataNum = baseNum;
    uint32_t smallTileNum = (smallCoreDataNum == 0) ? 0U : (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint32_t finalSmallTileNum = (smallTileNum == 0) ? 1U : smallTileNum;
    uint32_t smallTailDataNum = (smallTileNum == 0) ? 0U : smallCoreDataNum - tileDataNum * (smallTileNum - 1);
    if (smallTailDataNum == 0U) {
        smallTailDataNum = tileDataNum;
    }

    uint32_t bigCoreDataNum = baseNum + 1U;
    uint32_t bigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint32_t finalBigTileNum = (bigTileNum == 0) ? 1U : bigTileNum;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * (bigTileNum - 1);
    if (bigTailDataNum == 0U) {
        bigTailDataNum = tileDataNum;
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    context->SetBlockDim(finalCoreNum);

    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ws[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(finalCoreNum) * sizeof(float);

    uint64_t tilingKey = (dataType == ge::DT_FLOAT16) ? KLDIVV2_TPL_SCH_MODE_0 :
                         (dataType == ge::DT_BF16)    ? KLDIVV2_TPL_SCH_MODE_2 :
                                                        KLDIVV2_TPL_SCH_MODE_1;
    context->SetTilingKey(GET_TPL_TILING_KEY(tilingKey));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForKLDivV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct KLDivV2CompileInfo {};

IMPL_OP_OPTILING(KLDivV2).Tiling(KLDivV2TilingFunc).TilingParse<KLDivV2CompileInfo>(TilingParseForKLDivV2);

} // namespace optiling
