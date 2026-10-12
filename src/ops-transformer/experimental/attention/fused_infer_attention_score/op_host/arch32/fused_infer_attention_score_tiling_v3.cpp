/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_v3.cpp
 * \brief
 */
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_info_parser.h"
#include "../../../common/op_host/arch32/fia_tiling_nonquant_mla.h"
#include "../../../common/op_host/arch32/fia_tiling_nonquant.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "../fused_infer_attention_score_const.h"
using namespace AscendC;
namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore, FusedInferAttentionScoreTilingData)

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}
bool CheckSpecConditions(const gert::TilingContext *context)
{
    constexpr int64_t BLOCKSIZE_ALIGN_16 = 16;
    constexpr int64_t MAX_BLOCKSIZE = 512;
    auto tempQ = context->GetInputShape(QUERY_INDEX);
    auto tempK = context->GetInputShape(KEY_INDEX);
    auto tempV = context->GetInputShape(VALUE_INDEX);
    auto kvDimNum = tempK->GetStorageShape().GetDimNum();
    auto qRope = context->GetOptionalInputTensor(QUERY_ROPE_INDEX);
    auto kRope = context->GetOptionalInputTensor(KEY_ROPE_INDEX);
    auto qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();

    auto attrs = context->GetAttrs();
    std::string inputLayoutStr = std::string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
    int32_t headNum = *(attrs->GetAttrPointer<int32_t>(ATTR_N_INDEX));
    int32_t kvHeadNum = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM_KV_HEADS_INDEX));
    int32_t innerPrecise = *(attrs->GetAttrPointer<int32_t>(ATTR_INNER_PRECISE_INDEX));
    
    bool isLayoutSupported = (inputLayoutStr == "TND") ? true : false;
    bool isRopeSplitMla = (qRope != nullptr) && (kRope != nullptr);
    
    bool isMha = (kvHeadNum == 0) || (headNum == kvHeadNum);
    bool mhaConditions = isMha  && (qDataType == ge::DT_FLOAT16) && (innerPrecise == 1);
    bool nonMhaConditions = !isMha && (innerPrecise == 0);
    bool specConditionFlag = false;

    if (isLayoutSupported && !isRopeSplitMla && (nonMhaConditions || mhaConditions)) {
        int64_t tempQD = tempQ->GetStorageShape().GetDim(DIM_2);
        
        int64_t tempKD = tempK->GetStorageShape().GetDim(DIM_2);
        int64_t tempVD = tempV->GetStorageShape().GetDim(DIM_2);
        bool isFAIDSize = (tempQD <= 256 && tempKD <= 256 && tempVD <= 256) &&
                (tempQD == tempKD && tempQD == tempVD);
        if (isFAIDSize) {
            specConditionFlag = true;
        }
        
    }
    return specConditionFlag;
}

bool RouteToFia(gert::TilingContext *context)
{
    if ((context == nullptr) || context->GetAttrs() == nullptr ||
        (context->GetInputDesc(QUERY_INDEX) == nullptr) ||
        (context->GetInputDesc(KEY_INDEX) == nullptr)) {
        return false;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if((aicNum != aivNum) && (aicNum * 2U != aivNum)) {
        OP_LOGI(context->GetNodeName(), "aicNum(%u):aivNum(%u) only support 1:1 or 1:2.", aicNum, aivNum);
        return false;
    }

    ge::DataType qDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    ge::DataType kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    bool isRopeSplit = (context->GetOptionalInputTensor(QUERY_ROPE_INDEX) != nullptr &&
        context->GetOptionalInputTensor(KEY_ROPE_INDEX) != nullptr);
    if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
        if (!isRopeSplit) {
            if (CheckSpecConditions(context)) {
                return false;
            }
        }
        const std::string inputLayoutStr = std::string(context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX));
        if (inputLayoutStr == "NSD") {
            return false;
        }
        return true;
    }
    return false;
}
} // namespace optiling
