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
 * \file drop_out_v3_tiling_arch35.cpp
 * \brief
 */

#include "drop_out_v3_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/fp16.h"
#include "util/bfloat16.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_X = 0;
static constexpr uint16_t INPUT_IDX_P = 2;
static constexpr uint16_t INPUT_IDX_SEED = 3;
static constexpr uint16_t INPUT_IDX_OFFSET = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t OUTPUT_IDX_MASK = 1;
static constexpr int64_t DCACHE_SIZE = 32768;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t ALIGNMENT_32 = 32;
static constexpr int64_t OFFSET_LIMIT = 4;
static constexpr int64_t MASK_ALIGN_SIZE = 128;
static constexpr int64_t UINT8_BIT_SIZE = 8;

OpTilingConfig DropOutV3Tiling::BuildOpConfig(gert::TilingContext* context)
{
    OpTilingConfig config;

    int64_t xSize = -1;
    int64_t maskSize = -1;
    if (context != nullptr) {
        auto inputShape = context->GetRequiredInputShape(INPUT_IDX_X);
        if (inputShape != nullptr) {
            auto storageShape = inputShape->GetStorageShape();
            xSize = storageShape.IsScalar() ? 1 : storageShape.GetShapeSize();
            maskSize = Ops::Base::CeilAlign(xSize, MASK_ALIGN_SIZE) / UINT8_BIT_SIZE;
        }
    }

    config.inputCheckRules = {
        {INPUT_IDX_X, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_P, {{ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT32, ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 2, {}, nullptr}}};
    config.outputCheckRules = {{OUTPUT_IDX_Y, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, xSize, {}, nullptr}},
                               {OUTPUT_IDX_MASK, {{ge::DT_UINT8}, maskSize, {}, nullptr}}};

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto inputShape = ctx->GetRequiredInputShape(INPUT_IDX_X);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, inputShape);
        auto storageShape = inputShape->GetStorageShape();
        size = storageShape.IsScalar() ? 1 : storageShape.GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        gert::Shape seedShape;
        auto ret = ExtractTensorValue(ctx, INPUT_IDX_SEED, seedShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "get seed value failed"),
                    return ge::GRAPH_FAILED);
        seed = static_cast<int64_t>(seedShape.GetDim(0));
        gert::Shape offsetShape;
        ret = ExtractTensorValue(ctx, INPUT_IDX_OFFSET, offsetShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(ctx->GetNodeName(), "get offset value failed"),
                    return ge::GRAPH_FAILED);
        offset = static_cast<int64_t>(offsetShape.GetDim(1));
        if (offset % OFFSET_LIMIT != 0) {
            std::string valueStr = std::to_string(offset);
            std::string reasonMsg = "The offset must be a multiple of 4";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "input offset", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = true;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    return config;
}

ge::graphStatus DropOutV3Tiling::UniqueProcess()
{
    simtTilingData_.ubSize = ubSize_;

    auto pTensor = context_->GetRequiredInputTensor(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pTensor);
    if (pTensor->GetShapeSize() <= 0) {
        std::string valueStr = std::to_string(pTensor->GetShapeSize());
        std::string reasonMsg = "shape size of prob tensor must be greater than 0";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "shape size of prob tensor",
                                                  valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto pDescPtr = context_->GetRequiredInputDesc(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pDescPtr);
    float prob = 0.0f;
    switch (pDescPtr->GetDataType()) {
        case ge::DT_DOUBLE: {
            prob = static_cast<float>(double(1) - pTensor->GetData<double>()[0]);
            break;
        }
        case ge::DT_FLOAT16: {
            auto srcP = pTensor->GetData<Ops::Base::fp16_t>()[0];
            prob = 1.0f - srcP.toFloat();
            break;
        }
        case ge::DT_BF16: {
            float srcP = pTensor->GetData<Ops::Base::bfloat16>()[0];
            prob = 1.0f - srcP;
            break;
        }
        case ge::DT_FLOAT: {
            prob = 1.0f - pTensor->GetData<float>()[0];
            break;
        }
        default: {
            std::string valueStr = Ops::Base::ToString(pDescPtr->GetDataType());
            std::string reasonMsg = "Unsupported p dtype";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input p", valueStr.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (prob < 0.0f || prob > 1.0f) {
        std::string valueStr = std::to_string(1.0f - prob);
        std::string reasonMsg = "The value of p has to be between 0 and 1";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input p", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    simtTilingData_.prob = prob;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = Ops::Base::CeilAlign(simtTilingData_.outputSize, ALIGNMENT_32) * sizeof(uint8_t) +
                     ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DropOutV3(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DropOutV3 running DropOutV3 tiling.");
    DropOutV3Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4DropOutV3(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "DropOutV3");
}

IMPL_OP_OPTILING(DropOutV3)
    .Tiling(Tiling4DropOutV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4DropOutV3)
    .TilingInputsDataDependency({INPUT_IDX_P, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
