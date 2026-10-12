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
 * \file stateless_sample_multinomial_tiling.cpp
 * \brief Tiling implementation for StatelessSampleMultinomial kernel
 */

#include "stateless_sample_multinomial_tiling.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_X = 0;
static constexpr uint16_t INPUT_IDX_NORM_PROBS = 1;
static constexpr uint16_t INPUT_IDX_SEED = 2;
static constexpr uint16_t INPUT_IDX_OFFSET = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t RANDOM_NUM_PER_COUNTER = 4;

OpTilingConfig StatelessSampleMultinomialTiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {{INPUT_IDX_X, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
                              {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
                              {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}}};
    config.optionalInputCheckRules = {
        {INPUT_IDX_NORM_PROBS, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}}};
    config.outputCheckRules = {{OUTPUT_IDX_Y, {{ge::DT_INT64}, -1, {}, nullptr}}};

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto xShape = ctx->GetInputShape(INPUT_IDX_X);
        OP_CHECK_IF(xShape == nullptr, OP_LOGE(ctx->GetNodeName(), "get x shape failed"), return ge::GRAPH_FAILED);
        int64_t numDist = (xShape->GetStorageShape().GetDimNum() == 2) ? xShape->GetStorageShape().GetDim(0) : 1;
        auto numsamplesPtr = ctx->GetAttrs()->GetInt(0); // attr index 0: "num_samples"
        OP_CHECK_IF(numsamplesPtr == nullptr, OP_LOGE(ctx->GetNodeName(), "get num_samples attr failed"),
                    return ge::GRAPH_FAILED);
        int64_t realSize = numDist * (*numsamplesPtr);
        size = (realSize + RANDOM_NUM_PER_COUNTER - 1) / RANDOM_NUM_PER_COUNTER;
        return ge::GRAPH_SUCCESS;
    };

    // seed/offset are device-computed (offset is an l0op::Add output) with no host value at
    // tiling time; reading them faults. Tiling stores 0; the kernel reads real values from GM.
    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/, int64_t& seed, int64_t& offset) {
        seed = 0;
        offset = 0;
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = true;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    return config;
}

ge::graphStatus StatelessSampleMultinomialTiling::CheckXRankAndNormProbsShape()
{
    auto xShapePtr = context_->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    const auto& xShape = xShapePtr->GetStorageShape();
    auto xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(xDimNum != 1 && xDimNum != 2,
                OP_LOGE(context_->GetNodeName(), "x must be 1D or 2D, but got %zuD", xDimNum), return ge::GRAPH_FAILED);

    auto normProbsShapePtr = context_->GetOptionalInputShape(INPUT_IDX_NORM_PROBS);
    if (normProbsShapePtr != nullptr) {
        const auto& normProbsShape = normProbsShapePtr->GetStorageShape();
        OP_CHECK_IF(normProbsShape != xShape,
                    OP_LOGE(context_->GetNodeName(), "the shapes of x and norm_probs must be the same"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessSampleMultinomialTiling::UniqueProcess()
{
    auto ret = CheckXRankAndNormProbsShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    auto xShape = context_->GetInputShape(INPUT_IDX_X);
    if (xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t numDist = (xShape->GetStorageShape().GetDimNum() == 2) ? xShape->GetStorageShape().GetDim(0) : 1;
    int64_t numCategories = xShape->GetStorageShape().GetDim(xShape->GetStorageShape().GetDimNum() - 1);

    auto numsamplesPtr = context_->GetAttrs()->GetInt(0);
    OP_CHECK_IF(numsamplesPtr == nullptr, OP_LOGE(context_->GetNodeName(), "get num_samples attr failed"),
                return ge::GRAPH_FAILED);
    simtTilingData_.from = *numsamplesPtr;                         // num_samples
    simtTilingData_.extraInt64Param1 = numDist * (*numsamplesPtr); // numDist * numsamples
    simtTilingData_.range = static_cast<uint64_t>(numCategories);  // numCategories
    simtTilingData_.splitBlockCount = (static_cast<uint64_t>(numDist) * static_cast<uint64_t>(numCategories) >
                                           UINT32_MAX ||
                                       static_cast<uint64_t>(simtTilingData_.extraInt64Param1) > UINT32_MAX) ?
                                          1 :
                                          0; // index type uint64 or uint32

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StatelessSampleMultinomial(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StatelessSampleMultinomial running tiling.");
    StatelessSampleMultinomialTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessSampleMultinomial(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessSampleMultinomial");
}

IMPL_OP_OPTILING(StatelessSampleMultinomial)
    .Tiling(Tiling4StatelessSampleMultinomial)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessSampleMultinomial)
    .TilingInputsDataDependency({INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
