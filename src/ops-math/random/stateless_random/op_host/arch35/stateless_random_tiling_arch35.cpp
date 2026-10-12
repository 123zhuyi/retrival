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
 * \file stateless_random_tiling_arch35.cpp
 * \brief Tiling implementation for stateless_random operator on arch35
 */

#include "stateless_random_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t INPUT_IDX_FROM = 3;
static constexpr uint16_t INPUT_IDX_TO = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_IDX_MAX_RANGE_MODE = 1;
static constexpr int64_t DCACHE_SIZE = 32 * 1024;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t OFFSET_LIMIT = 4;
static constexpr int64_t MAX_NDIM = 8;
static constexpr int64_t MAX_RANGE = static_cast<int64_t>(UINT32_MAX);
static constexpr int64_t RAND_INT64_THRESHOLD = 268435456LL;
static constexpr int32_t FLOAT16_DIGITS = 11; // Number of digits for DT_FLOAT16
static constexpr int32_t BF16_DIGITS = 8;     // Number of digits for DT_BF16
static constexpr uint32_t UNROLL_2 = 2;
static constexpr uint32_t UNROLL_4 = 4;

static void GetMinAndMaxByDtype(ge::DataType dtype, int64_t& dtypeMin, int64_t& dtypeMax)
{
    switch (dtype) {
        case ge::DT_INT8:
            dtypeMin = static_cast<int64_t>(std::numeric_limits<int8_t>::min());
            dtypeMax = static_cast<int64_t>(std::numeric_limits<int8_t>::max());
            break;
        case ge::DT_UINT8:
            dtypeMin = static_cast<int64_t>(std::numeric_limits<uint8_t>::min());
            dtypeMax = static_cast<int64_t>(std::numeric_limits<uint8_t>::max());
            break;
        case ge::DT_INT16:
            dtypeMin = static_cast<int64_t>(std::numeric_limits<int16_t>::min());
            dtypeMax = static_cast<int64_t>(std::numeric_limits<int16_t>::max());
            break;
        case ge::DT_INT32:
            dtypeMin = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
            dtypeMax = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
            break;
        case ge::DT_INT64:
            dtypeMin = static_cast<int64_t>(std::numeric_limits<int64_t>::min());
            dtypeMax = static_cast<int64_t>(std::numeric_limits<int64_t>::max());
            break;
        case ge::DT_FLOAT:
            dtypeMin = -(1L << std::numeric_limits<float>::digits);
            dtypeMax = (1L << std::numeric_limits<float>::digits);
            break;
        case ge::DT_FLOAT16:
            dtypeMin = -(1L << FLOAT16_DIGITS);
            dtypeMax = (1L << FLOAT16_DIGITS);
            break;
        case ge::DT_BF16:
            dtypeMin = -(1L << BF16_DIGITS);
            dtypeMax = (1L << BF16_DIGITS);
            break;
        case ge::DT_BOOL:
            dtypeMin = 0;
            dtypeMax = 1;
            break;
        default:
            dtypeMin = 0;
            dtypeMax = 0;
            break;
    }
}

OpTilingConfig StatelessRandomTiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_SHAPE, {{ge::DT_INT64}, -1, {1}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}}};
    config.outputCheckRules = {
        {OUTPUT_IDX_Y,
         {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
           ge::DT_UINT8, ge::DT_BOOL},
          -1,
          {0, 1, 2, 3, 4, 5, 6, 7, 8},
          nullptr}}};

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
        shapeSize = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    // 当seed/offset为tensor时，tiling侧获取不到值，kernel从GM直接读取
    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/, int64_t& seed, int64_t& offset) -> ge::graphStatus {
        seed = 0;
        offset = 0;
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    config.enableSplitBlocks = true;
    return config;
}

ge::graphStatus StatelessRandomTiling::BeforeProcess()
{
    auto fromTensor = context_->GetOptionalInputTensor(INPUT_IDX_FROM);
    auto toTensor = context_->GetOptionalInputTensor(INPUT_IDX_TO);
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();

    if ((fromTensor == nullptr) && (toTensor == nullptr)) {
        config_.unrollFactor = (outputDtype == ge::DT_INT64) ? UNROLL_2 : UNROLL_4;
    } else if (fromTensor == nullptr) {
        auto toData = toTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, toData);
        config_.unrollFactor = (toData[0] >= RAND_INT64_THRESHOLD) ? UNROLL_2 : UNROLL_4;
    } else if (toTensor == nullptr) {
        int64_t dtypeMin = 0;
        int64_t dtypeMax = 0;
        GetMinAndMaxByDtype(outputDtype, dtypeMin, dtypeMax);
        auto fromData = fromTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, fromData);
        config_.unrollFactor =
            ((static_cast<uint64_t>(dtypeMax) + 1 - static_cast<uint64_t>(fromData[0])) >= RAND_INT64_THRESHOLD) ?
                UNROLL_2 : UNROLL_4;
    } else {
        auto fromData = fromTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, fromData);
        auto toData = toTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, toData);
        config_.unrollFactor =
            ((static_cast<uint64_t>(toData[0]) - static_cast<uint64_t>(fromData[0])) >= RAND_INT64_THRESHOLD) ?
                UNROLL_2 : UNROLL_4;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandomTiling::UniqueProcess()
{
    simtTilingData_.ubSize = ubSize_;

    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();

    int64_t dtypeMin = 0;
    int64_t dtypeMax = 0;
    int64_t from = 0;
    uint64_t range = 0;
    auto fromTensor = context_->GetOptionalInputTensor(INPUT_IDX_FROM);
    auto toTensor = context_->GetOptionalInputTensor(INPUT_IDX_TO);
    if ((fromTensor == nullptr) && (toTensor == nullptr)) {
        GetMinAndMaxByDtype(outputDtype, dtypeMin, dtypeMax);
        from = 0;
        range = static_cast<uint64_t>(dtypeMax) + 1;
    } else if (fromTensor == nullptr) {
        auto toData = toTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, toData);
        from = 0;
        range = toData[0];
    } else if (toTensor == nullptr) {
        GetMinAndMaxByDtype(outputDtype, dtypeMin, dtypeMax);
        auto fromData = fromTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, fromData);
        from = fromData[0];
        range = static_cast<uint64_t>(dtypeMax) + 1 - static_cast<uint64_t>(fromData[0]);
    } else {
        auto fromData = fromTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, fromData);
        auto toData = toTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, toData);
        from = fromData[0];
        range = static_cast<uint64_t>(toData[0]) - static_cast<uint64_t>(fromData[0]);
    }

    simtTilingData_.from = from;
    simtTilingData_.range = range;
    simtTilingData_.extraInt64Param1 = config_.unrollFactor;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StatelessRandom(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StatelessRandom running stateless_random tiling.");
    StatelessRandomTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessRandom(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessRandom");
}

IMPL_OP_OPTILING(StatelessRandom)
    .Tiling(Tiling4StatelessRandom)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessRandom)
    .TilingInputsDataDependency({INPUT_IDX_FROM, INPUT_IDX_TO});

} // namespace optiling