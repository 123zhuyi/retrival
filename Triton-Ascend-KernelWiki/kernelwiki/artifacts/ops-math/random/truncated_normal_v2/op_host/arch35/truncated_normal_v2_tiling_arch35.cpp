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
 * \file truncated_normal_v2_tiling_arch35.cpp
 * \brief
 */
#include "truncated_normal_v2_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_OFFSET = 1;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t INDEX_0 = 0;
static constexpr uint16_t INDEX_1 = 1;
static constexpr uint16_t INDEX_2 = 2;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t THREAD_DISPOSAL_NUM = 4;
static constexpr int64_t MAX_THREAD_NUM = 512;
OpTilingConfig TruncatedNormalV2Tiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_SHAPE, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}}};
    config.outputCheckRules = {
        {OUTPUT_IDX_Y, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}}};
    config.attrCheckRules = {
        {INDEX_2, [](gert::TilingContext* ctx) {
            const auto* attrs = ctx->GetAttrs();
            const int64_t* attrPtr = attrs ? attrs->GetAttrPointer<int64_t>(INDEX_2) : nullptr;
            const auto* outDesc = ctx->GetOutputDesc(OUTPUT_IDX_Y);
            return attrPtr != nullptr && outDesc != nullptr &&
                   static_cast<ge::DataType>(*attrPtr) == outDesc->GetDataType();
        }}};
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        return RandomUtils::GetAndCheckOutputSize<INPUT_IDX_SHAPE, OUTPUT_IDX_Y, false>(ctx, size);
    };
    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        auto attrs = ctx->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);
        const auto* seedAttr = attrs->GetAttrPointer<int64_t>(INDEX_0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seedAttr);
        const auto* seed2Attr = attrs->GetAttrPointer<int64_t>(INDEX_1);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seed2Attr);
        seed = *seedAttr;
        offset = *seed2Attr; // 复用 offset 字段传递 seed2
        if (seed == 0 && offset == 0) {
            seed = static_cast<int64_t>(New64());
            offset = static_cast<int64_t>(New64());
        }
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = true;
    return config;
}

ge::graphStatus TruncatedNormalV2Tiling::DoSimtBlockTiling()
{
    OP_CHECK_IF(
        (totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is less than or equal to 0. please check."),
        return ge::GRAPH_FAILED);
    int64_t threadNum = Ops::Base::CeilAlign(simtTilingData_.outputSize, THREAD_DISPOSAL_NUM);
    int64_t coreNum = Ops::Base::CeilAlign(threadNum, MAX_THREAD_NUM);
    simtTilingData_.usedCoreNum = std::min(coreNum, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4TruncatedNormalV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4TruncatedNormalV2 running TruncatedNormalV2 tiling.");
    TruncatedNormalV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4TruncatedNormalV2(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "TruncatedNormalV2");
}

IMPL_OP_OPTILING(TruncatedNormalV2)
    .Tiling(Tiling4TruncatedNormalV2)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4TruncatedNormalV2)
    .TilingInputsDataDependency({0});
} // namespace optiling
