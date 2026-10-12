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
 * \file stateless_bernoulli_tiling_arch35.cpp
 * \brief
 */
#include "stateless_bernoulli_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_PROB = 1;
static constexpr uint16_t INPUT_IDX_SEED = 2;
static constexpr uint16_t INPUT_IDX_OFFSET = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t CORE_MINIEST_NUM = 256;
static constexpr int64_t OFFSET_MULTIPLE = 4;

OpTilingConfig StatelessBernoulliTiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_SHAPE, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},
        {INPUT_IDX_PROB, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}},
    };
    config.outputCheckRules = {
        {OUTPUT_IDX_Y, {{ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16, ge::DT_UINT16,
        ge::DT_INT32, ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64, ge::DT_FLOAT,
        ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BOOL}, -1, {}, nullptr}},
    };

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        return RandomUtils::GetAndCheckOutputSize<INPUT_IDX_SHAPE, OUTPUT_IDX_Y, false>(ctx, size);
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        auto seedTensor = ctx->GetRequiredInputTensor(INPUT_IDX_SEED);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seedTensor);
        const int64_t* seedVal = seedTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, seedVal);
        seed = *seedVal;
        auto offsetTensor = ctx->GetRequiredInputTensor(INPUT_IDX_OFFSET);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, offsetTensor);
        const int64_t* offsetVal = offsetTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, offsetVal);
        offset = *offsetVal;
        if (offset % OFFSET_MULTIPLE != 0) {
            std::string valueStr = std::to_string(offset);
            std::string reasonMsg = "offset value must be a multiple of 4";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "input offset", valueStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    return config;
}

ge::graphStatus StatelessBernoulliTiling::DoSimtBlockTiling()
{
    OP_CHECK_IF((totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is less than or equal to 0. please check."),
        return ge::GRAPH_FAILED);

    auto probTensor = context_->GetRequiredInputTensor(INPUT_IDX_PROB);
    OP_CHECK_NULL_WITH_CONTEXT(context_, probTensor);
    int64_t probTensorSize = static_cast<int64_t>(probTensor->GetShapeSize());
    simtTilingData_.extraInt64Param1 = (probTensorSize == 1);
    // special branch
    int64_t effectiveSize = simtTilingData_.outputSize;
    if (simtTilingData_.extraInt64Param1 == 0 && effectiveSize > probTensorSize) {
        effectiveSize = probTensorSize;
    }
    int64_t avgPerCore = Ops::Base::CeilDiv(effectiveSize, totalCoreNum_);
    int64_t numOfPerCore = Ops::Base::CeilAlign(avgPerCore, CORE_MINIEST_NUM);
    int64_t usedCoreNum = Ops::Base::CeilDiv(effectiveSize, numOfPerCore);
    simtTilingData_.usedCoreNum = std::min(usedCoreNum, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StatelessBernoulli(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StatelessBernoulli running.");
    StatelessBernoulliTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessBernoulli(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessBernoulli");
}

IMPL_OP_OPTILING(StatelessBernoulli)
    .Tiling(Tiling4StatelessBernoulli)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessBernoulli)
    .TilingInputsDataDependency({INPUT_IDX_SHAPE, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
