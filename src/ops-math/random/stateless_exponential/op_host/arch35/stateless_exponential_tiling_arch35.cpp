/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_exponential_tiling_arch35.cpp
 * \brief Tiling implementation for StatelessExponential operator (ascend950 / SIMT).
 *
 * Hybrid of two references:
 *  - sim_thread_exponential_tiling_arch35.cpp : exponential-specific config (lambd>0 check,
 *    enableSplitBlocks, unrollFactor, prob=lambd, TilingKey routed by output dtype);
 *  - stateless_normal_tiling_arch35.cpp        : seed/offset are tensor inputs that must NOT
 *    be read at tiling time (an offset that is a device-computed intermediate has no host
 *    value here). Tiling stores 0 placeholders; the kernel reads real seed/offset from GM.
 */
#include "stateless_exponential_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {
static constexpr uint16_t INPUT_IDX_SELF = 0;
static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t OUTPUT_IDX_SELF = 0;

static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr uint32_t NUM_FOUR = 4;

static constexpr uint64_t TILING_KEY_FP16 = 1;
static constexpr uint64_t TILING_KEY_BF16 = 2;
static constexpr uint64_t TILING_KEY_FP32 = 3;

OpTilingConfig StatelessExponentialTilingSimt::BuildOpConfig()
{
    OpTilingConfig config;

    // self: FP16/BF16/FP32, any shape; seed/offset: INT64 scalar (shapeSize == 1).
    config.inputCheckRules = {{INPUT_IDX_SELF, {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}, -1, {}, nullptr}},
                              {INPUT_IDX_SEED, {{ge::DT_INT64}, 1, {}, nullptr}},
                              {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 1, {}, nullptr}}};
    config.outputCheckRules = {{OUTPUT_IDX_SELF, {{ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}, -1, {}, nullptr}}};

    // Output size == total elements of self (lambd is the only attr; count is derived).
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto selfShape = ctx->GetInputShape(INPUT_IDX_SELF);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, selfShape);
        size = selfShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    // Avoid reading device-computed seed/offset during GetWorkspaceSize to prevent
    // host-side page faults. Store 0 placeholders; kernel reads real values from GM at runtime.
    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/, int64_t& seed, int64_t& offset) {
        seed = 0;
        offset = 0;
        return ge::GRAPH_SUCCESS;
    };

    // lambd is attr index 0 (the only attribute) and must be > 0.
    config.attrCheckRules = {
        {0,
         [](gert::TilingContext* ctx) -> bool {
             auto attrs = ctx->GetAttrs();
             if (attrs == nullptr) {
                 return false;
             }
             const auto* lambdAttr = attrs->GetAttrPointer<float>(0);
             if (lambdAttr == nullptr) {
                 return false;
             }
             if (*lambdAttr <= 0.0f) {
                 std::string valueStr = std::to_string(*lambdAttr);
                 std::string reasonMsg = "lambd must be greater than 0";
                 OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "attr lambd", valueStr.c_str(),
                                                       reasonMsg.c_str());
                 return false;
             }
             return true;
         }},
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = false;
    config.unrollFactor = NUM_FOUR;
    config.enableSplitBlocks = true;
    return config;
}

ge::graphStatus StatelessExponentialTilingSimt::UniqueProcess()
{
    // Store lambd into the generic `prob` field; the kernel reads tilingData->prob as lambda.
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto* lambdAttr = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, lambdAttr);
    simtTilingData_.prob = *lambdAttr;

    // Route TilingKey by output dtype: 1=FP16, 2=BF16, 3=FP32 (matches kernel dispatch).
    auto outputDesc = context_->GetOutputDesc(OUTPUT_IDX_SELF);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    auto outputDtype = outputDesc->GetDataType();
    if (outputDtype == ge::DT_FLOAT16) {
        tilingKey_ = TILING_KEY_FP16;
    } else if (outputDtype == ge::DT_BF16) {
        tilingKey_ = TILING_KEY_BF16;
    } else {
        tilingKey_ = TILING_KEY_FP32;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace optiling {

static ge::graphStatus Tiling4StatelessExponentialArch35(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StatelessExponential running tiling.");
    StatelessExponentialTilingSimt tilingObj(context);
    return tilingObj.DoTiling();
}

ge::graphStatus TilingPrepare4StatelessExponentialArch35(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessExponential");
}

IMPL_OP_OPTILING(StatelessExponential)
    .Tiling(Tiling4StatelessExponentialArch35)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessExponentialArch35)
    .TilingInputsDataDependency({INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
