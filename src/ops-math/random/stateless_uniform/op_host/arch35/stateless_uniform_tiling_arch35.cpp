/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_uniform_tiling_arch35.cpp
 * \brief
 */

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include  "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "stateless_uniform_tiling_arch35.h"

using namespace ge;

namespace optiling {

static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t INPUT_IDX_FROM = 3;
static constexpr uint16_t INPUT_IDX_TO = 4;
static constexpr uint16_t NUM_4 = 4;
static constexpr int64_t DCACHE_SIZE = 32 * 1024;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t OFFSET_LIMIT = 4;

OpTilingConfig StatelessUniformTiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        // shape=DT_INT64, seed=DT_INT64, offset=DT_INT64, from=DT_DOUBLE, to=DT_DOUBLE
        // 输入索引: dtype列表, shapeSize, dim_num
        {0, {{ge::DT_INT64}, -1, {1}, nullptr}},                  // shape
        {1, {{ge::DT_INT64}, 1, {}, nullptr}},                    // seed
        {2, {{ge::DT_INT64}, 1, {}, nullptr}},                    // offset
        {3, {{ge::DT_DOUBLE}, 1, {}, nullptr}},                   // from
        {4, {{ge::DT_DOUBLE}, 1, {}, nullptr}}                    // to
    };

    config.outputCheckRules = {
        {0, {{ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}, -1, {0, 1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}  // y: 0~8D
    };

    // 获取 outputSize：输出 y 的 shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
        shapeSize = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    // 填0占位
    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/,
                                 int64_t& seed, int64_t& offset) -> ge::graphStatus {
        seed = 0;
        offset = 0;
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    config.isNeedSyncAll = false;
    config.unrollFactor = NUM_4;
    config.enableSplitBlocks = true;

    return config;
}

StatelessUniformTiling::StatelessUniformTiling(gert::TilingContext* ctx)
    : RandomTilingArch35(ctx, BuildOpConfig()) {}

ge::graphStatus StatelessUniformTiling::UniqueProcess()
{
    simtTilingData_.ubSize = ubSize_;
    // 读取 from/to 的 double 值，转为 float 写入 tiling 结构体
    // （kernel 侧不支持 double 操作，由 tiling 层完成 double→float 转换）
    auto fromTensor = context_->GetInputTensor(INPUT_IDX_FROM);
    OP_CHECK_NULL_WITH_CONTEXT(context_, fromTensor);
    const auto* fromPtr = fromTensor->GetData<double>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, fromPtr);
    simtTilingData_.fromFp32 = static_cast<float>(*fromPtr);

    auto toTensor = context_->GetInputTensor(INPUT_IDX_TO);
    OP_CHECK_NULL_WITH_CONTEXT(context_, toTensor);
    const auto* toPtr = toTensor->GetData<double>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, toPtr);
    simtTilingData_.toFp32 = static_cast<float>(*toPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StatelessUniformTiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessUniformTiling");
}

static ge::graphStatus TilingStatelessUniform(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingStatelessUniform");
    StatelessUniformTiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(StatelessUniform)
    .Tiling(TilingStatelessUniform)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessUniformTiling)
    .TilingInputsDataDependency({INPUT_IDX_FROM, INPUT_IDX_TO});
} // namespace optiling
