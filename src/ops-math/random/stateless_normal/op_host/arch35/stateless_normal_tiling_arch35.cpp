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
 * \file stateless_normal_tiling_arch35.cpp
 * \brief StatelessNormal V4 tiling implementation
 *        Uses base class enableSplitBlocks for automatic SplitUntil32Bit.
 *        seed/offset read via ValueDepend(OPTIONAL) + ExtractTensorValue.
 */

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include <string>
#include "op_common/op_host/util/platform_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "stateless_normal_tiling_arch35.h"

using namespace ge;

namespace optiling {

static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t INPUT_IDX_MEAN = 3;
static constexpr uint16_t INPUT_IDX_STDEV = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t CORE_ALIGN_SIZE = 512;
static constexpr int64_t OFFSET_LIMIT = 4;
static constexpr uint64_t DCACHE_SIZE = 32 * 1024;
static constexpr uint16_t NUM_4 = 4;

OpTilingConfig StatelessNormalTiling::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
        {0, {{ge::DT_INT64}, -1, {1}, nullptr}},                           // shape: 1D, INT64 only
        {1, {{ge::DT_INT64}, 1, {}, nullptr}},                             // seed: scalar
        {2, {{ge::DT_INT64}, -1, {}, nullptr}},                            // offset
        {3, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},  // mean
        {4, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},  // stdev
    };
    config.outputCheckRules = {
        {0, {{ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}, -1, {0, 1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}
    };

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
        shapeSize = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    // 填0占位，seed/offset 由 kernel 从 GM 直接读取
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

StatelessNormalTiling::StatelessNormalTiling(gert::TilingContext* ctx)
    : RandomTilingArch35(ctx, BuildOpConfig()) {}

ge::graphStatus StatelessNormalTiling::UniqueProcess()
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StatelessNormalTiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessNormalTiling");
}

static ge::graphStatus TilingStatelessNormal(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingStatelessNormal");
    StatelessNormalTiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(StatelessNormal)
    .Tiling(TilingStatelessNormal)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessNormalTiling)
    .TilingInputsDataDependency({INPUT_IDX_SEED, INPUT_IDX_OFFSET});

} // namespace optiling
