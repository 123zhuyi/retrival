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
* \file stateless_random_normal_v3_tiling_arch35.cpp
* \brief
*/

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include  "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "stateless_random_normal_v3_tiling_arch35.h"

using namespace ge;

namespace optiling {

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_KEY = 1;
static constexpr uint16_t INPUT_IDX_COUNTER = 2;
static constexpr uint16_t INPUT_IDX_MEAN = 3;
static constexpr uint16_t INPUT_IDX_STDEV = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t CORE_ALIGN_SIZE = 512;
static constexpr uint32_t MEAN_SCALAR_FLAG = 1;
static constexpr uint32_t STDEV_SCALAR_FLAG = 2;
static constexpr uint32_t BASE_SPLIT_UB_NUM = 5;
// ========== 仅需配置校验规则+特定字段回调函数 ==========
OpTilingConfig StatelessRandomNormalV3Tiling::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
            // 输入索引:  dtype列表，shapeSize，dim_num
            {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},  // shape
            {1, {{ge::DT_UINT64}, 1, {}, nullptr}}, // key
            {2, {{ge::DT_UINT64}, -1, {}, nullptr}}, // counter
            {3, {{ge::DT_FLOAT}, -1, {}, nullptr}},   // mean (scalar or tensor)
            {4, {{ge::DT_FLOAT}, -1, {}, nullptr}},  // stdev (scalar or tensor)
        };
    config.outputCheckRules = {
        {0, {{ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}, -1, {0,1,2,3,4,5,6,7,8}, nullptr}}
    };  // y
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
        shapeSize = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    config.getKeyAndCounter = []([[maybe_unused]] gert::TilingContext* ctx, uint32_t key[2], uint32_t counter[4]) -> ge::graphStatus {
        key[0] = 0; key[1] = 0;
        counter[0] = 0; counter[1] = 0; counter[2] = 0; counter[3] = 0;
        return ge::GRAPH_SUCCESS;
    };

    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto outputDesc = ctx->GetOutputDesc(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputDesc);
        uint32_t extraBufNum = 0;
        auto meanTensor = ctx->GetInputTensor(INPUT_IDX_MEAN);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, meanTensor);
        if (meanTensor->GetShapeSize() > 1) {
            extraBufNum++;  
        }
        auto stdevTensor = ctx->GetInputTensor(INPUT_IDX_STDEV);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, stdevTensor);
        if (stdevTensor->GetShapeSize() > 1) {
            extraBufNum++;  
        }

        bufNum = sizeof(float) * (BASE_SPLIT_UB_NUM + extraBufNum);
        return ge::GRAPH_SUCCESS;
    };

    config.coreAlignSize = CORE_ALIGN_SIZE;
    config.isNeedSyncAll = false;
    return config;
}

StatelessRandomNormalV3Tiling::StatelessRandomNormalV3Tiling(gert::TilingContext* ctx) : RandomTilingArch35(ctx, BuildOpConfig()){}

ge::graphStatus StatelessRandomNormalV3Tiling::UniqueProcess()
{
    uint32_t v3KernelMode = 0;

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    int64_t outputSize = outputShape->GetStorageShape().GetShapeSize();

    auto meanTensor = context_->GetInputTensor(INPUT_IDX_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanTensor);
    int64_t meanSize = meanTensor->GetShapeSize();
    if (meanSize == 1) {
        v3KernelMode |= MEAN_SCALAR_FLAG;
    } else if (meanSize != outputSize) {
        std::string valueStr = std::to_string(meanSize) + " and " + std::to_string(outputSize);
        std::string reasonMsg = "StatelessRandomNormalV3 does not support broadcast for mean, mean shapeSize must equal output shapeSize or be 1";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "mean shapeSize and output shapeSize",
            valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto stdevTensor = context_->GetInputTensor(INPUT_IDX_STDEV);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stdevTensor);
    int64_t stdevSize = stdevTensor->GetShapeSize();
    if (stdevSize == 1) {
        v3KernelMode |= STDEV_SCALAR_FLAG;
    } else if (stdevSize != outputSize) {
        std::string valueStr = std::to_string(stdevSize) + " and " + std::to_string(outputSize);
        std::string reasonMsg = "StatelessRandomNormalV3 does not support broadcast for stdev, stdev shapeSize must equal output shapeSize or be 1";
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context_->GetNodeName(), "stdev shapeSize and output shapeSize",
            valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    tilingData_.v3KernelMode = v3KernelMode;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StatelessRandomNormalV3Tiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessRandomNormalV3Tiling");
}

static ge::graphStatus TilingStatelessRandomNormalV3(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingStatelessRandomNormalV3");
    StatelessRandomNormalV3Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(StatelessRandomNormalV3)
    .Tiling(TilingStatelessRandomNormalV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessRandomNormalV3Tiling);

} // namespace optiling