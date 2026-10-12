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
 * \file stateless_random_uniform_v3_tiling_arch35.cpp
 * \brief
 */

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include  "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "stateless_random_uniform_v3_tiling_arch35.h"

using namespace ge;

namespace optiling {

static constexpr uint16_t OUTPUT_IDX_Y = 0;
// v3KernelMode 是 OpDef 中第 2 个属性（dtype 为索引 0，v3KernelMode 为索引 1）
static constexpr uint32_t ATTR_IDX_V3_KERNEL_MODE = 1;
static constexpr uint16_t CORE_ALIGN_SIZE = 512;
static constexpr int32_t COEF_VAL_FP32 = 1;
static constexpr int32_t COEF_VAL_FP16_BF16 = 2;
static constexpr uint32_t BUFFER_NUM = 2;

// ========== 仅需配置校验规则+特定字段回调函数 ==========
OpTilingConfig StatelessRandomUniformV3Tiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        // 输入索引: dtype列表, shapeSize, dim_num
        {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},   // shape
        {1, {{ge::DT_UINT64}, 1, {}, nullptr}},                   // key
        {2, {{ge::DT_UINT64}, -1, {}, nullptr}},                  // counter
        {3, {{ge::DT_FLOAT}, 1, {}, nullptr}},                    // from
        {4, {{ge::DT_FLOAT}, 1, {}, nullptr}}                     // to
    };

    config.outputCheckRules = {
        {0, {{ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}, -1, {0,1,2,3,4,5,6,7,8}, nullptr}}  // y
    };

    // 获取 outputSize：输出 y 的 shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto outputShape = ctx->GetOutputShape(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputShape);
        shapeSize = outputShape->GetStorageShape().GetShapeSize();
        return ge::GRAPH_SUCCESS;
    };

    // key/counter 现在由 kernel 从 GM 直接读取，tiling 不再负责
    config.getKeyAndCounter = []([[maybe_unused]] gert::TilingContext* ctx, uint32_t key[2], uint32_t counter[4]) -> ge::graphStatus {
        key[0] = 0; key[1] = 0;
        counter[0] = 0; counter[1] = 0; counter[2] = 0; counter[3] = 0;
        return ge::GRAPH_SUCCESS;
    };

    // 获取 bufNum：每个输出元素所需的 UB 字节数
    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto outputDesc = ctx->GetOutputDesc(OUTPUT_IDX_Y);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outputDesc);
        auto outDtype = outputDesc->GetDataType();
        auto outputDtypeSize = ge::GetSizeByDataType(outDtype);
        auto coefVal = COEF_VAL_FP32;
        if (outDtype == ge::DataType::DT_FLOAT16 || outDtype == ge::DataType::DT_BF16) {
            coefVal = COEF_VAL_FP16_BF16;
        }
        bufNum = outputDtypeSize * (BUFFER_NUM + coefVal);
        return ge::GRAPH_SUCCESS;
    };

    config.coreAlignSize = CORE_ALIGN_SIZE;
    config.isNeedSyncAll = true;

    return config;
}

StatelessRandomUniformV3Tiling::StatelessRandomUniformV3Tiling(gert::TilingContext* ctx)
    : RandomTilingArch35(ctx, BuildOpConfig()) {}

ge::graphStatus StatelessRandomUniformV3Tiling::UniqueProcess()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const auto* scaleModeAttr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_V3_KERNEL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleModeAttr);
    tilingData_.v3KernelMode = static_cast<uint32_t>(*scaleModeAttr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4StatelessRandomUniformV3Tiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessRandomUniformV3Tiling");
}

static ge::graphStatus TilingStatelessRandomUniformV3(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingStatelessRandomUniformV3");
    StatelessRandomUniformV3Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(StatelessRandomUniformV3)
    .Tiling(TilingStatelessRandomUniformV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessRandomUniformV3Tiling);

} // namespace optiling