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
 * \file stateless_drop_out_gen_mask_tiling_arch35.cpp
 * \brief
 */
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "stateless_drop_out_gen_mask_tiling_arch35.h"
#include "log/log.h"
#include <string>
#include "register/op_impl_registry.h"

using namespace ge;

namespace optiling {

static constexpr int64_t IN_SHAPE_IDX = 0;
static constexpr int64_t IN_PROB_IDX = 1;
static constexpr int64_t IN_SEED_IDX = 2;
static constexpr int64_t IN_SEED1_IDX = 3;
static constexpr int64_t IN_OFFSET_IDX = 4;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr uint64_t BUFFER_NUM = 2;
static constexpr uint64_t EXIST_NODE_NUM = 3;
static constexpr uint64_t CORE_ALIGN_SIZE = 256;
static constexpr uint64_t UB_ALIGN_SIZE = 256;
static constexpr uint64_t REGBASE_CCEC_CACHE_SIZE = 8 * 1024;
static constexpr uint32_t RIGHT_SHIFT_NUM = 32;
static constexpr uint32_t INT32_SIZE = 4;
static constexpr uint32_t INT64_SIZE = 8;

// ========== 仅需配置校验规则+特定字段回调函数 ==========
OpTilingConfig StatelessDropOutGenMaskTiling::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
        // 输入索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},              // shape
        {1, {{ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16}, 1, {}, nullptr}}, // prob
        {2, {{ge::DT_INT32, ge::DT_INT64}, 1, {}, nullptr}},                // seed
        {3, {{ge::DT_INT32, ge::DT_INT64}, 1, {}, nullptr}},                // seed1
    };

    config.optionalInputCheckRules = {
        {4, {{ge::DT_INT64}, -1, {}, nullptr}}, // offset (可选输入)
    };

    config.outputCheckRules = {// 输出索引:  dtype列表，shapeSize，dim_num
                               {0, {{ge::DT_UINT8}, -1, {1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}}; // y

    // 获取output_size：输入0(shape)的shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        // 获取Input ShapeSize
        gert::Shape constShape;
        auto ret = ExtractTensorValue(ctx, 0, constShape);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(ctx->GetNodeName(), "GetOutputSize failed");
            return ret;
        }
        shapeSize = 1;
        uint32_t shapeRank = constShape.GetDimNum();
        for (uint32_t idx = 0; idx < shapeRank; idx++) {
            shapeSize *= static_cast<int64_t>(constShape.GetDim(idx));
        }
        if (shapeSize == 0) {
            std::string valueStr = std::to_string(shapeSize);
            std::string reasonMsg = "input shape should not be empty tensor";
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(ctx->GetNodeName(), "input shape", valueStr.c_str(),
                                                      reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    };

    // key/counter 由 kernel 从 GM 直接读取，tiling 侧不再做 D2H memcpy
    // 通过 key[] 传递元数据给 kernel：key[0]=offsetElemCount, key[1]=seedByteSize
    config.getKeyAndCounter = []([[maybe_unused]] gert::TilingContext* ctx, uint32_t key[2],
                                 uint32_t counter[4]) -> ge::graphStatus {
        counter[0] = 0;
        counter[1] = 0;
        counter[2] = 0;
        counter[3] = 0;

        // offset element count (shape info, no D2H needed)
        auto offsetTensor = ctx->GetOptionalInputTensor(IN_OFFSET_IDX);
        uint32_t offsetElemCount = 0;
        if (offsetTensor != nullptr) {
            offsetElemCount = static_cast<uint32_t>(offsetTensor->GetShapeSize());
        }
        key[0] = offsetElemCount;

        // seed byte size: 4 for INT32, 8 for INT64 (dtype info, no D2H needed)
        auto seedDesc = ctx->GetRequiredInputDesc(IN_SEED_IDX);
        uint32_t seedByteSize = 8;
        if (seedDesc != nullptr) {
            auto seedDtype = seedDesc->GetDataType();
            seedByteSize = (seedDtype == ge::DT_INT32) ? INT32_SIZE : INT64_SIZE;
        }
        key[1] = seedByteSize;

        return ge::GRAPH_SUCCESS;
    };

    config.ubAlignSize = UB_ALIGN_SIZE;

    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto outDesc = ctx->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outDesc);
        bufNum = sizeof(uint32_t) + sizeof(float) * BUFFER_NUM + sizeof(float) * BUFFER_NUM;
        return ge::GRAPH_SUCCESS;
    };

    config.coreAlignSize = CORE_ALIGN_SIZE;
    config.isNeedSyncAll = true;
    return config;
}

StatelessDropOutGenMaskTiling::StatelessDropOutGenMaskTiling(gert::TilingContext* context)
    : RandomTilingArch35(context, BuildOpConfig())
{}

static ge::graphStatus TilingPrepare4StatelessDropOutGenMaskTiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessDropOutGenMask");
}

static ge::graphStatus TilingStatelessDropOutGenMask(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingStatelessDropOutGenMask");
    StatelessDropOutGenMaskTiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(StatelessDropOutGenMask)
    .Tiling(TilingStatelessDropOutGenMask)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessDropOutGenMaskTiling)
    .TilingInputsDataDependency({IN_SHAPE_IDX, IN_PROB_IDX});
} // namespace optiling
