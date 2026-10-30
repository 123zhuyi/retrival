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
 * \file drop_out_do_mask_v3_tiling_arch35.cpp
 * \brief
 */

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include <string>
#include "op_common/op_host/util/platform_util.h"
#include "random/random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "drop_out_do_mask_v3_tiling_arch35.h"
#include "util/fp16.h"
#include "util/bfloat16.h"

using namespace ge;

namespace optiling {

constexpr int64_t ALIGN256 = 256;
constexpr int64_t MASK_ALIGN_RESERVE = 192;
constexpr int64_t KEEP_PROB_IDX = 2;
constexpr int64_t INDEX_INPUT_X = 0;

// ========== 仅需配置校验规则+特定字段回调函数 ==========
OpTilingConfig DropOutDoMaskV3Tiling::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
        // 输入索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}, // shape
        {1, {{ge::DT_UINT8, ge::DT_BOOL}, -1, {1}, nullptr}},                                      // mask
        {2, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, 1, {1}, nullptr}},                       // keep_prob
    };

    config.outputCheckRules = {
        // 输出索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}}; // y

    // 获取output_size：输入0(shape)的shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        auto xinputShapePtr = ctx->GetRequiredInputShape(INDEX_INPUT_X);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, xinputShapePtr);
        const auto& shape = Ops::Base::EnsureNotScalar(xinputShapePtr->GetStorageShape());
        shapeSize = 1;
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            shapeSize *= shape[i];
        }
        OP_LOGD(ctx->GetNodeName(), "DropOutDoMaskV3 Output shapeSize num is %ld.", shapeSize);
        return ge::GRAPH_SUCCESS;
    };

    // 获取key[2]：从attr1(seed) counter[4] attr(seed2)
    config.getKeyAndCounter = []([[maybe_unused]] gert::TilingContext* ctx, [[maybe_unused]] uint32_t key[2],
                                 [[maybe_unused]] uint32_t counter[4]) -> ge::graphStatus { return ge::GRAPH_SUCCESS; };

    config.coreAlignSize = ALIGN256;
    config.sharedTmpBufSize = MASK_ALIGN_RESERVE; // double mask buffer 128B 对齐预留

    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto inDesc = ctx->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, inDesc);
        auto inDtype = inDesc->GetDataType();
        auto inputDtypeSize = ge::GetSizeByDataType(inDtype);
        static constexpr uint32_t BUFFER_NUM = 2;
        bufNum = (inputDtypeSize * BUFFER_NUM + 1) * BUFFER_NUM;
        return ge::GRAPH_SUCCESS;
    };

    config.isNeedSyncAll = false;
    return config;
}

DropOutDoMaskV3Tiling::DropOutDoMaskV3Tiling(gert::TilingContext* ctx) : RandomTilingArch35(ctx, BuildOpConfig())
{}

ge::graphStatus DropOutDoMaskV3Tiling::UniqueProcess()
{
    auto keepProbTensor = context_->GetInputTensor(KEEP_PROB_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keepProbTensor);
    ge::DataType keepDtype = keepProbTensor->GetDataType();
    float keepProbNum = 0.0f;
    switch (keepDtype) {
        case DT_FLOAT: {
            const auto* const_data_ptr = keepProbTensor->GetData<float>();
            OP_CHECK_NULL_WITH_CONTEXT(context_, const_data_ptr);
            keepProbNum = *const_data_ptr;
            tilingData_.keepProb = keepProbNum;
            break;
        }
        case DT_FLOAT16: {
            const auto* const_data_ptr = keepProbTensor->GetData<Ops::Base::fp16_t>();
            OP_CHECK_NULL_WITH_CONTEXT(context_, const_data_ptr);
            keepProbNum = static_cast<float>(*const_data_ptr);
            tilingData_.keepProb = keepProbNum;
            break;
        }
        case DT_BF16: {
            const auto* const_data_ptr = keepProbTensor->GetData<Ops::Base::bfloat16>();
            OP_CHECK_NULL_WITH_CONTEXT(context_, const_data_ptr);
            keepProbNum = static_cast<float>(*const_data_ptr);
            tilingData_.keepProb = keepProbNum;
            break;
        }
        default: {
            std::string valueStr = Ops::Base::ToString(keepDtype);
            std::string reasonMsg = "Unsupported keep_prob data type";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "input keep_prob", valueStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (keepProbNum < 0 || keepProbNum > 1) {
        std::string valueStr = std::to_string(keepProbNum);
        std::string reasonMsg = "keepProbNum must be in range [0, 1]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "input keep_prob", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4DropOutDoMaskV3Tiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "DropOutDoMaskV3");
}

static ge::graphStatus TilingDropOutDoMaskV3(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingDropOutDoMaskV3");
    DropOutDoMaskV3Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(DropOutDoMaskV3)
    .Tiling(TilingDropOutDoMaskV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4DropOutDoMaskV3Tiling)
    .TilingInputsDataDependency({KEEP_PROB_IDX});

} // namespace optiling