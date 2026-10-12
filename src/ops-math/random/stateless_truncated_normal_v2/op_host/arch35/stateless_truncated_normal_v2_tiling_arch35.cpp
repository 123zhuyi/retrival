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
 * \file stateless_truncated_normal_v2_tiling_arch35.cpp
 * \brief
 */
#include "stateless_truncated_normal_v2_tiling_arch35.h"
#include <string>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_SHAPE = 0;
static constexpr uint16_t INPUT_IDX_KEY = 1;
static constexpr uint16_t INPUT_IDX_COUNTER = 2;
static constexpr uint16_t INPUT_IDX_ALG = 3;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t INDEX_0 = 0;
static constexpr int64_t DCACHE_SIZE = 128 * 1024;
static constexpr int64_t THREAD_DISPOSAL_NUM = 4;
static constexpr int64_t MAX_THREAD_NUM = 512;
static constexpr uint16_t CORE_ALIGN_SIZE = 512;

OpTilingConfig StatelessTruncatedNormalV2Tiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {{INPUT_IDX_SHAPE, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}},
                              {INPUT_IDX_KEY, {{ge::DT_UINT64}, 1, {1}, nullptr}},
                              {INPUT_IDX_COUNTER, {{ge::DT_UINT64}, 2, {1}, nullptr}},
                              {INPUT_IDX_ALG, {{ge::DT_INT32}, 1, {0, 1}, [](gert::TilingContext* ctx) {
                                                   const auto* algTensor = ctx->GetInputTensor(INPUT_IDX_ALG);
                                                   if (algTensor == nullptr) {
                                                       return false;
                                                   }
                                                   const int32_t* algData = algTensor->GetData<int32_t>();
                                                   if (algData == nullptr) {
                                                       return false;
                                                   }
                                                   constexpr int32_t ALG_PHILOX = 1;
                                                   if (algData[0] != ALG_PHILOX) {
                                                       std::string valueStr = std::to_string(algData[0]);
                                                       std::string reasonMsg = "Unsupported algorithm id: " + valueStr;
                                                       OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                                                           ctx->GetNodeName(), "input alg", valueStr.c_str(),
                                                           reasonMsg.c_str());
                                                       return false;
                                                   }
                                                   return true;
                                               }}}};
    config.outputCheckRules = {{OUTPUT_IDX_Y, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}}};
    config.attrCheckRules = {{INDEX_0, [](gert::TilingContext* ctx) {
                                  const auto* attrs = ctx->GetAttrs();
                                  const int64_t* attrPtr = attrs ? attrs->GetAttrPointer<int64_t>(INDEX_0) : nullptr;
                                  const auto* outDesc = ctx->GetOutputDesc(OUTPUT_IDX_Y);
                                  return attrPtr != nullptr && outDesc != nullptr &&
                                         static_cast<ge::DataType>(*attrPtr) == outDesc->GetDataType();
                              }}};
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        // 0-dim scalar output: the `shape` input is an empty tensor (ShapeSize==0).
        // GetData<>() on an empty tensor legally returns nullptr, which ExtractTensorValue
        // mistakes for an error. A scalar output has element count 1, so short-circuit here
        // before calling ExtractTensorValue.
        const auto* shapeTensor = ctx->GetInputTensor(INPUT_IDX_SHAPE);
        if (shapeTensor != nullptr && shapeTensor->GetShapeSize() == 0) {
            size = 1;
            return ge::GRAPH_SUCCESS;
        }
        return RandomUtils::GetAndCheckOutputSize<INPUT_IDX_SHAPE, OUTPUT_IDX_Y, false>(ctx, size);
    };
    // StatelessTruncatedNormalV2: key/counter are read directly from GM by kernel
    // Tiling fills seed=0, offset=0 as placeholders (same pattern as stateless_normal)
    // CalcExecutionPoliciesForBlocks uses offset=0 as base, so kernelOffset is pure relative increment
    config.getSeedAndOffset = [](gert::TilingContext* /*ctx*/, int64_t& seed, int64_t& offset) {
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

ge::graphStatus StatelessTruncatedNormalV2Tiling::DoSimtBlockTiling()
{
    OP_CHECK_IF((totalCoreNum_ <= 0), OP_LOGE(opName_, "totalCoreNum is less than or equal to 0. please check."),
                return ge::GRAPH_FAILED);
    int64_t threadNum = Ops::Base::CeilAlign(simtTilingData_.outputSize, THREAD_DISPOSAL_NUM);
    int64_t coreNum = Ops::Base::CeilAlign(threadNum, MAX_THREAD_NUM);
    simtTilingData_.usedCoreNum = std::min(coreNum, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessTruncatedNormalV2Tiling::UniqueProcess()
{
    // 0-dim scalar / empty-tensor fix (post-split, pre-serialize):
    // The output StorageShape for a 0-dim scalar is dirty (ndim=1, dim0=huge) because
    // IMPL_OP_INFERSHAPE sets gert OriginShape while tiling reads GE StorageShape (set
    // by canndev COMMON_INFER_FUNC_REG at graph compile time — two separate registries).
    // This dirty shape causes Is32bitIndexable() to fail (GetMaxOffsetBytes ~3.9e9 > 2.1e9),
    // CalcSplitBlocks wrongly splits the scalar into 2 blocks, and the splitBlockCount>1
    // pre-accumulation adds +4 to kernelOffset, producing wrong Philox output.
    //
    // UniqueProcess() runs AFTER FillUnifiedSimtTilingData (split) and BEFORE
    // WriteBackToContext (serialize), so we renormalize <=1-element outputs to a clean
    // single block here. This aligns with TF's num_elements()-driven semantics.
    // For genuine 1-element tensors (shape=[1]) the common path already yields identical
    // values, so this is a no-op; for numel>1 the branch is never entered.
    if (simtTilingData_.outputSize <= 1) {
        simtTilingData_.splitBlockCount = 1;
        simtTilingData_.splitBlocks[0].numel = simtTilingData_.outputSize;
        simtTilingData_.splitBlocks[0].gmOffset = 0;
        simtTilingData_.splitBlocks[0].kernelOffset = 0;
        simtTilingData_.splitBlocks[0].grid = 1;
        simtTilingData_.splitBlocks[0].totalThreads = SIMT_THREAD_GROUP_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4StatelessTruncatedNormalV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4StatelessTruncatedNormalV2 running tiling.");
    StatelessTruncatedNormalV2Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4StatelessTruncatedNormalV2(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "StatelessTruncatedNormalV2");
}

IMPL_OP_OPTILING(StatelessTruncatedNormalV2)
    .Tiling(Tiling4StatelessTruncatedNormalV2)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4StatelessTruncatedNormalV2)
    .TilingInputsDataDependency({INPUT_IDX_SHAPE, INPUT_IDX_KEY, INPUT_IDX_COUNTER, INPUT_IDX_ALG});
} // namespace optiling
