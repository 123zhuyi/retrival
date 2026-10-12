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
 * \file reduce_mean_with_count_tiling_arch35.cpp
 * \brief Tiling for ReduceMeanWithCount on ascend950 (arch35).
 *
 * Prototype: INPUT(x, count, count_sum) -> OUTPUT(y), ATTR(axes[required], keep_dims)
 * Computation: y = ReduceSum(x * count / count_sum, axes)
 * The DAG embeds Mul/Div pre-processing: CopyIn -> Cast -> Mul -> Div -> ReduceSumOp -> Cast -> CopyOut
 *
 * Follows the kl_div_v2 pattern: embed ReduceOpTilingData in a custom struct.
 */

#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_base_util.h"
#include "util/platform_util.h"
#include "math/reduce_mean_with_count/op_kernel/reduce_mean_with_count_tiling_data.h"
#include "math/reduce_mean_with_count/op_kernel/arch35/reduce_mean_with_count_dag.h"
#include "math/reduce_mean_with_count/op_kernel/arch35/reduce_mean_with_count_tiling_key.h"

using namespace Ops::Base;

namespace optiling {

// ---- Compile info (embed ReduceOpCompileInfo) ----
struct ReduceMeanWithCountCompileInfo {
    ReduceOpCompileInfo opInfo;
};

static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static constexpr size_t ATTR_INDEX_AXES = 0;
static constexpr size_t ATTR_INDEX_KEEP_DIMS = 1;

static ge::graphStatus DoTilingAscendC(
    gert::TilingContext* context, const ReduceMeanWithCountCompileInfo* compileInfo,
    ReduceOpInputParam& opInput, ReduceTilingKey& key,
    ReduceOpTilingData* reduceTiling)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<ReduceMeanWithCount::ReduceMeanWithCountDag<float, float>::OpDag>(
            context, opInput, key, &compileInfo->opInfo, reduceTiling);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<ReduceMeanWithCount::ReduceMeanWithCountDag<half, float>::OpDag>(
            context, opInput, key, &compileInfo->opInfo, reduceTiling);
    }

    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(),
                "ReduceOp Tiling failed, dtype should be in (bfloat16/float16/float)"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4ReduceMeanWithCount(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceMeanWithCountCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    // Get input shape and dtype from input 0 (x), no axes input tensor
    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(context, opInput, 0) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"), return ge::GRAPH_FAILED);

    // axes is a required attribute (not an input tensor)
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axesListPtr = attrs->GetListInt(ATTR_INDEX_AXES);
    if (axesListPtr != nullptr && axesListPtr->GetSize() > 0) {
        opInput.axes.resize(axesListPtr->GetSize());
        for (size_t i = 0; i < axesListPtr->GetSize(); i++) {
            int64_t ax = axesListPtr->GetData()[i];
            if (ax < 0) {
                ax += static_cast<int64_t>(opInput.shape.size());
            }
            opInput.axes[i] = ax;
        }
        std::sort(opInput.axes.begin(), opInput.axes.end());
    } else {
        // Empty axes -> reduce all dims
        opInput.axes.resize(opInput.shape.size());
        for (size_t i = 0; i < opInput.shape.size(); i++) {
            opInput.axes[i] = static_cast<int64_t>(i);
        }
    }

    // Get custom tiling data
    auto tilingData = context->GetTilingData<ReduceMeanWithCountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTilingAscendC(context, compileInfo, opInput, key, &(tilingData->reduceTiling)) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for ReduceMeanWithCount"),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key);
    OP_LOGI(
        context->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, isContiguous:%d, Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, key.isContiguous ? 1 : 0, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// ---- Tiling parse (get platform info at compile time) ----
template <typename ContextT>
static ge::graphStatus TilingPrepare4ReduceOp(ContextT* context, ReduceOpCompileInfo* compileInfo)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0UL),
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= CACHE_BUF_SIZE,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                ubSize, CACHE_BUF_SIZE),
        return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = GetCacheLineSize(context);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.",
                compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.",
                compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.", compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static T* GetCompileInfoPtr(gert::TilingParseContext* context)
{
    return context->GetCompiledInfo<T>();
}

static ge::graphStatus TilingParse4ReduceMeanWithCount(gert::TilingParseContext* context)
{
    auto compileInfo = GetCompileInfoPtr<ReduceMeanWithCountCompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return TilingPrepare4ReduceOp(context, &compileInfo->opInfo);
}

IMPL_OP_OPTILING(ReduceMeanWithCount)
    .Tiling(Tiling4ReduceMeanWithCount)
    .TilingParse<ReduceMeanWithCountCompileInfo>(TilingParse4ReduceMeanWithCount);

}  // namespace optiling
