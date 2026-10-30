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
 * \file kl_div_v2_tiling_arch35.cpp
 * \brief
 */

#include <map>
#include <cmath>
#include "kl_div_v2_tiling_arch35.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "math/kl_div_v2/op_kernel/kl_div_v2_tiling_data.h"
#include "math/kl_div_v2/op_kernel/kl_div_v2_dag.h"
#include "math/kl_div_v2/op_kernel/kl_div_v2_tiling_key.h"

namespace optiling {
using namespace Ops::Base;

static const int32_t ATTR_INDEX_REDUCTION = 0;
static const int32_t ATTR_INDEX_LOGTARGET = 1;
static const std::map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"mean", 1}, {"sum", 2}, {"batchmean", 3}};
static const int32_t SIZE2 = 2;
static const int32_t SIZE4 = 4;
static const float INVALID_VAL = -0.1f;

ge::graphStatus CheckDtype(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "KLDivV2 CheckDtype enter.");
    auto inputXDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXDesc);
    auto inputTargetDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputTargetDesc);
    OP_CHECK_IF(inputXDesc->GetDataType() != inputTargetDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "input x and target",
                    (Ops::Base::ToString(inputXDesc->GetDataType()) + " and " +
                     Ops::Base::ToString(inputTargetDesc->GetDataType()))
                        .c_str(),
                    "The dtypes of input x and target must be the same");
                , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetEmptyValue(gert::TilingContext* context, KLDivV2TilingData* tilingData, const std::string reduction)
{
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    const gert::Shape& input_shape = EnsureNotScalar(x_shape->GetStorageShape());

    float emptyValue = INVALID_VAL;
    for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
        if (input_shape.GetDim(i) == 0) {
            if (reduction == "sum") {
                emptyValue = 0.0f;
            } else if (reduction == "mean") {
                emptyValue = std::nan("");
            } else if (reduction == "batchmean") {
                if (i == 0) {
                    emptyValue = std::nan("");
                } else {
                    emptyValue = 0.0f;
                }
            }
            break;
        }
    }
    tilingData->emptyValue = emptyValue;

    if (reduction == "batchmean" && input_shape.GetDim(0) != 0) {
        // 模板计算的均值是针对所有轴的，batchMean语义不一样
        tilingData->reduceTiling.meanVar = 1.0 / input_shape.GetDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTilingAscendC(
    gert::TilingContext* context, const KLDivV2CompileInfo* compileInfo, ReduceOpInputParam& opInput,
    ReduceTilingKey& key, ReduceOpTilingData* tilingData, const std::string reduction, const bool logTarget)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (logTarget == false && (reduction == "sum" || reduction == "none")) {
        if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagSumLogFalse<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagSumLogFalse<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        }
    } else if (logTarget == false && (reduction == "mean" || reduction == "batchmean")) {
        if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagMeanLogFalse<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagMeanLogFalse<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        }
    } else if (logTarget == true && (reduction == "sum" || reduction == "none")) {
        if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagSumLogTrue<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagSumLogTrue<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        }
    } else if (logTarget == true && (reduction == "mean" || reduction == "batchmean")) {
        if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagMeanLogTrue<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
            status = Tiling4ReduceOp<KLDivV2::KLDivDagMeanLogTrue<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, tilingData);
        }
    }

    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "x", Ops::Base::ToString(opInput.inputDtype).c_str(), "bfloat16, float16 or float"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus Tiling4KLDivV2AscendC(gert::TilingContext* context, const KLDivV2CompileInfo* compileInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const std::string reduction = std::string(attrs->GetStr(ATTR_INDEX_REDUCTION));
    const bool logTarget = *(attrs->GetBool(ATTR_INDEX_LOGTARGET));
    ReduceOpInputParam opInput;
    if (reduction == "none") {
        OP_CHECK_IF(
            (ReduceOpTmpl::GetInputParam(context, opInput, 0) == ge::GRAPH_FAILED),
            OP_LOGE(context->GetNodeName(), "[None] ReduceOp get x input param failed"), return ge::GRAPH_FAILED);
        opInput.axes.clear();
    } else {
        OP_CHECK_IF(
            (ReduceOpTmpl::GetInputParam(context, opInput, 0, 1, 0) == ge::GRAPH_FAILED),
            OP_LOGE(context->GetNodeName(), "ReduceOp get x input param failed"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(
        CheckDtype(context) == ge::GRAPH_FAILED, OP_LOGE(context, "check dtype failed"), return ge::GRAPH_FAILED);

    ReduceTilingKey key;
    auto tilingData = context->GetTilingData<KLDivV2TilingData>();
    OP_CHECK_IF(
        (DoTilingAscendC(context, compileInfo, opInput, key, &(tilingData->reduceTiling), reduction, logTarget) ==
         ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for KLDivV2"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetEmptyValue(context, tilingData, reduction) == ge::GRAPH_FAILED, OP_LOGE(context, "SetEmptyValue failed"),
        return ge::GRAPH_FAILED);

    auto it = STR_2_INT.find(reduction);
    OP_CHECK_IF(
        it == STR_2_INT.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "reduction", reduction.c_str(),
            "The value of reduction must be in (none, mean, sum, batchmean)"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key, it->second, static_cast<uint32_t>(logTarget));
    OP_LOGI(
        context->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, reduction is:%u, logTarget is %u Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, it->second, static_cast<uint32_t>(logTarget), tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus KLDivV2Tiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter KLDivV2Tiling");
    auto compile_info = reinterpret_cast<const KLDivV2CompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);

    return Tiling4KLDivV2AscendC(context, compile_info);
}

template <typename ContextT>
static ge::graphStatus TilingPrepare4ReduceOp(ContextT* context, ReduceOpCompileInfo* compileInfo)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->vectorCoreNum == 0UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "vectorCoreNum", std::to_string(compileInfo->vectorCoreNum).c_str(),
            "The value of vectorCoreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize <= CACHE_BUF_SIZE) {
        std::string reasonMsg = "The value of ubSize must be greater than " + std::to_string(CACHE_BUF_SIZE);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = GetCacheLineSize(context);
    if (compileInfo->cacheLineSize == 0UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "cacheLineSize", std::to_string(compileInfo->cacheLineSize).c_str(),
            "The value of cacheLineSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    compileInfo->ubBlockSize = GetUbBlockSize(context);
    if (compileInfo->ubBlockSize == 0UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "ubBlockSize", std::to_string(compileInfo->ubBlockSize).c_str(),
            "The value of ubBlockSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    compileInfo->vRegSize = GetVRegSize(context);
    if (compileInfo->vRegSize == 0UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "vRegSize", std::to_string(compileInfo->vRegSize).c_str(),
            "The value of vRegSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(
        context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
        compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
        compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static T* GetCompileInfoPtr(gert::TilingParseContext* context)
{
    return context->GetCompiledInfo<T>();
}

static ge::graphStatus KLDivV2Parse(gert::TilingParseContext* context)
{
    auto compile_info = GetCompileInfoPtr<KLDivV2CompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return TilingPrepare4ReduceOp(context, &compile_info->opInfo);
}

IMPL_OP_OPTILING(KLDivV2).Tiling(KLDivV2Tiling).TilingParse<KLDivV2CompileInfo>(KLDivV2Parse);
} // namespace optiling