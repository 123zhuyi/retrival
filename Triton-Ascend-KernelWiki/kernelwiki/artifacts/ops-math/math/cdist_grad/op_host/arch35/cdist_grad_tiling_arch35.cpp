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
 * \file cdist_grad_tiling_arch35.cpp
 * \brief Tiling for CdistGrad on ascend950 (arch35).
 *
 * Prototype: INPUT(grad, x1, x2, cdist) -> OUTPUT(y), ATTR(p)
 * normMode is encoded into TilingKey as a compile-time parameter (not in tilingData).
 */

#include <cmath>
#include <vector>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "util/platform_util.h"
#include "math/cdist_grad/op_kernel/cdist_grad_tiling_data.h"
#include "math/cdist_grad/op_kernel/arch35/cdist_grad_dag.h"
#include "math/cdist_grad/op_kernel/arch35/cdist_grad_tiling_key.h"

using namespace Ops::Base;

namespace optiling {

struct CdistGradCompileInfo {
    ReduceOpCompileInfo opInfo;
};

static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;

static ge::graphStatus DoTilingAscendC(
    gert::TilingContext* context, const CdistGradCompileInfo* compileInfo,
    ReduceOpInputParam& opInput, ReduceTilingKey& key,
    ReduceOpTilingData* reduceTiling, int32_t normMode)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        // float32
        if (normMode == CdistGrad::NORM_MODE_INF) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradInfDag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_LARGE_P) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradLargePDag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P0) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP0Dag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P1) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP1Dag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P2) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP2Dag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else {
            status = Tiling4ReduceOp<CdistGrad::CdistGradDag<float, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        }
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        // float16 / bfloat16 — both use half for tiling calc, DTYPE_GRAD handles actual type
        if (normMode == CdistGrad::NORM_MODE_INF) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradInfDag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_LARGE_P) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradLargePDag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P0) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP0Dag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P1) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP1Dag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else if (normMode == CdistGrad::NORM_MODE_P2) {
            status = Tiling4ReduceOp<CdistGrad::CdistGradP2Dag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        } else {
            status = Tiling4ReduceOp<CdistGrad::CdistGradDag<half, float>::OpDag>(
                context, opInput, key, &compileInfo->opInfo, reduceTiling);
        }
    }

    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(),
                "ReduceOp Tiling failed, dtype should be in (float16/bfloat16/float)"),
        return ge::GRAPH_FAILED);
    return status;
}

static int32_t ComputeNormMode(float p)
{
    if (std::isinf(p) || p == -1.0f) {
        return CdistGrad::NORM_MODE_INF;
    } else if (p == 0.0f) {
        return CdistGrad::NORM_MODE_P0;
    } else if (p == 1.0f) {
        return CdistGrad::NORM_MODE_P1;
    } else if (p == 2.0f) {
        return CdistGrad::NORM_MODE_P2;
    } else if (p > 2.0f) {
        return CdistGrad::NORM_MODE_LARGE_P;
    } else {
        return CdistGrad::NORM_MODE_GENERAL;
    }
}

static ge::graphStatus Tiling4CdistGrad(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const CdistGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputParam(context, opInput, 0) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "ReduceOp get grad input param failed"), return ge::GRAPH_FAILED);

    int64_t dimNum = static_cast<int64_t>(opInput.shape.size());
    OP_CHECK_IF(
        dimNum < 2,
        OP_LOGE(context->GetNodeName(), "CdistGrad requires at least 2D input, got: %ld", dimNum),
        return ge::GRAPH_FAILED);
    opInput.axes = {dimNum - 2};

    // Get attr p
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    float p = 2.0f;
    if (attrs->GetAttrNum() > 0) {
        const float* pAttr = attrs->GetAttrPointer<float>(0);
        p = (pAttr == nullptr) ? 2.0f : *pAttr;
    }

    // Compute normMode from p — only used for TilingKey and DAG selection, not stored in tilingData
    int32_t normMode = ComputeNormMode(p);

    auto tilingData = context->GetTilingData<CdistGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);

    tilingData->powCdist = p - 1.0f;
    tilingData->powDiff = p - 2.0f;

    OP_LOGI(context->GetNodeName(), "CdistGrad attr p = %f, normMode = %d", p, normMode);

    ReduceTilingKey key;
    OP_CHECK_IF(
        (DoTilingAscendC(context, compileInfo, opInput, key,
                         &(tilingData->reduceTiling), normMode) == ge::GRAPH_FAILED),
        OP_LOGE(context->GetNodeName(), "DoTiling Failed for CdistGrad"),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey;
    GEN_REDUCE_TILING_KEY(tilingKey, key, static_cast<uint32_t>(normMode));
    OP_LOGI(
        context->GetNodeName(),
        "patternID:%u, loopARCount:%u, loopInnerARCount:%u, normMode:%d, Tiling Key is:%lu",
        key.patternID, key.loopARCount, key.loopInnerARCount, normMode, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

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

static ge::graphStatus TilingParse4CdistGrad(gert::TilingParseContext* context)
{
    auto compileInfo = GetCompileInfoPtr<CdistGradCompileInfo>(context);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return TilingPrepare4ReduceOp(context, &compileInfo->opInfo);
}

IMPL_OP_OPTILING(CdistGrad)
    .Tiling(Tiling4CdistGrad)
    .TilingParse<CdistGradCompileInfo>(TilingParse4CdistGrad);

}  // namespace optiling
