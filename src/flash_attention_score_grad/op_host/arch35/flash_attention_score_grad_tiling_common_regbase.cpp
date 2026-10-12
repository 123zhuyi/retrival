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
 * \file flash_attention_score_grad_tiling_common.cpp
 * \brief
 */

#include "flash_attention_score_grad_tiling_common_regbase.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace fag {

ge::graphStatus CheckSoftmaxMaxShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, bool isQuant)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 4) { // softmaxMax only support 4 dimensions
        OP_LOGE(context, "The shape of softmaxMax is invalid, got %lu dimensions", softmaxMaxShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxShape->GetStorageShape().GetDim(3); // 3:8
    int64_t validDim3 = isQuant ? 1 : 8;
    OP_CHECK_IF((dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != validDim3),
              OP_LOGE(context, "The shape of softmaxMax is invalid, got (%ld,%ld,%ld,%ld), should be (%ld,%ld,%ld,%ld)",
                        dim0, dim1, dim2, dim3, b, n1, s1, validDim3),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxSumShape(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, bool isQuant)
{
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 4) { // softmaxSum only support 4 dimensions
        OP_LOGE(context, "The shape of softmaxSum is invalid, got %lu dimensions", softmaxSumShapeDim);
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxSumShape->GetStorageShape().GetDim(3); // 3:8
    int64_t validDim3 = isQuant ? 1 : 8;
    OP_CHECK_IF((dim0 != b || dim1 != n1 || dim2 != s1 || dim3 != validDim3),
              OP_LOGE(context, "The shape of softmaxSum is invalid, got (%ld,%ld,%ld,%ld), should be (%ld,%ld,%ld,%ld)",
              dim0, dim1, dim2, dim3, b, n1, s1, validDim3),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(gert::TilingContext *context)
{
    auto attentionInShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTENTION_IN));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE(context, "The dimnum of attentionIn %zu should be equal to query %zu", attentionInShapeDim,
                  queryShapeDim);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckShapeValid(gert::TilingContext *context, int64_t b, int64_t n1, int64_t s1, int64_t d)
{
    auto isShapeInValid = (b == 0 || n1 == 0 || s1 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
              OP_LOGE(context, "input shape error, got 0 in bnsd(%ld,%ld,%ld,%ld)", b, n1, s1, d),
              return ge::GRAPH_FAILED);

    auto queryType = context->GetInputDesc(0)->GetDataType();
    bool isQuant = queryType == ge::DT_HIFLOAT8;
    auto ret = CheckSoftmaxMaxShape(context, b, n1, s1, isQuant);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxSumShape(context, b, n1, s1, isQuant);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext *context, int64_t t1, int64_t n1, int64_t d)
{
    if (context == nullptr) {
        OP_LOGE(context, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0);
    OP_CHECK_IF(isShapeInValid,
              OP_LOGE(context, "input shape error, got 0 in tnd(%ld,%ld,%ld)", t1, n1, d),
              return ge::GRAPH_FAILED);

    auto ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

}
} // namespace optiling
