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
 * \file chunk_cat_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "op_api/op_util.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {

static ge::graphStatus InferShape4ChunkCat(gert::InferShapeContext* context)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    uint32_t inputNum = anchorInstanceInfo->GetInstanceNum();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t dim = *attrs->GetAttrPointer<int64_t>(0);
    const int64_t numChunks = *attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_IF(dim != 0,
        OP_LOGE(context->GetNodeName(), "dim only support 0 now"),
        return ge::GRAPH_FAILED);

    auto outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(dim + 2); // 2是由于输出的维度在dim维后所有维度合维
    outShape->SetDim(dim, numChunks);

    int64_t outputCol = 0;
    for (uint32_t i = 0; i < inputNum; i++) {
        auto inputTensorShape = context->GetDynamicInputShape(0, i);

        int64_t chunkDimSize = inputTensorShape->GetDim(dim);
        int64_t chunkCol = (chunkDimSize + numChunks - 1) / numChunks;

        int64_t dim1Size = chunkCol;
        int64_t inputTensorDimNum = inputTensorShape->GetDimNum();
        for (int64_t j = 0; j < dim; j++) {
            outShape->SetDim(j, inputTensorShape->GetDim(j));
        }
        for (int64_t j = dim + 1; j < inputTensorDimNum; j++) {
            dim1Size *= inputTensorShape->GetDim(j);
        }
        outputCol += dim1Size;
    }
    outShape->SetDim(dim + 1, outputCol);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkCat).InferShape(InferShape4ChunkCat);
} // namespace ops
