/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"

using namespace ge;
namespace ops {
constexpr size_t CONJUGATE_TRANSPOSE_IDX_IN_X = 0;
constexpr size_t CONJUGATE_TRANSPOSE_IDX_IN_PERM = 1;
constexpr size_t CONJUGATE_TRANSPOSE_IDX_OUT_Y = 0;

template <typename T>
static bool ConjugateTransposeInferCommon(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                          const T* permValue, gert::Shape* yShape)
{
    OP_LOGD(context->GetNodeName(), "start to do ConjugateTransposeInferCommon");
    size_t inputDimSize = xShape->GetDimNum();
    yShape->SetDimNum(inputDimSize);
    for (size_t i = 0; i < inputDimSize; ++i) {
        OP_CHECK_IF(
            permValue[i] < 0 || permValue[i] >= static_cast<T>(inputDimSize),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "perm", std::to_string(permValue[i]).c_str(),
                                                  "Each value of perm must be in the range of [0, xShapeDimNum - 1]."
                                                  " The value of perm depends on the number of shape axes of x"),
            return false);
        yShape->SetDim(i, xShape->GetDim(permValue[i]));
    }
    OP_LOGD(context->GetNodeName(), "end to do ConjugateTransposeInferCommon");
    return true;
}

static ge::graphStatus ConjugateTransposeInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do ConjugateTransposeInferShape");
    const gert::Shape* xShape = context->GetInputShape(CONJUGATE_TRANSPOSE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(CONJUGATE_TRANSPOSE_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* permTensor = context->GetInputTensor(CONJUGATE_TRANSPOSE_IDX_IN_PERM);
    OP_CHECK_NULL_WITH_CONTEXT(context, permTensor);

    int64_t permSize = permTensor->GetShapeSize();
    size_t inputDimSize = xShape->GetDimNum();
    OP_CHECK_IF(permSize != static_cast<int64_t>(inputDimSize),
                OP_LOGE_FOR_INVALID_SHAPESIZE(context->GetNodeName(), "perm", ConcatString(permSize).c_str(),
                                              ConcatString(inputDimSize).c_str()),
                return ge::GRAPH_FAILED);

    ge::DataType permDtype = permTensor->GetDataType();
    switch (permDtype) {
        case ge::DT_INT32: {
            const int32_t* permValue = permTensor->GetData<int32_t>();
            if (!ConjugateTransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        case ge::DT_INT64: {
            const int64_t* permValue = permTensor->GetData<int64_t>();
            if (!ConjugateTransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "perm", Ops::Base::ToString(permDtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do ConjugateTransposeInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConjugateTranspose)
    .InferShape(ConjugateTransposeInferShape)
    .InputsDataDependency({CONJUGATE_TRANSPOSE_IDX_IN_PERM});
} // namespace ops
