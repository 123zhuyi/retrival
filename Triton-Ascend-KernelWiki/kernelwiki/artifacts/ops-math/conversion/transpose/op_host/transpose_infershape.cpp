/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
constexpr size_t TRANSPOSE_IDX_IN_X = 0;
constexpr size_t TRANSPOSE_IDX_IN_PERM = 1;
constexpr size_t TRANSPOSE_IDX_OUT_Y = 0;

template <typename T>
static bool TransposeInferCommon(const gert::InferShapeContext* context, const gert::Shape* xShape, const T* permValue,
                                 gert::Shape* yShape)
{
    OP_LOGD(context->GetNodeName(), "start to do TransposeInferCommon");
    size_t inputDimSize = xShape->GetDimNum();
    yShape->SetDimNum(inputDimSize);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    int64_t insertedByFe = 0;
    if (attrs->GetAttrNum() > 0) {
        const int64_t* insertedByFeFlag = attrs->GetInt(0);
        insertedByFe = insertedByFeFlag == nullptr ? 0 : *insertedByFeFlag;
    }
    if (insertedByFe == 0) {
        for (size_t i = 0; i < inputDimSize; ++i) {
            OP_CHECK_IF(!IsDimValid(inputDimSize, permValue[i]),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            context->GetNodeName(), "perm", std::to_string(permValue[i]).c_str(),
                            "Each value of perm must be in the range of [-xShapeDimNum, xShapeDimNum - 1]."
                            " The value of perm depends on the number of shape axes of x"),
                        return false);
            T permV = permValue[i] < 0 ? permValue[i] + inputDimSize : permValue[i];
            yShape->SetDim(i, xShape->GetDim(permV));
        }
    } else {
        for (size_t i = 0; i < inputDimSize; ++i) {
            yShape->SetDim(i, xShape->GetDim(i));
        }
    }

    OP_LOGD(context->GetNodeName(), "end to do TransposeInferCommon");
    return true;
}

static ge::graphStatus TransposeInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do TransposeInferShape");
    const gert::Shape* xShape = context->GetInputShape(TRANSPOSE_IDX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* yShape = context->GetOutputShape(TRANSPOSE_IDX_OUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::Tensor* permTensor = context->GetInputTensor(TRANSPOSE_IDX_IN_PERM);
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
            if (!TransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        case ge::DT_INT64: {
            const int64_t* permValue = permTensor->GetData<int64_t>();
            if (!TransposeInferCommon(context, xShape, permValue, yShape)) {
                return ge::GRAPH_FAILED;
            }
            break;
        }
        default:
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "perm", Ops::Base::ToString(permDtype).c_str(),
                                      "int32 or int64");
            return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "End to do TransposeInferShape");
    return ge::GRAPH_SUCCESS;
}
static int64_t privateDefaultValue = 0;
IMPL_OP_INFERSHAPE(Transpose)
    .InferShape(TransposeInferShape)
    .InputsDataDependency({TRANSPOSE_IDX_IN_PERM})
    .PrivateAttr("_inserted_by_fe", privateDefaultValue);
} // namespace ops
