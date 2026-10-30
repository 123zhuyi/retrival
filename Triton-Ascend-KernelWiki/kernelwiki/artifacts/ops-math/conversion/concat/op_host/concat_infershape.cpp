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
 * \file concat_infershape.cpp
 * \brief
 */
#include "concat_infershape.h"
#include "log/log.h"
#include "op_api/op_util.h"

using namespace ge;
namespace ops {

ge::graphStatus ConcatInferShapeCommon(
    gert::InferShapeContext* context, const int64_t dynamic_input_idx, int64_t num_concat, int64_t axis)
{
    auto in_shape = context->GetDynamicInputShape(dynamic_input_idx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    OP_LOGD(context->GetNodeName(), "input_shape 0:%s", Ops::Base::ToString(*in_shape).c_str());
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;

    if (num_concat == 1) {
        // dynamic case or the input only one will use dynamic infer func
        return ge::GRAPH_SUCCESS;
    }

    if (out_shape->IsScalar()) {
        // scalar to shape [1]
        out_shape->AppendDim(1);
    }

    size_t output_dim = out_shape->GetDimNum();
    OP_CHECK_IF(
        !IsDimValid(output_dim, axis),
        OP_LOGE(context->GetNodeName(), "%s", GenInvalidDimMsg("concat_dim", output_dim, axis).c_str()),
        return ge::GRAPH_FAILED);
    if (axis < 0) {
        axis += output_dim;
    }

    int64_t concat_dim_size = out_shape->GetDim(axis);
    for (int64_t relative_index = 1; relative_index < num_concat; relative_index++) {
        const gert::Shape* input_i_shape = context->GetDynamicInputShape(dynamic_input_idx, relative_index);
        OP_LOGD(
            context->GetNodeName(), "input_shape %ld:%s", relative_index, Ops::Base::ToString(*input_i_shape).c_str());
        if (input_i_shape->IsScalar() && output_dim == 1) {
            concat_dim_size += 1;
            continue;
        }
        if (input_i_shape->GetDimNum() != output_dim) {
            // input shape size is not equal output
            OP_LOGE(
                context->GetNodeName(), "shape[%" PRId64 "].GetDimNum %zu, must be equal to %zu!", relative_index,
                input_i_shape->GetDimNum(), output_dim);
            return ge::GRAPH_FAILED;
        }
        // check whether the non concat dim is equal
        for (int64_t check_dim = 0; check_dim < static_cast<int64_t>(output_dim); check_dim++) {
            if (check_dim != axis && input_i_shape->GetDim(check_dim) != out_shape->GetDim(check_dim)) {
                OP_LOGE(
                    context->GetNodeName(),
                    "shape[%" PRId64 "][%" PRId64 "] is %" PRId64 ", must be equal to %" PRId64 "!", relative_index,
                    check_dim, input_i_shape->GetDim(check_dim), out_shape->GetDim(check_dim));
                return ge::GRAPH_FAILED;
            }
        }
        concat_dim_size += input_i_shape->GetDim(axis);
    }
    out_shape->SetDim(axis, concat_dim_size);
    OP_LOGD(context->GetNodeName(), "output_shape:%s", Ops::Base::ToString(*out_shape).c_str());
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline static int64_t GetConcatDim(gert::InferShapeContext* context, int64_t dimIdx)
{
    auto concatDimTensor = context->GetRequiredInputTensor(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimTensor);
    const T* concatDimValPtr = concatDimTensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimValPtr);

    int64_t concatDim = concatDimValPtr[0];
    return concatDim;
}

ge::graphStatus InferShapeForConcatAndConcatV2(gert::InferShapeContext* context, int64_t inputIdx, int64_t dimIdx)
{
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorInstanceInfo);
    int64_t inputNum = static_cast<int64_t>(anchorInstanceInfo->GetInstanceNum());
    auto concatDimPtr = context->GetRequiredInputDesc(dimIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, concatDimPtr);
    ge::DataType concatDimType = concatDimPtr->GetDataType();
    int64_t concatDim = 0;
    if (concatDimType == ge::DT_INT32) {
        concatDim = GetConcatDim<int32_t>(context, dimIdx);
    } else {
        concatDim = GetConcatDim<int64_t>(context, dimIdx);
    }

    return ConcatInferShapeCommon(context, inputIdx, inputNum, concatDim);
}

static ge::graphStatus InferShape4Concat(gert::InferShapeContext* context)
{
    return InferShapeForConcatAndConcatV2(context, INPUT_IDX, INDEX_CONCAT_DIM);
}

IMPL_OP_INFERSHAPE(Concat).InferShape(InferShape4Concat).InputsDataDependency({INDEX_CONCAT_DIM});
} // namespace ops
