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
 * \file strided_slice.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "strided_slice_util.h"
#include "log/log.h"
#include "util/const_util.h"

namespace ops {
using namespace ge;

static const size_t IDX_X = 0;
static const size_t IDX_BEGIN = 1;
static const size_t IDX_END = 2;
static const size_t IDX_STRIDES = 3;
static const size_t IDX_Y = 0;
static const size_t IDX_MASK_BEGIN = 0;
static const size_t IDX_MASK_END = 1;
static const size_t IDX_MASK_ELLIPSIS = 2;
static const size_t IDX_MASK_NEW_AXIS = 3;
static const size_t IDX_MASK_SHRINK_AXIS = 4;

template <typename T>
static void GetConstValueToShape(const gert::Tensor* tensor, size_t size, gert::Shape* shape)
{
    const T* value = tensor->GetData<T>();
    shape->SetDimNum(size);
    for (size_t i = 0; i < size; i++) {
        shape->SetDim(i, value[i]);
    }
}

static bool GetValueList(size_t idx, const gert::Tensor* tensor, int64_t size, QuickVector& valueList)
{
    if (size > 0) {
        if (tensor->GetDataType() == ge::DT_INT32) {
            GetConstValueToShape<int32_t>(tensor, size, &valueList);
        } else if (tensor->GetDataType() == ge::DT_INT64) {
            GetConstValueToShape<int64_t>(tensor, size, &valueList);
        } else {
            OP_LOGD(OP_NAME, "input[%zu] data type is invalid: %d", idx, tensor->GetDataType());
        }
    }

    return (valueList.GetDimNum() != 0);
}

static int64_t CalcMaxShapeSize(int64_t begin_shape_size, int64_t end_shape_size, int64_t strides_shape_size)
{
    int64_t shape_max = -1L;

    shape_max = std::max(begin_shape_size, shape_max);
    shape_max = std::max(end_shape_size, shape_max);
    shape_max = std::max(strides_shape_size, shape_max);

    return shape_max;
}

static bool CheckInputValid(int64_t shape_max, bool valid_strides)
{
    if (shape_max == -1L) {
        OP_LOGD(OP_NAME, "max shape is -1.");
        return false;
    }

    if (!valid_strides) {
        OP_LOGD(OP_NAME, "strides is invalid.");
        return false;
    }

    return true;
}

static void MakeParamSameLen(int64_t shape_max, const gert::Shape* shape_x, QuickVector& list_begin,
                             QuickVector& list_end, QuickVector& list_strides)
{
    // If  begin is invalid, set begin with begin_len count of 0, for inference output ranges.
    // For example, begin_len is 2 set begin's value to [0, 0]
    if (list_begin.GetDimNum() == 0) {
        for (size_t i = 0; i < static_cast<size_t>(shape_max); i++) {
            list_begin.AppendDim(0);
        }
    }

    // If end is invalid, set end with begin_len count with same index of the input shape dims, for inference output
    // ranges. If begin_len greater than the length of input shape, set the end[i] to input_shape.back()
    // which i >= input_shape.size().
    // For example, begin_len is 2 and input shape is (5, 6, 7, 8), set end's value to [5, 6].
    //              begin_len is 5 and input shape is (5, 6, 7, 8), set end's value to [5, 6, 7, 8, 8].
    if (list_end.GetDimNum() == 0) {
        size_t dim_num_x = shape_x->GetDimNum();
        if (shape_max < static_cast<int64_t>(dim_num_x)) {
            for (size_t i = 0; static_cast<int64_t>(i) < shape_max; i++) {
                list_end.AppendDim(shape_x->GetDim(i));
            }
        } else {
            for (size_t i = 0; i < dim_num_x; i++) {
                list_end.AppendDim(shape_x->GetDim(i));
            }
            for (size_t i = dim_num_x; static_cast<int64_t>(i) < shape_max; i++) {
                list_end.AppendDim(shape_x->GetDim(dim_num_x - 1));
            }
        }
    }

    OP_LOGD(OP_NAME, "begin_list:%s", Ops::Base::ToString(list_begin).c_str());
    OP_LOGD(OP_NAME, "end_list:%s", Ops::Base::ToString(list_end).c_str());
    OP_LOGD(OP_NAME, "stride_list:%s", Ops::Base::ToString(list_strides).c_str());
}

static ge::graphStatus InferShape4StridedSlice(gert::InferShapeContext* context)
{
    StridedSliceParams input_params;

    const auto shape_x = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x);
    auto shape_y = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_y);
    auto shape_begin = context->GetInputShape(IDX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_begin);
    auto shape_end = context->GetInputShape(IDX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_end);
    auto shape_strides = context->GetInputShape(IDX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_strides);
    auto tensor_begin = context->GetInputTensor(IDX_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor_begin);
    auto tensor_end = context->GetInputTensor(IDX_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor_end);
    auto tensor_strides = context->GetInputTensor(IDX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor_strides);

    // calc max shape of (begin, end, strides)
    int64_t shape_max = CalcMaxShapeSize(shape_begin->GetDim(0), shape_end->GetDim(0), shape_strides->GetDim(0));

    // construct (input_params.list_begin, input_params.list_end, input_params.list_strides)
    bool valid_begin = GetValueList(IDX_BEGIN, tensor_begin, shape_begin->GetDim(0), input_params.begin);
    bool valid_end = GetValueList(IDX_END, tensor_end, shape_end->GetDim(0), input_params.end);
    bool valid_strides = GetValueList(IDX_STRIDES, tensor_strides, shape_strides->GetDim(0), input_params.strides);
    // necessary input valid check
    if (!CheckInputValid(shape_max, valid_strides)) {
        shape_y->SetDimNum(0);
        shape_y->AppendDim(UNKNOWN_DIM_NUM);
        return GRAPH_SUCCESS;
    }

    // make (begin, end, strides) same lenght
    MakeParamSameLen(shape_max, shape_x, input_params.begin, input_params.end, input_params.strides);

    // check (begin, end) shape size same
    if (input_params.end.GetDimNum() != input_params.begin.GetDimNum()) {
        OP_LOGE(OP_NAME, "end shape, begin shape length mismatch!");
        return GRAPH_FAILED;
    }

    // Get relevant masks from const node
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* mask_begin = attrs->GetAttrPointer<int64_t>(IDX_MASK_BEGIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_begin);
    const int64_t* mask_end = attrs->GetAttrPointer<int64_t>(IDX_MASK_END);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_end);
    const int64_t* mask_ellipsis = attrs->GetAttrPointer<int64_t>(IDX_MASK_ELLIPSIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_ellipsis);
    const int64_t* mask_new_axis = attrs->GetAttrPointer<int64_t>(IDX_MASK_NEW_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_new_axis);
    const int64_t* mask_shrink_axis = attrs->GetAttrPointer<int64_t>(IDX_MASK_SHRINK_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, mask_shrink_axis);

    // infer shape
    input_params.input_shape = *shape_x;
    input_params.begin_mask = static_cast<uint64_t>(*mask_begin);
    input_params.end_mask = static_cast<uint64_t>(*mask_end);
    input_params.ellipsis_mask = static_cast<uint64_t>(*mask_ellipsis);
    input_params.new_axis_mask = static_cast<uint64_t>(*mask_new_axis);
    input_params.shrink_axis_mask = static_cast<uint64_t>(*mask_shrink_axis);
    input_params.begin_valid = valid_begin;
    input_params.end_valid = valid_end;
    input_params.stride_valid = valid_strides;
    if (!InferShape(input_params, shape_y)) {
        return GRAPH_FAILED;
    }

    OP_LOGD(context, "output_shape:%s", Ops::Base::ToString(*shape_y).c_str());
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StridedSlice)
    .InferShape(InferShape4StridedSlice)
    .InputsDataDependency({IDX_BEGIN, IDX_END, IDX_STRIDES});
} // namespace ops
