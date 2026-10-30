/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/shape_util.h"
#include "op_api/op_util.h"
#include "util/const_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
static graphStatus UpdateDynamicShape(gert::InferShapeContext* context, const gert::Shape* x_shape, const int64_t num_split) {
  for (int64_t i = 0; i < num_split; i++) {
    gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
    *out_shape_dynamic = *x_shape;
  }
  return GRAPH_SUCCESS;
}

static graphStatus UpdatetAllUnknownDim(gert::InferShapeContext* context, const int64_t num_split, const int64_t rank) {
  for (int64_t i = 0; i < num_split; i++) {
    gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
    SetUnknownShape(rank, *out_shape_dynamic);
  }
  return GRAPH_SUCCESS;
}

static graphStatus CheckSplitVParams(const gert::InferShapeContext* context, const gert::Shape*& x_shape,
                                     const int64_t split_dim, const int64_t num_split, const int64_t size_splits_size) {
  OP_CHECK_IF(num_split <= 0,
           OP_LOGE(context->GetNodeName(), "%s",
                                               ConcatString("num_split must be greater than 0, but it's ", num_split).c_str()),
           return GRAPH_FAILED);

  int64_t x_dim_num = x_shape->GetDimNum();
  OP_CHECK_IF(
      !IsDimValid(x_dim_num, split_dim),
      OP_LOGE(context->GetNodeName(),  "%s", GenInvalidDimMsg("split_dim", x_dim_num, split_dim).c_str()),
      return GRAPH_FAILED);

  OP_CHECK_IF(
      size_splits_size != num_split,
      OP_LOGE(
          context->GetNodeName(),  "%s", ConcatString("the size of size_splits must be equal to num_split. ",
                                               "size_splits_size is ", size_splits_size, " num_split is ", num_split).c_str()),
      return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

template <typename T>
graphStatus CalcSplitVOut(gert::InferShapeContext* context, const gert::Tensor* size_splits_vec) {
  const gert::Shape* x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

  const int64_t* numSplit = attrs->GetAttrPointer<int64_t>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);
  int64_t num_split = *numSplit;

  OP_CHECK_IF(IsUnknownRank(*x_shape),
           OP_LOGD(context->GetNodeName(), "input x is unknown rank, will set all output the same as input."),
           return UpdateDynamicShape(context, x_shape, num_split));

  int64_t split_dim = 0;
  static constexpr int64_t input_split_num_idx = 2;
  if (!GetConstInt(context, input_split_num_idx, split_dim)) {
    OP_LOGD(context->GetNodeName(), "get split_dim unsuccessful, will set output to -1.");
    const int64_t input_rank = x_shape->GetDimNum();
    return UpdatetAllUnknownDim(context, num_split, input_rank);
  }

  int64_t size_splits_size = static_cast<int64_t>(size_splits_vec->GetShapeSize());
  OP_CHECK_IF(CheckSplitVParams(context, x_shape, split_dim, num_split, size_splits_size) == GRAPH_FAILED,
           OP_LOGE(context->GetNodeName(), "check split params failed"),
           return GRAPH_FAILED);

  split_dim = split_dim < 0 ? split_dim + x_shape->GetDimNum() : split_dim;
  const T* split_size_value = size_splits_vec->GetData<T>();
  int64_t dynamic_value_idx = -1;
  int64_t split_size_value_sum = 0;
  int64_t dynamic_value_num = 0;

  for (int64_t i = 0; i < size_splits_size; i++) {
    if (split_size_value[i] == -1) {
      dynamic_value_num = dynamic_value_num + 1;
      OP_CHECK_IF(dynamic_value_num > 1,
               OP_LOGE(context->GetNodeName(), "value of split_size can only have one -1"),
               return GRAPH_FAILED);

      dynamic_value_idx = i;
    } else {
      split_size_value_sum = split_size_value_sum + split_size_value[i];
    }
  }

  // update dynamic output
  for (int64_t i = 0; i < num_split; i++) {
    gert::Shape* out_shape = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *x_shape;
    out_shape->SetDim(split_dim, split_size_value[i]);
  }

  if (dynamic_value_idx != -1) {
    gert::Shape* out_shape = context->GetOutputShape(dynamic_value_idx);
    if (x_shape->GetDim(split_dim) == -1) {
      out_shape->SetDim(split_dim, -1);
    } else {
      out_shape->SetDim(split_dim, x_shape->GetDim(split_dim) - split_size_value_sum);
    }
  }

  return GRAPH_SUCCESS;
}

static graphStatus InferShape4SplitV(gert::InferShapeContext* context) {
  const gert::Tensor* size_splits_desc = context->GetInputTensor(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, size_splits_desc);

  DataType size_splits_dtype = size_splits_desc->GetDataType();
  if (size_splits_dtype == DT_INT32) {
    OP_CHECK_IF(CalcSplitVOut<int32_t>(context, size_splits_desc) == GRAPH_FAILED,
             OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v"),
             return GRAPH_FAILED);
  } else {
    OP_CHECK_IF(CalcSplitVOut<int64_t>(context, size_splits_desc) == GRAPH_FAILED,
             OP_LOGE(context->GetNodeName(), "Failed to calculate the output of split_v"),
             return GRAPH_FAILED);
  }

  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(SplitV).InferShape(InferShape4SplitV).InputsDataDependency({1, 2});

}  // namespace ops