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
#include "util/math_util.h"

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

static graphStatus CheckSplitParams(const gert::InferShapeContext* context, const gert::Shape*& x_shape, int64_t split_dim,
                                    const int64_t num_split) {
  OP_CHECK_IF(num_split <= 0,
           OP_LOGE(context->GetNodeName(), "%s",
                                               ConcatString("num_split must be greater than 0, but it's ", num_split).c_str()),
           return GRAPH_FAILED);

  int64_t x_shape_dim = x_shape->GetDimNum();

  OP_CHECK_IF(!IsDimValid(x_shape_dim, split_dim),
           OP_LOGE(context->GetNodeName(),  "%s",
                                               GenInvalidDimMsg("split_dim", x_shape_dim, split_dim).c_str()),
           return GRAPH_FAILED);

  if (split_dim < 0) {
    split_dim += x_shape_dim;
  }

  OP_CHECK_IF((x_shape->GetDim(split_dim) % num_split != 0) && (x_shape->GetDim(split_dim) != -1),
           OP_LOGE(
               context->GetNodeName(),  "%s", ConcatString("the split_dim dimension of x_shape must be divided by num_split.",
                                                    " x_shape is ", ToString(*x_shape), ", x_shape on split_dim is ",
                                                    x_shape->GetDim(split_dim), ", num_split is ", num_split).c_str()),
           return GRAPH_FAILED);

  return GRAPH_SUCCESS;
}

static void CalOutShape(const gert::Shape*& x_shape, gert::Shape*& out_shape, const int64_t num_split,
                        const int64_t split_dim) {
  int64_t output_dim_size = x_shape->GetDim(split_dim) / num_split;
  *out_shape = *x_shape;
  out_shape->SetDim(split_dim, output_dim_size);
}

static graphStatus InferShape4Split(gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(1);
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
  if (!GetConstInt(context, 0, split_dim)) {
    OP_LOGD(context->GetNodeName(), "get split_dim unsuccessful, will set output to -1.");
    const int64_t input_rank = x_shape->GetDimNum();
    return UpdatetAllUnknownDim(context, num_split, input_rank);
  }

  OP_CHECK_IF(CheckSplitParams(context, x_shape, split_dim, num_split) == GRAPH_FAILED,
           OP_LOGE(context->GetNodeName(), "check split params failed."),
           return GRAPH_FAILED);

  split_dim = split_dim < 0 ? split_dim + x_shape->GetDimNum() : split_dim;
  if (x_shape->GetDim(split_dim) == -1) {
    OP_LOGD(context->GetNodeName(), "the split dim is -1 input x, will set all output the same as input.");
    return UpdateDynamicShape(context, x_shape, num_split);
  }

  gert::Shape* out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  CalOutShape(x_shape, out_shape, num_split, split_dim);

  // update dynamic output
  for (int64_t i = 1; i < num_split; i++) {
    gert::Shape* out_shape_dynamic = context->GetOutputShape(i);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_dynamic);
    *out_shape_dynamic = *out_shape;
  }

  return GRAPH_SUCCESS;
}

static graphStatus InferShapeRange4Split(gert::InferShapeRangeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferShapeRange4Split start");

  int64_t SPLIT_DIM_DATA_LEN = 3;
  auto split_dim_range = context->GetInputShapeRange(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, split_dim_range);
  auto x_range = context->GetInputShapeRange(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_range);
  auto y_range = context->GetOutputShapeRange(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_range);

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  const int64_t* numSplit = attrs->GetAttrPointer<int64_t>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, numSplit);
  int64_t num_split = *numSplit;

  OP_CHECK_NULL_WITH_CONTEXT(context, x_range->GetMax());
  int64_t x_dim = x_range->GetMax()->GetDimNum();
  int64_t split_dim_data_len = strlen(ToString(*split_dim_range->GetMax()).c_str());
  if (split_dim_data_len != SPLIT_DIM_DATA_LEN) {
    OP_LOGD(context->GetNodeName(), "Get constValue unsuccessful of [split_dim_data]");
    if (x_range->GetMax()->GetDimNum() == 1) {
      if (x_range->GetMax()->GetDim(0) == -1) {
          y_range->GetMax()->SetDimNum(1);
          y_range->GetMax()->SetDim(0, -1);
          y_range->GetMin()->SetDim(0, 0);
      } else {
          y_range->GetMax()->SetDimNum(1);
          y_range->GetMax()->SetDim(0, Ops::Base::CeilDiv(x_range->GetMax()->GetDim(0), num_split));
          y_range->GetMin()->SetDim(0, Ops::Base::CeilDiv(x_range->GetMax()->GetDim(0), num_split));
      }
    } else {
      for (size_t i = 0; i < x_range->GetMax()->GetDimNum(); ++i) {
        y_range->GetMax()->SetDimNum(x_range->GetMax()->GetDimNum());
        y_range->GetMax()->SetDim(i, x_range->GetMax()->GetDim(i));
        y_range->GetMin()->SetDim(i, 0);
      }
    }
    OP_LOGD(context->GetNodeName(), "InferShapeRange4Split end");
    return GRAPH_SUCCESS;
  }

  auto split_dim_num = split_dim_range->GetMin()->GetDimNum();
  int64_t split_dim = 1;
  if (split_dim_num > 0) {
    split_dim = split_dim_range->GetMin()->GetDim(0);
  }

  if (split_dim < 0) {
    split_dim += x_dim;
  }

  for (int64_t i = 0; i < x_dim; ++i) {
    if (split_dim == static_cast<int>(i)) {
      x_range->GetMin()->GetDim(i) == 1 ? y_range->GetMin()->SetDim(i, 1) : \
                  y_range->GetMin()->SetDim(i, floor(static_cast<float>(x_range->GetMin()->GetDim(i))\
                  / static_cast<float>(num_split)));

      x_range->GetMax()->GetDim(i) == -1 ? y_range->GetMax()->SetDim(i, -1) :
                  y_range->GetMax()->SetDim(i, Ops::Base::CeilDiv(x_range->GetMax()->GetDim(i), num_split));
    } else {
      y_range->GetMax()->SetDim(i, x_range->GetMax()->GetDim(i));
      y_range->GetMin()->SetDim(i, x_range->GetMin()->GetDim(i));
    }
  }
  OP_LOGD(context->GetNodeName(), "InferShapeRange4Split end");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType4Split(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4Split start");
  auto input_x_dtype = context->GetInputDataType(1);
  context->SetOutputDataType(0, input_x_dtype);
  OP_LOGD(context->GetNodeName(), "InferDataType4Split end");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Split).InferShape(InferShape4Split).InputsDataDependency({0})
              .InferShapeRange(InferShapeRange4Split)
              .InferDataType(InferDataType4Split);

}  // namespace ops