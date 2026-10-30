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
 * \file tile_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {
constexpr size_t ATTR_INDEX_MULTIPLES = 0;
constexpr size_t TILE_IN_IDX = 0;
constexpr size_t TILE_OUT_IDX = 0;
static constexpr size_t MAXDIMNUM = 8;
constexpr size_t INPUT_INDEX_MULTIPLES = 1;

template <typename T>
ge::graphStatus TileInferShapeCommon(gert::InferShapeContext* context, const T* multiples_data, size_t multiples_len) {
  auto in_shape = context->GetInputShape(TILE_IN_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
  auto out_shape = context->GetOutputShape(TILE_OUT_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto in_shape_len = in_shape->GetDimNum();
  OP_CHECK_IF(multiples_len > MAXDIMNUM,
           OP_LOGE(context->GetNodeName(), "the tile multiples len is more than MaxDimNum 8"),
           return ge::GRAPH_FAILED);
  // align shape for input
  gert::Shape in_shape_new;
  if (in_shape_len < multiples_len) {
    OP_LOGI(context->GetNodeName(), "The tile multiples len is more than the input len.");
    int32_t len_diff = multiples_len - in_shape_len;
    for (int32_t i = 0; i < len_diff; i++) {
      in_shape_new.AppendDim(1);
    }
    for (size_t i = 0; i < in_shape_len; i++) {
      in_shape_new.AppendDim(in_shape->GetDim(i));
    }
    in_shape_len = multiples_len;
  } else {
    OP_LOGI(context->GetNodeName(), "The tile multiples len is less or equal than the input len.");
    in_shape_new = *in_shape;
  }
  // in shape == [], out shape = []
  if (in_shape_len == 0) {
    OP_LOGI(context->GetNodeName(), "input shape is [], output shape is [].");
    *out_shape = *in_shape;
    return GRAPH_SUCCESS;
  }
  // calculate output shape dim value
  out_shape->SetDimNum(in_shape_len);
  for (uint64_t i = 0; i < in_shape_len; i++) {
    if (in_shape_new[i] >= 0) {
      int32_t multiples_index = multiples_len - in_shape_len + i;
      out_shape->SetDim(i, in_shape_new[i] * (multiples_index >= 0 ? multiples_data[i] : 1));
    } else {
      OP_LOGE(context->GetNodeName(), "Runtime infershape illegal input dim:%lu, value is %ld", i, in_shape_new[i]);
      return ge::GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Tile(gert::InferShapeContext* context) {
  auto x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
  auto multiples_tensor = context->GetInputTensor(INPUT_INDEX_MULTIPLES);
  OP_CHECK_NULL_WITH_CONTEXT(context, multiples_tensor);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
  auto multiples_size = static_cast<size_t>(multiples_tensor->GetShapeSize());
  OP_LOGD(context->GetNodeName(), "multiples_size is %zu", multiples_size);
  DataType data_type = multiples_tensor->GetDataType();
  OP_CHECK_IF((data_type != DT_INT32) && (data_type != DT_INT64),
           OP_LOGE(
               context->GetNodeName(),
               "multiples's dtype %s must be in (int32,int64)!", ToString(data_type).c_str()),
           return ge::GRAPH_FAILED);

  if (data_type == DT_INT32) {
    const int32_t* multiples_data = multiples_tensor->GetData<int32_t>();
    return TileInferShapeCommon<int32_t>(context, multiples_data, multiples_size);
  }
  const int64_t* multiples_data = multiples_tensor->GetData<int64_t>();
  return TileInferShapeCommon<int64_t>(context, multiples_data, multiples_size);
}

IMPL_OP_INFERSHAPE(Tile).InferShape(InferShape4Tile).InputsDataDependency({INPUT_INDEX_MULTIPLES});
}  // namespace ops
