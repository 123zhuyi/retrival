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
 * \file dropout_gen_mask_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"


namespace {
constexpr size_t kMatrixDimNum = 2U;
constexpr size_t kInputIndex0 = 0U;
}  // namespace

using namespace ge;
namespace ops {
static int64_t InferShapeGetShapeSizeBytes(int64_t shape_size) {
  if (shape_size < 0) {
    return ge::UNKNOWN_DIM;
  }

  const int64_t bit_number = 128;  
  // 128 bits are generated at a time.
  int64_t n128s = shape_size / bit_number;
  if (shape_size % bit_number != 0) {
    n128s++;
  }

  // 128 bits occupy 16 bytes.
  int64_t n8s = n128s * 16;

  return n8s;
}

template<typename T>
static graphStatus InferShapeImpl(const T *shape_data, gert::Shape &output_shape, int64_t shape_size) {
  int64_t output_shapesize = 1;
  for (int32_t i = 0; i < shape_size; i++) {
    output_shapesize *=  shape_data[i];
  }
  int64_t output_shape_num = InferShapeGetShapeSizeBytes(output_shapesize);
  output_shape.SetDimNum(1);
  output_shape.SetDim(0, output_shape_num);
  return ge::GRAPH_SUCCESS;
}

static graphStatus DropOutGenMaskInferShapeFunc(gert::InferShapeContext *context) {
  auto shape_tensor = context->GetInputTensor(0);
  auto output_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, shape_tensor);
  OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);

  auto shape_size = shape_tensor->GetShapeSize();

  if (shape_tensor->GetDataType() == ge::DT_INT32) {
    auto shape_data = shape_tensor->GetData<int32_t>();
    return InferShapeImpl<int32_t>(shape_data, *output_shape, shape_size);
  } else {
    auto shape_data = shape_tensor->GetData<int64_t>();
    return InferShapeImpl<int64_t>(shape_data, *output_shape, shape_size);
  }
}

IMPL_OP_INFERSHAPE(StatelessDropOutGenMask)
    .InputsDataDependency({kInputIndex0})
    .InferShape(DropOutGenMaskInferShapeFunc);

}  // namespace ops