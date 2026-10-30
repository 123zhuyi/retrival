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
 * \file affine_grid.cpp
 * \brief
 */

#include "affine_grid.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AffineGrid);

// AICPU算子kernel
static const aclTensor* AffineGridAiCpu(const aclTensor *theta, const aclTensor *size, bool alignCorners,
                                        aclTensor* out, aclOpExecutor* executor) {
  L0_DFX(AffineGridAiCpu, theta, size, alignCorners, out);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Arange算子加入任务队列
  static internal::AicpuTaskSpace space("AffineGrid");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(AffineGrid, OP_ATTR_NAMES({"align_corners"}), OP_INPUT(theta, size),
                                        OP_OUTPUT(out), OP_ATTR(alignCorners));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AffineGridAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *AffineGrid(const aclTensor *theta, const aclIntArray *size, bool alignCorners,
                            aclOpExecutor *executor) {
  auto sizeTensor = executor->ConvertToTensor(size, DataType::DT_INT32);

  op::Shape outShape;
  outShape.AppendDim((*size)[0]);
  outShape.AppendDim(size->Size() == 4 ? (*size)[2] * (*size)[3] : (*size)[2] * (*size)[3] * (*size)[4]);
  outShape.AppendDim(size->Size() == 4 ? 2 : 3);
  auto out = executor->AllocTensor(outShape, theta->GetDataType());

  return AffineGridAiCpu(theta, sizeTensor, alignCorners, out, executor);
}
}  // namespace l0op
