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
 * \file dot.cpp
 * \brief
 */
#include "dot.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Dot);

const aclTensor *Dot(const aclTensor *self, const aclTensor *tensor, aclOpExecutor *executor) {
  L0_DFX(Dot, self, tensor);

  // 根据Dot算子语义，通过输入shape推导算子输出shape
  Shape outShape;
  outShape.SetDimNum(0);
  // 根据推导出的输出shape申请输出tensor
  auto out = executor->AllocTensor(outShape, self->GetDataType(), Format::FORMAT_ND);
  if (out == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
    return nullptr;
  }

  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将Dot算子加入任务队列
  // Dot是算子的OpType，self、tensor是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Dot, OP_INPUT(self, tensor), OP_OUTPUT(out));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "DotAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}

}  // namespace l0op