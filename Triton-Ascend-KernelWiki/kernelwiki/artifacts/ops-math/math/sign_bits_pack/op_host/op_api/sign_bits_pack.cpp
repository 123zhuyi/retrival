/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sign_bits_pack.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SignBitsPack);
static constexpr size_t OUT_DIM = 2;

const aclTensor* SignBitsPack(const aclTensor* self, int64_t size,aclOpExecutor* executor) {
  L0_DFX(SignBitsPack, self, size);
  
  int64_t selfDimOne = self->GetViewShape().GetDim(0);
  int64_t outDimTwo = 0;

  auto ysize = (selfDimOne + 7) / 8;
  if(size != 0)
  {
    outDimTwo = ysize / size;
  }
  
  op::Shape outShape;
  outShape.SetDimNum(OUT_DIM);
  outShape.SetDim(0, size);
  outShape.SetDim(1, outDimTwo);

  auto out = executor->AllocTensor(outShape, op::DataType::DT_UINT8, op::Format::FORMAT_ND);
  CHECK_RET(out != nullptr, nullptr);

  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SignBitsPack, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(size));
  OP_CHECK(ret ==  ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SignBitsPackAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
    return nullptr);
  return out;
}
}  // namespace l0op
