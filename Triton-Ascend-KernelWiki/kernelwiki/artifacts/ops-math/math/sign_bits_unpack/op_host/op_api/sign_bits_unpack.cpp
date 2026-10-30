/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sign_bits_unpack.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"


using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SignBitsUnpack);
static constexpr size_t OUT_DIM = 2; 
static constexpr size_t OUT_SIZE = 8;

const aclTensor *SignBitsUnpack(const aclTensor *self, int64_t size, op::DataType dtype, aclOpExecutor *executor) {
    L0_DFX(SignBitsUnpack, self, size, dtype);

    int64_t selfDimOne = self->GetViewShape().GetDim(0);
    op::Shape outShape;
    outShape.SetDimNum(OUT_DIM);
    outShape.SetDim(0, size);

    if(size <= 0) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "size is zero or less than zero, cannot perform division.");
        return nullptr;
    }
    outShape.SetDim(1, (selfDimOne * OUT_SIZE) / size);

    auto out = executor->AllocTensor(outShape, dtype, op::Format::FORMAT_ND);
    if(out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc out tensor failed.");
        return nullptr;
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SignBitsUnpack, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(size, dtype));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SignBitsUnpack ADD_TO_LAUNCHER_LIST_AICORE failed."), 
        return nullptr);

    return out;
}
}  // namespace l0op