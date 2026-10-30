/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "triangular_solve.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(TriangularSolve);

// AICPU算子kernel
static const aclTensor* TriangularSolveAiCPU(const aclTensor *self, const aclTensor *A, bool upper,
                                             bool transpose, aclTensor *X, aclOpExecutor *executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST，将AiCPU MatrixTriangularSolve
    // TriangularSolve, self, A, upper, transpose是算子的输入，X是算子的输出
    L0_DFX(TriangularSolveAiCPU, self, A, upper, transpose, X);
    static internal::AicpuTaskSpace space("MatrixTriangularSolve", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(TriangularSolve,
                                          OP_ATTR_NAMES({ "lower", "adjoint" }),
                                          OP_INPUT(A, self),
                                          OP_OUTPUT(X),
                                          OP_ATTR(!upper, transpose));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);

    return X;
}

// 只支持 AICPU
const aclTensor* TriangularSolve(const aclTensor *self, const aclTensor *A, bool upper,
                                 bool transpose, const aclTensor *xOut, aclOpExecutor *executor)
{
    auto X = executor->AllocTensor(xOut->GetViewShape(), self->GetDataType(), self->GetStorageFormat());
    return TriangularSolveAiCPU(self, A, upper, transpose, X, executor);
}
}

