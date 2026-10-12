/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stateless_exponential.cpp
 * \brief Op API (l0) implementation for StatelessExponential.
 *
 * seed/offset are passed as INPUT tensors (not ATTRs) so they can flow from tensor-based
 * callers (e.g. aclnnMultinomialTensor). self is both INPUT and OUTPUT (in-place); the
 * caller is responsible for allocating the buffer to be filled.
 */
#include "stateless_exponential.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(StatelessExponential);

const aclTensor* StatelessExponential(const aclTensor* self, const aclTensor* seed, const aclTensor* offset,
                                      float lambd, aclOpExecutor* executor)
{
    L0_DFX(StatelessExponential, self, seed, offset, lambd);

    // In-place: self is both the input and the output. seed/offset are value-dependent
    // scalar inputs consumed at tiling time.
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(StatelessExponential, OP_INPUT(self, seed, offset), OP_OUTPUT(self),
                                           OP_ATTR(lambd));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return self;
}

} // namespace l0op
