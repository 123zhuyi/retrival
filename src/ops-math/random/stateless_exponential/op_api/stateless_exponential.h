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
 * \file stateless_exponential.h
 * \brief Op API header for StatelessExponential
 */
#ifndef STATELESS_EXPONENTIAL_OP_API_H
#define STATELESS_EXPONENTIAL_OP_API_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief Fill `self` in-place with Exp(lambda) random numbers (Philox4x32-10).
 *
 * This is an in-place operator: the random numbers are written into the tensor passed as
 * `self`, and `self` itself is returned. To avoid overwriting an existing weights tensor,
 * the caller should pre-allocate a fresh tensor (e.g. executor->AllocTensor with self's
 * shape/dtype) and pass that as `self`.
 *
 * @param self     Tensor to be filled in-place with Exp(lambda) samples (FP16/BF16/FP32, ND)
 * @param seed     Seed tensor (INT64 scalar)
 * @param offset   Offset tensor (INT64 scalar, must be a multiple of 4)
 * @param lambd    Rate parameter lambda, must be > 0
 * @param executor Op executor
 * @return self (same tensor, now holding exponential random numbers)
 */
const aclTensor* StatelessExponential(const aclTensor* self, const aclTensor* seed, const aclTensor* offset,
                                      float lambd, aclOpExecutor* executor);

} // namespace l0op

#endif // STATELESS_EXPONENTIAL_OP_API_H
