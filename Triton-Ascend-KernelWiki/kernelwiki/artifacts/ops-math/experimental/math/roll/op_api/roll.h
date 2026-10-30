/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file roll.h
 * @brief Roll L0 API.
 */

#ifndef ROLL_OP_API_H_
#define ROLL_OP_API_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* Roll(const aclTensor* x, const aclIntArray* shifts, const aclIntArray* dims, aclOpExecutor* executor);
const aclTensor* Roll(const aclTensor* x,
                      const aclIntArray* shifts,
                      const aclIntArray* dims,
                      aclTensor* out,
                      aclOpExecutor* executor);
} // namespace l0op

#endif // ROLL_OP_API_H_
