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
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file mul_no_nan.h
 * @brief ACLNN L0 API declaration for MulNoNan
 */

#ifndef OP_API_INC_LEVEL0_MUL_NO_NAN_H_
#define OP_API_INC_LEVEL0_MUL_NO_NAN_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* MulNoNan(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_MUL_NO_NAN_H_
