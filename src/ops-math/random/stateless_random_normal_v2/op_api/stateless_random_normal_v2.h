/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_STATELESS_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_STATELESS_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* StatelessRandomNormalV2(
    const aclTensor* result, const aclIntArray* key, const aclIntArray* counter, const aclTensor* alg,
    aclOpExecutor* executor);

const aclTensor* StatelessRandomNormalV2(
    const aclTensor* result,
    const aclTensor* keyTensor, const aclTensor* counterTensor,
    const aclTensor* alg, aclOpExecutor* executor);

}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_STATELESS_OP_H_