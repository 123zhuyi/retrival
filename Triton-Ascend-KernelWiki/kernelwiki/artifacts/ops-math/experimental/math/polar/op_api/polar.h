/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_POLAR_H_
#define OP_API_INC_LEVEL0_POLAR_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * Polar L0：input/angle 已通过 op_api 层 Contiguous + BroadcastTo 对齐到同 shape；
 * 此处分配 complex64 输出张量并调度 AICore kernel。
 */
const aclTensor* Polar(const aclTensor* input, const aclTensor* angle, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_POLAR_H_
