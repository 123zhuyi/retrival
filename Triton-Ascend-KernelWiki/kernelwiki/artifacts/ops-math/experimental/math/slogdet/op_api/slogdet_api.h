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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_SLOGDET_API_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_SLOGDET_API_H_

#include <tuple>
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

// L0 Slogdet：拉起 AiCore Slogdet kernel（OpType Slogdet），返回 (signValue, logValue) 两个 batch 形状张量。
const std::tuple<const aclTensor*, const aclTensor*> Slogdet(const aclTensor* self, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_OP_API_COMMON_INC_LEVEL0_OP_SLOGDET_API_H_
