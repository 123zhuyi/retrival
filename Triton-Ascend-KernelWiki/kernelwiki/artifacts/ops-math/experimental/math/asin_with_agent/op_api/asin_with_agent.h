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
 * @file asin_with_agent.h
 * @brief ACLNN L0 API 接口声明 - AsinWithAgent 算子
 *
 * L0 API 职责：形状推导、Kernel 调度
 */

#ifndef OP_API_INC_LEVEL0_ASIN_WITH_AGENT_H_
#define OP_API_INC_LEVEL0_ASIN_WITH_AGENT_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief AsinWithAgent L0 API 入口
 * @param x   输入张量（支持 FLOAT/FLOAT16/DOUBLE/INT8/INT16/INT32/INT64/UINT8/BOOL）
 * @param executor  算子执行器
 * @return 输出张量（dtype 根据设计：浮点类型与输入相同，整数/BOOL 输出 FLOAT32）
 */
const aclTensor* AsinWithAgent(const aclTensor* x, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ASIN_WITH_AGENT_H_
