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
 * @file add_mat_mat_elements.h
 * @brief ACLNN L0 API 接口声明 - AddMatMatElements 算子
 *
 * L0 API 是 ACLNN 底层实现，由 L2 API 调用。
 * 职责：形状推导、Kernel 调度
 */

#ifndef OP_API_INC_LEVEL0_ADD_MAT_MAT_ELEMENTS_H_
#define OP_API_INC_LEVEL0_ADD_MAT_MAT_ELEMENTS_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief AddMatMatElements L0 API
 *
 * @param a       [in] 输入 tensor a（已保证连续）
 * @param b       [in] 输入 tensor b（已保证连续）
 * @param c       [in] 输入 tensor c（已保证连续）
 * @param alpha   [in] 标量 alpha
 * @param beta    [in] 标量 beta
 * @param executor [in] 执行器
 * @return const aclTensor* 输出 tensor（失败返回 nullptr）
 */
const aclTensor* AddMatMatElements(
    const aclTensor* a, const aclTensor* b, const aclTensor* c, float alpha, float beta, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_ADD_MAT_MAT_ELEMENTS_H_
