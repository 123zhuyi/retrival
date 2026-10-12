/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_DROPOUT_V3_GRAD_H_
#define OP_API_INC_DROPOUT_V3_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDropoutV3Grad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：dropout反向，按照mask比特位将gradY对应元素置零或按scale缩放。
 * 计算公式：gradX_i = (mask_i == 1) ? scale * gradY_i : 0
 *
 * @param [in] gradY: npu device侧的aclTensor。
 * 数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND，shape为0-8维，支持非连续的Tensor，支持空Tensor。
 * @param [in] mask: 位掩码数据，npu device侧的aclTensor。
 * 数据类型支持UINT8，数据格式支持ND。元素个数为align(gradY元素个数,128)/8。
 * @param [in] scale: 输出缩放因子，数据类型支持DOUBLE，来自前向1/(1-p)。
 * @param [in] gradX: npu device侧的aclTensor。
 * 数据类型支持FLOAT、FLOAT16、BFLOAT16，数据类型需为gradY可转换类型，数据格式支持ND，shape与gradY一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDropoutV3GradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* mask, double scale,
                                                         aclTensor* gradX, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnDropoutV3Grad的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnDropoutV3GradGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDropoutV3Grad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DROPOUT_V3_GRAD_H_
