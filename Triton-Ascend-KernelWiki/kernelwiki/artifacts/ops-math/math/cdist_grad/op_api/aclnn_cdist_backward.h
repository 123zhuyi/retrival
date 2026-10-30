/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_cdist_backward.h
 * \brief
 */
#ifndef MATH_CDIST_GRAD_OP_API_ACLNN_CDIST_BACKWARD_H_
#define MATH_CDIST_GRAD_OP_API_ACLNN_CDIST_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnCdistBackward第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：实现aclnnCdist的反向。
 *
 * @param [in] grad: npu device侧的aclTensor，数据类型支持float32 float16，维度在2D-8D。
 * @param [in] x1: npu device侧的aclTensor，数据类型支持float32 float16，维度与grad相等。
 * @param [in] x2: npu device侧的aclTensor，数据类型支持float32 float16，维度与grad相等。
 * @param [in] cdist: npu device侧的aclTensor，数据类型支持float32 float16，维度与grad相等。
 * @param [in] p: int64类型。
 * @param [out] out: npu device侧的aclTensor，数据类型支持float32 float16，维度与grad相等。
 * @param [out] workspaceSize: 返回用户需要在npu侧申请的workspace大小。
 * @param [out] executor: 返回op执行器。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnCdistBackwardGetWorkspaceSize(
    const aclTensor* grad, const aclTensor* x1, const aclTensor* x2, const aclTensor* cdist, float p,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnCdistBackward的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnCdistBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnCdistBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // MATH_CDIST_GRAD_OP_API_ACLNN_CDIST_BACKWARD_H_
