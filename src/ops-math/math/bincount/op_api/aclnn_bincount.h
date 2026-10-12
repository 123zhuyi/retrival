/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_BINCOUNT_H_
#define OP_API_INC_BINCOUNT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBincount 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算非负整数数组中每个数的频率。
 * 计算公式：
 * 如果n是self在位置i上的值，如果指定了weights，则
 * out[n] = out[n] + weights[i]
 * 否则
 * out[n] = out[n] + 1
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持INT8、INT16、INT32、INT64、UINT8，
 * 且必须是非负整数，数据格式支持1维ND。支持非连续的Tensor。
 * @param [in] weights:  npu device侧的aclTensor，self每个值的权重，可为空指针。
 * 数据类型支持FLOAT、FLOAT16、FLOAT64、INT8、INT16、INT32、INT64、UINT8、BOOL，
 * 数据格式支持1维ND，且shape必须与self一致。支持非连续的Tensor。
 * @param [in] minlength: host侧的int型，指定输出tensor最小长度。如果计算出来的size的最大值小于minlength，
 * 则输出长度为minlength，否则为size。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持INT32、INT64、FLOAT、DOUBLE。数据格式支持1维ND。支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBincountGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weights, int64_t minlength, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnBincount的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能：计算非负整数数组中每个数的频率。
 * 计算公式：
 * 如果n是self在位置i上的值，如果指定了weights，则
 * out[n] = out[n] + weights[i]
 * 否则
 * out[n] = out[n] + 1

 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnBincountGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BINCOUNT_H_