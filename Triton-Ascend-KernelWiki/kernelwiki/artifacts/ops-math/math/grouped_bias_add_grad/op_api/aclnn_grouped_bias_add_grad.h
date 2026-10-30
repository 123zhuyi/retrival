/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_BIAS_ADD_GRAD_H_
#define OP_API_INC_GROUPED_BIAS_ADD_GRAD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedBiasAddGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：实现groupBiasAdd的反向计算。
 * @param [in] gradY: 反向传播梯度，公式中的gradY，Device侧的aclTensor，数据类型支持FLOAT,FLOAT16,BFLOAT16。
 *                    有可选输入groupIdxOptional时，shape仅支持2维，无可选输入groupIdxOptional时，shape仅支持3维。
 * @param [in] groupIdxOptional:
 * 每个分组结束位置，公式中的groupIdxOptional，Device侧的aclTensor，数据类型支持INT32，INT64，shape仅支持1维。
 * @param [out] out:
 * bias的梯度，公式中的out，Device侧的aclTensor，数据类型支持FLOAT,FLOAT16,BFLOAT16，数据类型必须与gradY的数据类型一致，shape仅支持2维。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnGroupedBiasAddGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedBiasAddGradGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGroupedBiasAddGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnGroupedBiasAddGrad的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train
 * 算子功能：实现groupBiasAdd的反向计算。
 * @param [in] gradY: 反向传播梯度，公式中的gradY，Device侧的aclTensor，数据类型支持FLOAT,FLOAT16,BFLOAT16。
 *                    有可选输入groupIdxOptional时，shape仅支持2维，无可选输入groupIdxOptional时，shape仅支持3维。
 * @param [in] groupIdxOptional:
 * 每个分组结束位置，公式中的groupIdxOptional，Device侧的aclTensor，数据类型支持INT32，INT64，shape仅支持1维。
 * @param [in] groupIdxType: 表示groupIdx的类型。支持的值为0和1，
 * 0：表示groupIdxOptional中的值为每个group的结束索引，1：表示groupIdxOptional中的值为每个group的大小。数据类型支持Int64。
 * @param [out] out:
 * bias的梯度，公式中的out，Device侧的aclTensor，数据类型支持FLOAT,FLOAT16,BFLOAT16，数据类型必须与gradY的数据类型一致，shape仅支持2维。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedBiasAddGrad的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnGroupedBiasAddGradV2GetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnGroupedBiasAddGradV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
