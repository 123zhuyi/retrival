/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_NORMAL_H_
#define OP_API_INC_NORMAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnInplaceNormal的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点、复数数据类型，且数据类型需要与other构成互相推导关系，shape需要与other满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW，且数据格式需要与other一致。
 * @param [in] other: npu
 * device侧的aclTensor，数据类型支持整型，浮点、复数数据类型，且数据类型需要与self构成互相推导关系，shape需要与self满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW，且数据格式需要与self一致。
 * @param [in] alpha: host侧的aclScalar，数据类型需要可转换成self与other推导后的数据类型。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNormalGetWorkspaceSize(
    const aclTensor* selfRef, float mean, float std, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceNormal的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceNormal(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceNormalTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：生成均值mean和标准差std的离散正态分布的随机数，并将其填充到self张量中
 * @param [in] selfRef: npu device侧的aclTensor，
 * 数据类型支持FLOAT、INT32、INT64、FLOAT16、BFLOAT16、INT16、INT8、UINT8、BOOL、DOUBLE。数据格式支持ND。支持非连续的Tensor。
 * @param [in] mean: host侧的浮点型，进行离散均匀分布取值的左边界，输入为FLOAT数据类型。
 * @param [in] std: host侧的浮点型，进行离散均匀分布取值的右边界，输入为FLOAT数据类型。
 * @param [in] seedTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的种子，它影响生成的随机数序列。
 * @param [in] offsetTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
 * @param [in] offset: host侧的整型，随机数生成器的图内偏移量，它影响生成的随机数序列的位置。输入为INT64_T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(
    const aclTensor* selfRef, float mean, float std, const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceNormalTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceNormalTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceNormalTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_NORMAL_H_
