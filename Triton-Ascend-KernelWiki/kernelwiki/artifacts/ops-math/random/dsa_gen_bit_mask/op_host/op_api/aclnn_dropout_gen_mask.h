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
 * \file aclnn_dropout_gen_mask.h
 * \brief
 */
#ifndef OP_API_INC_DROPOUT_GEN_MASK_H_
#define OP_API_INC_DROPOUT_GEN_MASK_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDropoutGenMask的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnDropoutGenMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnDropoutGenMaskV2的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclDataType probDataType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus
aclnnDropoutGenMaskV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnDropoutGenMaskV2Tensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：训练过程中，按照概率prob生成mask，用于元素置零。
 * @param [in] shape: npu device侧的aclIntArray，
 * 数据类型支持INT32或者INT64。数据格式支持ND。支持非连续的Tensor。
 * @param [in] prob: host侧的浮点型，置零的概率，输入为DOUBLE数据类型。
 * @param [in] seedTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的种子，它影响生成的随机数序列。
 * @param [in] offsetTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
 * @param [in] offset: host侧的整型，随机数生成器的偏移量，它影响生成的随机数序列的位置。输入为INT64_T数据类型。
 * @param [in] probDataType: Host侧的数据类型枚举，表示prob的数据类型。数据类型支持FLOAT、FLOAT16、BFLOAT16。
 * @param [out] out: 运算结果，npu device侧的aclTensor，bit类型并使用UINT8类型存储的mask数据。
 * 数据类型支持UINT8，shape需要为(align(input的元素个数,128)/8)。数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(
    const aclIntArray* shape, double prob, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    aclDataType probDataType, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDropoutGenMaskV2Tensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnDropoutGenMaskV2TensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnDropoutGenMaskV2Tensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DROPOUT_GEN_MASK_H_
