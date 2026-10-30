/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
#define OP_API_INC_LEVEL2_ACLNN_RANDOM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceRandom的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、INT32、INT64、BFLOAT16、FLOAT16、
 * INT16、INT8、UINT8、BOOL、DOUBLE。数据格式支持ND。支持非连续的Tensor。
 * @param [in] from: host侧的浮点型，进行离散均匀分布取值的左边界，输入为DOUBLE数据类型。
 * @param [in] to: host侧的浮点型，进行离散均匀分布取值的右边界，输入为DOUBLE数据类型。
 * @param [in] seed: 随机数生成器的种子,它影响生成的随机数序列，输入为UINT64_T数据类型。
 * @param [in] offset: 随机数生成器的偏移量,它影响生成的随机数序列的位置。设置偏移量后，
 * 生成的随机数序列会从指定位置开始。输入为UINT64_T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRandomGetWorkspaceSize(
    const aclTensor* selfRef, int64_t from, int64_t to, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRandom的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandom获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus
aclnnInplaceRandom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceRandomTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT、INT32、INT64、BFLOAT16、FLOAT16、
 * INT16、INT8、UINT8、BOOL、DOUBLE。数据格式支持ND。支持非连续的Tensor。
 * @param [in] from: host侧的浮点型，进行离散均匀分布取值的左边界，输入为DOUBLE数据类型。
 * @param [in] to: host侧的浮点型，进行离散均匀分布取值的右边界，输入为DOUBLE数据类型。
 * @param [in] seedTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的种子,它影响生成的随机数序列。
 * @param [in] offsetTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
 * @param [in] offset: host侧的整型，随机数生成器的偏移量。输入为INT64_T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(
    const aclTensor* selfRef, int64_t from, int64_t to, const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRandomTensor的第二段接口，用于执行计算。
 *
 * 算子功能：从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandom获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus
aclnnInplaceRandomTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceRandomWithoutFromTo的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、INT32、INT64、BFLOAT16、FLOAT16、
 * INT16、INT8、UINT8、BOOL。数据格式支持ND。支持非连续的Tensor。
 * @param [in] seed: 随机数生成器的种子,它影响生成的随机数序列，输入为INT64数据类型。
 * @param [in] offset: 随机数生成器的偏移量,它影响生成的随机数序列的位置。设置偏移量后，
 * 生成的随机数序列会从指定位置开始。输入为INT64T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRandomWithoutFromToGetWorkspaceSize(
    const aclTensor* selfRef, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRandomWithoutFromTo的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandomWithoutFromToGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus
aclnnInplaceRandomWithoutFromTo(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceRandomWithoutFromToTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持FLOAT、INT32、INT64、BFLOAT16、FLOAT16、
 * INT16、INT8、UINT8、BOOL。数据格式支持ND。支持非连续的Tensor。
 * @param [in] seedTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的种子,它影响生成的随机数序列。
 * @param [in] offsetTensor: npu device侧的aclTensor，数据类型支持INT64。数据格式支持ND。
 * 随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
 * @param [in] offset: host侧的整型，随机数生成器的偏移量。输入为INT64_T数据类型。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceRandomWithoutFromToTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* seedTensor, const aclTensor* offsetTensor,
    int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceRandomWithoutFromToTensor的第二段接口，用于执行计算。
 *
 * 算子功能：从[from, to-1]的离散均匀分布中采样的数填充selfRef张量。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandomWithoutFromToTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */

ACLNN_API aclnnStatus
aclnnInplaceRandomWithoutFromToTensor(void* workspace, uint64_t workspaceSize,
                                      aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_RANDOM_H_
