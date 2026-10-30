/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_
#define OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSearchSorted的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 在一个已排序的张量（sortedSequence）中查找给定值（self）应该插入的位置。
 * @param [in] sortedSequence: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] self: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] outInt32: host侧的BOOL类型，指定输出Tensor是否维INT32类型。
 * @param [in] right: host侧的BOOL类型，如果找到的是相同值，是否返回右侧位置。
 * @param [in] sorter: npu device侧的aclTensor, 数据类型支持INT64, 指定sortedSequence中元素的顺序。
 * @param [in] out: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor* sortedSequence, const aclTensor* self,
                                                        bool outInt32, bool right, const aclTensor* sorter,
                                                        aclTensor* out, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);

/**
 * @brief aclnnSearchSorted的第一段接口，根据具体的计算流程，计算workspace大小。
 *
 * 算子功能： 在一个已排序的张量（sortedSequence）中查找给定值（self）应该插入的位置。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSearchSortedGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSearchSorted(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

/**
 * @brief aclnnSearchSorteds的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 在一个已排序的张量（sortedSequence）中查找给定值（self）应该插入的位置。
 * @param [in] sortedSequence: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in] self: npu device侧的aclScalar, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64。
 * @param [in] outInt32: host侧的BOOL类型，指定输出Tensor是否维INT32类型。
 * @param [in] right: host侧的BOOL类型，如果找到的是相同值，是否返回右侧位置。
 * @param [in] sorter: npu device侧的aclTensor, 数据类型支持INT64, 指定sortedSequence中元素的顺序。
 * @param [in] out: npu device侧的aclTensor, 数据类型支持FLOAT16, FLOAT32, INT32, INT8, UINT8, INT16, INT64,
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor* sortedSequence, const aclScalar* self,
                                                         const bool outInt32, const bool right, const aclTensor* sorter,
                                                         aclTensor* out, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnSearchSorteds的第二段接口，用于执行计算。
 *
 * 算子功能： 在一个已排序的张量（sortedSequence）中查找给定值（self）应该插入的位置。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSearchSortedsGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSearchSorteds(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_SEARCHSORTED_H_
