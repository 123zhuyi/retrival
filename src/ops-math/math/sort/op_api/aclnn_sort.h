/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_SORT_H_
#define OP_API_INC_SORT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSort的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能： 对输入Tensor完成sort操作
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8。支持空Tensor，
 * 支持[非连续的Tensor](#)。
 * @param [in] stable: 是否稳定排序, True为稳定排序，False为非稳定排序, 数据类型为BOOLEAN。
 * @param [in] dim: 用来作为排序标准的维度, 数据类型为INT。范围为 [-N, N-1]。
 * @param [in] descending: 控制排序顺序，True为降序，False为升序, 数据类型为BOOLEAN。
 * @param [in] valuesOut: npu device侧的aclTensor, 数据类型支持FLOAT16、FLOAT32、INT8、INT16、INT32、INT64、UINT8。支持
 * 空Tensor，支持[非连续的Tensor](#)。数据格式支持ND([参考](#))。shape和数据格式需要与self一致。
 * @param [in] indicesOut: npu device侧的aclTensor,
 * 数据类型支持INT64。支持空Tensor，支持[非连续的Tensor](#)。数据格式支持 ND([参考](#))。shape和数据格式需要与self一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor* self, bool stable, int64_t dim, bool descending,
                                                aclTensor* valuesOut, aclTensor* indicesOut, uint64_t* workspaceSize,
                                                aclOpExecutor** executor);

/**
 * @brief: aclnnSort的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成sort操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSortGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSort(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
