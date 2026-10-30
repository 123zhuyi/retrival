/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SPLIT_TENSOR_H_
#define OP_API_INC_LEVEL2_ACLNN_SPLIT_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSplitTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、UINT32、INT64、UINT64、INT16、
 * UINT16、INT8、UINT8、BOOL、BFLOAT16、COMPLEX128和COMPLEX64。支持非连续的Tensor，数据格式支持ND。
 * @param [in] splitSections: host侧的整型数值, 数据类型为UINT64。表示沿dim轴均匀切分后的块大小,
 * 最后一块可以小于splitSections。
 * @param [in] dim: host侧的整型数值，数据类型为INT64，表示输入tensor被split的维度。
 * @param [in] out: npu device侧的aclTensorList，表示被split后的输出tensor的列表，数据类型支持FLOAT、FLOAT16、DOUBLE、
 * INT32、UINT32、INT64、UINT64、INT16、UINT16、INT8、UINT8、BOOL、BFLOAT16、COMPLEX128和COMPLEX64。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSplitTensorGetWorkspaceSize(const aclTensor* self, uint64_t splitSections, int64_t dim,
                                                       aclTensorList* out, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor);

/**
 * @brief aclnnSplitTensor的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSplitTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSplitTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SPLIT_TENSOR_H_
