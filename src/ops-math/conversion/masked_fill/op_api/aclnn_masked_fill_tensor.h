/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_MASKEF_FILL_TENSOR_H_
#define OP_API_INC_MASKEF_FILL_TENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceMaskedFillTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A1[(mask)] -->B1([Contiguous])
 *  B1  -->C1([Cast])
 *  C1  -->D
 *
 *  A2[(value)]-->B2[(Cast)]
 *  B2-->D
 * ```
 *
 * @param [in] selfRef: npu device侧的aclTensor，数据类型支持BOOL、UINT8、INT8、INT16、INT32、INT64、FLOAT、
 *                      FLOAT16、BFLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
 *                      支持非连续的Tensor，数据格式支持ND。
 * @param [in] mask: npu device侧的aclTensor，数据类型支持BOOL。且shape与selfRef满足broadcast关系。数据格式支持ND。
 * @param [in] value: npu device侧的aclTensor。且shape与selfRef满足broadcast关系。数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* value, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnInplaceMaskedFillTensor的第二段接口，用于执行计算。
 *
 * 算子功能：用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *  A[(selfRef)] -->B([Contiguous])
 *  B  -->C([Unsqueeze])
 *  C  -->D([MaskedFill])
 *  D  -->I([Squeeze])
 *  I   --> J([ViewCopy])
 *  J   --> K[(out)]
 *
 *  A2[(mask)] -->B2([Contiguous])
 *  B2 --> C2([Cast])
 *  C2  -->D
 *
 *  A1[(value)]-->B1([Contiguous])
 *  B1-->C1([Cast])
 *  C1-->D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu
 * device侧申请的workspace大小，由第一段接口aclnnInplaceMaskedFillTensorGetWorkspaceSize。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceMaskedFillTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MASKEF_FILL_TENSOR_H_