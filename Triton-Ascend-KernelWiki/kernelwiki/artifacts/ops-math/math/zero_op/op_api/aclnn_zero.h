/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_Zero_H_
#define OP_API_INC_Zero_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnInplaceZero的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：将self张量填充为全零。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(selfRef)]-->B([l0op::Contiguous])
 *     B-->C([l0op::Zeros_Like])
 *     C-->D([l0op::ViewCopy])
 *     D-->E[(selfRef)]
 * ```
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持INT8、INT16、INT32、INT64、QINT8、QINT16、QINT32、UINT8、UINT16、
 * UINT32、UINT64、QUINT8、QUINT16、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、
 * FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E1M2、FLOAT4_E2M1。
 * 其中，FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E1M2、FLOAT4_E2M1仅支持Ascend950及后续芯片。
 * 除FLOAT4_E1M2、FLOAT4_E2M1外，其他均支持[非连续的Tensor](#)，数据格式支持ND
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceZeroGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceZero的第二段接口，用于执行计算。
 *
 * 算子功能：将self张量填充为全零。
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(selfRef)]-->B([l0op::Contiguous])
 *     B-->C([l0op::Zeros_Like])
 *     C-->D([l0op::ViewCopy])
 *     D-->E[(selfRef)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceZeroGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceZero(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_Zero_H_
