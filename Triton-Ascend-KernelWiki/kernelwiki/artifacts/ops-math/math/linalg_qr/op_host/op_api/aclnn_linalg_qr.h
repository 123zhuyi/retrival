/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef OP_API_INC_LINALG_QR_H_
#define OP_API_INC_LINALG_QR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLinalgQr的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能： 对输入Tensor完成QR操作
 * @param [in] self: 计算输入, 数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128, 数据格式支持ND,
 * 支持非连续的Tensor
 * @param [in] mode: 计算输入, 指定输出的tensor shape是否为complete, 数据类型支持int64_t
 * @param [out] Q: 计算输出,
 * Qr分解后得到的正交矩阵，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128，且与self相同 数据格式支持ND,
 * 支持非连续的Tensor。
 * @param [out] R: 计算输出,
 * Qr分解后得到的上三角矩阵，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128，且与self相同 数据格式支持ND,
 * 支持非连续的Tensor。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor* self, int64_t mode, aclTensor* Q, aclTensor* R,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief: aclnnLinalgQr的第二段接口，用于执行计算
 *
 * 算子功能： 对输入Tensor完成Qr 操作
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnLinalgQrGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLinalgQr(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LINALG_QR_H_