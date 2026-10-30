/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_IM2COL_H_
#define OP_API_INC_IM2COL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIm2col的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：将滑动窗口的数据块展开成列
 * @param [in] self:
 * npu device侧的aclTensor，数据类型支持INT8、UINT8、INT16、UINT16、INT32、UINT32、
 * INT64、UINT64、BF16、FLOAT16、FLOAT、DOUBLE、BOOL、COMPLEX632、COMPLEX64。
 * 支持[非连续的Tensor](#)，数据格式支持NCHW。
 * 其中，Ascend950之前芯片仅支持数据类型BF16、FLOAT16、FLOAT。
 * @param [in] kernelSize: npu device侧的aclIntArray，数组长度必须为2。
 * @param [in] dilation: npu device侧的aclIntArray，数组长度必须为2。
 * @param [in] padding: npu device侧的aclIntArray，数组长度必须为2。
 * @param [in] stride: npu device侧的aclIntArray，数组长度必须为2。
 * @param [in] out:
 * npu device侧的aclTensor，数据类型支持INT8、UINT8、INT16、UINT16、INT32、UINT32、
 * INT64、UINT64、BF16、FLOAT16、FLOAT、DOUBLE、BOOL、COMPLEX632、COMPLEX64。
 * 支持[非连续的Tensor](#)，数据格式支持NCHW。
 * 其中，Ascend950之前芯片仅支持数据类型BF16、FLOAT16、FLOAT。
 * 
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIm2colGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIm2col的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnIm2col(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
