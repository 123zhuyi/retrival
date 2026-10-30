/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_RIGHT_SHIFT_H_
#define OP_API_INC_LEVEL2_ACLNN_RIGHT_SHIFT_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnRightShift的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：输入张量input中每个元素，根据shiftBits对应位置的参数，按位进行右移。
 * 计算公式：
 * output_{i} = input_{i}>>shiftBits_{i}
 * @domain aclnn_math
 * 参数描述：
 * @param [in]   input
 * 输入Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与shiftBits构成互相推导关系，input需要与shiftBits满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [in]   shiftBits
 * 输入Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * 数据类型需要与input构成互相推导关系，input需要与shiftBits满足broadcast关系。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  out
 * 输出Tensor，数据类型支持 INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64。
 * shape需要与input和shiftBits做broadcast后的shape一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRightShiftGetWorkspaceSize(const aclTensor *input, const aclTensor *shiftBits,
                                                      aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnRightShift的第二段接口，用于执行计算。
 * 功能描述：输入张量input中每个元素，根据shiftBits对应位置的参数，按位进行右移。
 * 计算公式：
 * output_{i} = input_{i}>>shiftBits_{i}
 * @domain aclnn_math
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * flowchart LR
 *    input[(input)]-->l0op::Contiguous_input([l0op::Contiguous])
 *    l0op::Contiguous_input([l0op::Contiguous])-->l0op::Cast_input([l0op::Cast])
 *    l0op::Cast_input([l0op::Cast])-->l0op::rightshift([l0op::rightshift])
 *    shiftBits[(shiftBits)]-->l0op::Contiguous_shiftBits([l0op::Contiguous])
 *    -->l0op::Cast_shiftBits([l0op::Cast])-->l0op::rightshift([l0op::RightShift])
 *    l0op::rightshift([l0op::RightShift])-->l0op::Cast_out([l0op::Cast])
 *    l0op::Cast_out([l0op::Cast])-->l0op::ViewCopy_out([l0op::ViewCopy])-->out[(out)]
 * ```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRightShiftGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRightShift(void *workspace, uint64_t workspaceSize,
                                     aclOpExecutor *executor, aclrtStream stream);


#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_RIGHT_SHIFT_H_
