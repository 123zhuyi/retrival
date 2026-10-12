/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_ADDN_H_
#define OP_API_INC_LEVEL2_ACLNN_ADDN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnAddN的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入tensors进行主元素相加求和操作。
 * 计算公式：
 * out = tensors_{1} + tensors_{2}+ \dots + tensors_{n}
 * @domain aclnn_math
 * 参数描述：
 * @param [in]   tensors
 * 输入TensorList，数据类型支持 INT32，INT64，FLOAT16，BFLOAT16，FLOAT32。
 * tensors中的tensor需要满足shape一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  out
 * 输出Tensor，数据类型支持 INT32，INT64，FLOAT16，BFLOAT16，FLOAT32。
 * shape需要与tensors中的tensor的shape一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *workspaceSize, 
                                                aclOpExecutor **executor);

/**
 * @brief aclnnAddN的第二段接口，用于执行计算。
 * 功能描述：对输入tensors进行主元素相加求和操作。
 * 计算公式：
 * out = tensors_{1} + tensors_{2}+ \dots + tensors_{n}
 * @domain aclnn_math
 * 实现说明：
 * api计算的基本路径：
```mermaid
flowchart LR
   A[(tensors)]-->B([l0op::Contiguous])-->C([l0op::AddN])
   -->D([l0op::ViewCopy])-->E[(out)]
```
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnAddNGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnAddN(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_ADDN_H_