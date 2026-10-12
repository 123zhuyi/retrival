/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ISCLOSE_H_
#define OP_API_INC_LEVEL2_ISCLOSE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnIsClose的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：返回一个带有布尔元素的新张量，判断self和other在epsilon内是否相等。
 * 计算公式
 * $$ \left | input-other\right | = atol + rtol\times \left | other \right | $$
 *
 * @param [in] self: npu
 * device侧的aclTensor，维度支持1~8维，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、DOUBLE，
 * 支持非连续的Tensor，dtype与other的dtype必须一致，shape需要与other满足broadcast关系。数据格式支持ND。
 * @param [in] other: npu
 *  device侧的aclTensor，维度支持1~8维，数据类型支持FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、BOOL、DOUBLE，
 * 支持非连续的Tensor，dtype与self的dtype必须一致，shape需要与self满足broadcast关系。数据格式支持ND。
 * @param [in] rtol: 绝对宽容。数据类型支持DOUBLE。
 * @param [in] atol: 相对公差。数据类型支持DOUBLE。
 * @param [in] equal_nan:
 * NaN值比较选项。如果为True，则两个NaN将被视为相等；如果为False，则两个NaN将被视为不相等。数据类型支持BOOL。
 * @param [out] out: npu
 * device侧的aclTensor，维度支持1~8维，数据类型支持BOOL，支持非连续的Tensor，shape需要是self与other
 * broadcast之后的shape，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnIsCloseGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, double rtol, double atol, bool equal_nan, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnIsClose的第二段接口，用于执行计算。
 *
 * 算子功能：返回一个带有布尔元素的新张量，判断self和other在epsilon内是否相等。
 * 计算公式
 * $$ \left | input-other\right | = atol + rtol\times \left | other \right | $$
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnIsCloseGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnIsClose(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ISCLOSE_H_
