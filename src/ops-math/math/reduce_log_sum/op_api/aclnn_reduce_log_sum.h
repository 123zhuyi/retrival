/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_REDUCE_LOG_SUM_H_
#define OP_API_INC_REDUCE_LOG_SUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnReduceLogSum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 * * 算子功能：使用输入边界的反射填充输入tensor。
 * @param [in] data:
 * 表示参与计算的目标张量，维度小于8维，Device侧的aclTensor，支持非连续的Tensor，
 * 数据类型支持FLOAT16、FLOAT32，数据格式支持ND。
 * @param [in] axes: 指定计算维度，Host侧的aclIntArray，数据类型支持INT64，取值范围为[-self.dim(), self.dim()-1]。
 * @param [in] keepDims: 指定是否在输出张量中保留输入张量的维度，Host侧的BOOL值。
 * @param [in] noopWithEmptyAxes:
 * 指定axes为空时的行为：false即对所有轴进行计算；true即不进行计算，输出张量等于输入张量，Host侧的BOOL值。
 * @param [in] reduce:
 * 表示计算后的结果，维度小于8维，Device侧的aclTensor，支持非连续的Tensor，
 * 数据类型支持FLOAT16、FLOAT32，需与data一致，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor* data, const aclIntArray* axes, bool keepDims,
                                                        bool noopWithEmptyAxes, aclTensor* reduce,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnReduceLogSum的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceRandom获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnReduceLogSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_REDUCE_LOG_SUM_H_
