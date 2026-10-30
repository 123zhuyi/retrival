/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OP_API_INC_SEGSUM_H_
#define OP_API_INC_SEGSUM_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnExpSegsum的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。
 * 计算公式：如下所示（以4D输入为例）
 * 1. 输入self由（N1,N2,N3,N4）升维成（N1,N2,N3,N4,1）
 * 2. 进行广播得到（N1,N2,N3,N4,N4）
 * 3. 生成（N4,N4）类型为bool的三角矩阵A，上三角为True，下三角为False，对角线为True。
 * 4. 用0填充输入self里面与矩阵A中值为Ture的位置相对应的元素。
 * $$
 * self_i=
 * \begin{cases}self_i,\quad A_i==False
 * \\0, \quad A_i==True
 * \end{cases}
 * $$
 * 5. 以self的倒数第二维进行cumsum累加。从维度视角来看的某个元素（其它维度下标不变，当前维度下标依次递增），
 * $selfTemp\_{i}$是输出张量中对应位置的元素。
 * $$
 * selfTemp_{i} = self_{1} + self_{2} + self_{3} + ...... + self_{i}
 * $$
 * 6. 生成（N4,N4）类型为bool的三角矩阵B，上三角为True，下三角为False，对角线为False。
 * 7. 用-inf填充selfTemp里面与矩阵B中值为Ture的位置相对应的元素。
 * $$
 * out_i=
 * \begin{cases}selfTemp_i,\quad B_i==False
 * \\-inf, \quad BD_i==True
 * \end{cases}
 * $$
 * 8. 计算selfTemp里面每个元素的指数。
 * $$
 * out_i=e^{selfTemp_i}
 * $$
 *
 * @param [in] self: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 支持非连续的Tensor。
 * @param [out] out: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，
 * 且数据类型与self的数据类型一致。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor);

/**
 * @brief aclnnExpSegsum的第二段接口，用于执行计算。
 *
 * 算子功能：进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnExpSegsumGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpSegsum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SEGSUM_H_