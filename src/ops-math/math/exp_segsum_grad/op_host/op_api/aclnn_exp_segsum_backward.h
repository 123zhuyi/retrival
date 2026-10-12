/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef OP_API_INC_EXP_SEGSUM_BACKWARD_H_
#define OP_API_INC_EXP_SEGSUM_BACKWARD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnExpSegsumBackward的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 *
 * 算子功能：aclnnExpSegsum的反向计算。
 * 计算公式：如下所示（以5D输入为例）
 * 1. 输入gradOutput(N1,N2,N3,N4,N4)与输入gradSelf(正向的输出)相乘。
 * $$
 * out\_mul = gradOutput * gradSelf
 * $$
 * 2. 生成（N4,N4）类型为bool的三角矩阵A，上三角为True，下三角为False，对角线为True。
 * 3. 用0填充输入$out\_mul$里面与矩阵A中值为Ture的位置相对应的元素。
 * $$
 * out\_mul_i=
 * \begin{cases}out\_mul_i,\quad A_i==False
 * \\0, \quad A_i==True
 * \end{cases}
 * $$
 * 4.对out_mul的倒数第二维进行倒序生成out_flip。
 * 5. 以$out\_flip$的倒数第二维进行cumsum累加。从维度视角来看的某个元素（其它维度下标不变，当前维度下标依次递增），$out\_cumsum\_{i}$是输出张量中对应位置的元素。
 * $$
 * out\_cumsum_{i} = out\_flip_{1} + out\_flip_{2} + out\_flip_{3} + ...... + out\_flip_{i}
 * $$
 * 6. 对$out\_cumsum$的-2维进行倒序生成out_flip2。
 * 7. 生成（N4,N4）类型为bool的三角矩阵B，上三角为True，下三角为False，对角线为True。
 * 8. 用0填充$out\_flip2$里面与矩阵B中值为Ture的位置相对应的元素。
 * $$
 * out\_flip2_i=
 * \begin{cases}out\_flip2_i,\quad B_i==False
 * \\0, \quad B_i==True
 * \end{cases}
 * $$
 * 9. 返回out\_flip2最后一维每行的和。
 *
 * @param [in] gradOutput: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 支持非连续的Tensor，支持ND格式。
 * @param [in] gradSelf: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16。
 * 数据类型与gradOutput一致。支持非连续的Tensor，支持ND格式。
 * @param [out] gradInput: npu device侧的aclTensor，数据类型支持FLOAT、BFLOAT16、FLOAT16，
 * 且数据类型与self的数据类型一致，支持ND格式。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(const aclTensor* gradOutput,const aclTensor* gradSelf,
                                                             aclTensor* gradInput, uint64_t* workspaceSize,
                                                             aclOpExecutor** executor);

 /**
 * @brief aclnnExpSegsumBackward的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，
 * 由第一段接口aclnnExpSegsumBackwardGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL Stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnExpSegsumBackward(void* workspace, uint64_t workspaceSize,
                                             aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_EXP_SEGSUM_BACKWARD_H_