/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_SVD_H_
#define OP_API_INC_LEVEL2_ACLNN_SVD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief aclnnSvd的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：对输入张量input进行奇异值分解，得到左奇异矩阵u，奇异值向量sigma，右奇异矩阵v。
 * 计算公式：
 * input = u × diag(sigma) x v_T
 * \mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
 * @domain aclnn_math
 * @param [in]   input
 * 输入Tensor，数据类型支持 DOUBLE, FLOAT, COMPLEX64, COMPLEX128。
 * 数据类型需要与sigma、u、v保持一致，支持非连续的Tensor，数据格式支持ND。
 * @param [in]   fullMatrices
 * 输入参数，是否完整计算矩阵u和v。当设为true时输出完整的u、v。设为false时只计算经济版本以节省内存和计算资源。
 * fullMatrices为true或false时，input与sigma、u、v的shape对应关系如下：
 * input:[..., m, n]，k = min(m, n)
 * fullMatrices == true --> u:[..., m, m]，(sigma):[..., k]，v:[..., n, n]，diag(sigma):[..., m, n]
 * fullMatrices == false --> u:[..., m, k]，(sigma):[..., k]，v:[..., n, k]，diag(sigma):[..., k, k]
 * @param [in]   computeUV
 * 输入参数，是否计算矩阵u和v。computeUV设为false时，只计算sigma，输出张量u、v将填充0。
 * @param [out]  sigma
 * 输出Tensor，奇异值向量。支持非连续的Tensor，数据格式支持ND，数据类型需要与input保持一致。
 * shape需要根据input的shape及fullMatrices参数进行推导。
 * @param [out]  u
 * 输出Tensor，左奇异矩阵。支持非连续的Tensor，数据格式支持ND，数据类型需要与input保持一致。
 * shape需要根据input的shape及fullMatrices参数进行推导。
 * @param [out]  v
 * 输出Tensor，右奇异矩阵。支持非连续的Tensor，数据格式支持ND，数据类型需要与input保持一致。
 * shape需要根据input的shape及fullMatrices参数进行推导。
 * @param [out]  workspaceSize 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSvdGetWorkspaceSize(const aclTensor *input, const bool fullMatrices, const bool computeUV,
                                               aclTensor *sigma, aclTensor *u, aclTensor *v, uint64_t *workspaceSize,
                                               aclOpExecutor **executor);

/**
 * @brief aclnnSvd的第二段接口，用于执行计算。
 * 功能描述：对输入张量input进行奇异值分解，得到左奇异矩阵u，奇异值向量sigma，右奇异矩阵v。
 * 计算公式：
 * input = u × diag(sigma) x v_T
 * \mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
 * @domain aclnn_math
 * 实现说明：
 * flowchart LR
 *  input[(input)]-->l0op::Contiguous_input([l0op::Contiguous])
 *  l0op::Contiguous_input([l0op::Contiguous])-->l0op::Svd([l0op::Svd])
 *  l0op::Svd([l0op::Svd])-->l0op::ViewCopy_u([l0op::ViewCopy])-->u[(u)]
 *  l0op::Svd([l0op::Svd])-->l0op::ViewCopy_sigma([l0op::ViewCopy])-->sigma[(sigma)]
 *  l0op::Svd([l0op::Svd])-->l0op::ViewCopy_v([l0op::ViewCopy])-->v[(v)]
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnSvdGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnSvd(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_SVD_H_