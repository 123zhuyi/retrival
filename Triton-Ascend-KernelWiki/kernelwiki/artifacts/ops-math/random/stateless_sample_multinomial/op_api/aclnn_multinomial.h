/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_
#define OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMultinomial的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 */
ACLNN_API aclnnStatus aclnnMultinomialGetWorkspaceSize(const aclTensor* self, int64_t numsamples, bool replacement,
                                                       int64_t seed, int64_t offset, aclTensor* out,
                                                       uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMultinomial的第二段接口，用于执行计算。
 */
ACLNN_API aclnnStatus aclnnMultinomial(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

/**
 * @brief aclnnMultinomialTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：在输入张量中根据每个对象分布的概率，抽取numsamples个样本，并将这些样本的索引存储在输出张量中。
 * @param [in] self: npu device侧的aclTensor，shape为(N, C)或(C)，self的取值范围需要大于等于0且self与out的维度一致。
 * 数据类型支持BFLOAT16、FLOAT16、FLOAT、DOUBLE。数据格式支持ND。支持非连续的Tensor。
 * @param [in] numsamples: host侧的整形，从每个多项分布中抽取的样本数。
 * numsamples为非负整数，当replacement为false时，numsamples不大于C。输入为INT64_t数据类型。
 * @param [in] replacement: host侧的布尔类型，决定了抽样时元素是否有放回。输入为BOOL数据类型。
 * @param [in] seedTensor: npu device侧的aclTensor，数据类型支持INT64_t。数据格式支持ND。
 * 随机数生成器的种子,它影响生成的随机数序列。
 * @param [in] offsetTensor: npu device侧的aclTensor，数据类型支持INT64_t。数据格式支持ND。
 * 随机数生成器的偏移量，它影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。
 * @param [in] offset: host侧的整型，随机数生成器的偏移量，它影响生成的随机数序列的位置。输入为INT64_T数据类型。
 * @param [in] out: npu device侧的aclTensor，数据类型支持INT64。shape为(N, numsamples)或(numsamples)，self与out的维度一致。
 * 支持非连续的Tensor，数据格式支持ND。
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor* self, int64_t numsamples, bool replacement,
                                                       const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
                                                       aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMultinomialTensor的第二段接口，用于执行计算。
 *
 * 算子功能：在输入张量中根据每个对象分布的概率，抽取numsamples个样本，并将这些样本的索引存储在输出张量中。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMultinomialTensorGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnMultinomialTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_LEVEL2_ACLNN_MULTINOMIAL_H_
