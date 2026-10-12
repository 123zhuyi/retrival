/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_BERNOULLI_H_
#define OP_API_INC_BERNOULLI_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBernoulli的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(self_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] prob: host侧的aclScalar，浮点类型，需要满足$ 0≤p≤1 $
 * @param [in] seed: host侧的aclScalar
 * @param [in] offset: host侧的aclScalar
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBernoulliGetWorkspaceSize(
    const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnBernoulli的第二段接口，用于执行计算。
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnBernoulliTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(self_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] self: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] prob: npu
 * device侧的aclTensor，数据类型支持浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] seed: host侧的aclScalar
 * @param [in] offset: host侧的aclScalar
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnBernoulliTensor的第二段接口，用于执行计算。
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Out)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceBernoulli的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(selfRef_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] prob: host侧的aclScalar，浮点类型，需要满足$ 0≤p≤1 $
 * @param [in] seed: host侧的aclScalar
 * @param [in] offset: host侧的aclScalar
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBernoulli的第二段接口，用于执行计算。
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceBernoulliTensor的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_rand
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(selfRef_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] selfRef: npu
 * device侧的aclTensor，数据类型支持整型，浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] prob: npu
 * device侧的aclTensor，数据类型支持浮点类型，支持非连续的Tensor，数据格式支持ND
 * @param [in] seed: host侧的aclScalar
 * @param [in] offset: host侧的aclScalar
 * @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnInplaceBernoulliTensor的第二段接口，用于执行计算。
 *
 * 算子功能：从伯努利分布中提取二进制随机数
 * 计算公式：
 * $$ out_i∼Bernoulli(input_i) $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)]  --> B([l0::Contiguous]) -->D([l0op::StatelessBernoulli]) --> I([l0op::ViewCopy]) --> J[(Self)]
 * K((p)) --> K0([ConvertToTensor]) --> D
 * E((seed)) --> D
 * F((offset)) --> D
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BERNOULLI_H_
