/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LTTENSOR_H_
#define OP_API_INC_LTTENSOR_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnLtTensorGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Tensor中的元素，返回一个Bool类型的Tensor，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ? [True]
 * : [False] $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持float,int32,int8,uint8,int64,int16,double,kHalf,kBool数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持float,int32,int8,uint8,int64,int16,double,kHalf,kBool数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND
 * @param [in] out: npu device侧的aclTensor，输出一个数据类型为BOOL类型的Tensor，数据格式支持ND。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnLtTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnLtTensor的第二段接口，用于执行计算。
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Tensor中的元素，返回一个Bool类型的Tensor，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ? [True]
 * : [False] $$
 *
 * 实现说明：api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnAddGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnLtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

/**
 * @brief aclnnInplaceLtTensorGetWorkspaceSize的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Tensor中的元素，返回计算后的selfRef，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ?  [True] :
 * [False] $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] self: npu device侧的aclTensor，
 * 数据类型支持float,int32,int8,uint8,int64,int16,double,kHalf,kBool数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND。
 * @param [in] other: npu device侧的aclTensor，
 * 数据类型支持float,int32,int8,uint8,int64,int16,double,kHalf,kBool数据类型，
 * shape需要与other满足broadcast关系，支持非连续的Tensor，数据格式支持ND
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnInplaceLtTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnInplaceLtTensor的第二段接口，用于执行计算。
 *
 * 算子功能：计算self Tensor中的元素是否小于(<)other
 * Tensor中的元素，返回一个Bool类型的Tensor，self<other的为True，否则为False 计算公式：$$ out = (self < other)  ? [True]
 * : [False] $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 * A[(Self)] -->B([Contiguous])
 * B-->C1([Cast])-->D([Less])
 * E[(other)] -->F([Contiguous])
 * F --> C2([Cast])-->D
 * D -->F1([ViewCopy])--> J[(out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnInplaceLtTensorGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnInplaceLtTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LTTENSOR_H_
