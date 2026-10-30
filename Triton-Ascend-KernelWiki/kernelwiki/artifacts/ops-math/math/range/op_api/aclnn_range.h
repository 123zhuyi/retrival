/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_RANGE_H_
#define OP_API_INC_RANGE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnRange的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_math
 *
 * 功能描述：从start起始到end结束按照step的间隔取值，并返回大小为 $ \lfloor \frac{end - start} {step} \rfloor + 1
 * $的1维张量。其中，步长step是张量中 相邻两个值的间隔。
 *
 * 计算公式：$$ out_{i+1}=out_i+step $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Start)]--> E([l0op::Arange])
 *     B[(End)]--> D[(Limit = End + Step)]
 *     C[(Step)]--> D
 *     D--> E
 *     C--> E
 *     E--> H([l0op::Cast])
 *     H--> M([l0op::ViewCopy])
 *     M--> N[(Out)]
 * ```
 *
 * 参数描述：
 * @param [in]   start
 * 获取值的范围的起始位置：host侧的aclScalar，数据类型支持整型，浮点数据类型。数据格式支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。
 * @param [in]   end
 * 获取值的范围的结束位置：host侧的aclScalar，数据类型支持整型，浮点数据类型。数据格式支持ND。需要满足在step大于0时输入的start小于end，或者step小于0时输入的start大于end。
 * @param [in]   step
 * 获取值的步长：host侧的aclScalar，数据类型支持整型，浮点数据类型。数据格式支持ND。需要满足step不等于0。
 * @param [in]   out              指定的输出tensor：npu
 * device侧的aclTensor，数据类型支持整型，浮点数据类型，数据格式支持ND。
 * @param [out]  workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out]  executor: 返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRangeGetWorkspaceSize(
    const aclScalar* start, const aclScalar* end, const aclScalar* step, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);
/**
 * @brief aclnnRange的第二段接口，用于执行计算。
 *
 *
 * 功能描述：从start起始到end结束按照step的间隔取值，并返回大小为 $ \lfloor \frac{end - start} {step} \rfloor + 1
 * $的1维张量。其中，步长step是张量中 相邻两个值的间隔。
 *
 * 计算公式：$$ out_{i+1}=out_i+step $$
 *
 * 实现说明：
 * api计算的基本路径：
 * ```mermaid
 * graph LR
 *     A[(Start)]--> E([l0op::Arange])
 *     B[(End)]--> D[(Limit = End + Step)]
 *     C[(Step)]--> D
 *     D--> E
 *     C--> E
 *     E--> H([l0op::Cast])
 *     H--> M([l0op::ViewCopy])
 *     M--> N[(Out)]
 * ```
 *
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnRangeGetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnRange(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif