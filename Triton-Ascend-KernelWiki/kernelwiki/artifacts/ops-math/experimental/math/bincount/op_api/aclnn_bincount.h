/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_BINCOUNT_H_
#define OP_API_INC_BINCOUNT_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnBincount 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_math
 *
 * 算子功能：参考 torch.bincount，计算非负整数数组中每个数的频率。仅支持非负整数（与 torch.bincount 一致）；
 *   输入含负数时在 kernel 运行期检查到后报错中止（ascendc_assert，运行时错误码 507015），不产出结果。
 *   底层算子原型为 GE/TF 风格 array/size/weights/bins；本 aclnn 用户接口对齐 torch.bincount 的
 *   self/weights/minlength/out，内部将 minlength 构造为 size 输入下发给底层算子。
 * 计算公式（idx = value，无偏移）：如果指定了 weights，则
 *     out[self_i] = out[self_i] + weights_i
 *   否则
 *     out[self_i] = out[self_i] + 1
 *
 * @param [in] self: npu device 侧的 aclTensor，数据类型支持 INT8、INT16、INT32、INT64、UINT8，
 *   仅支持非负整数（含负数时运行期报错中止），数据格式支持 1 维 ND，支持非连续的 Tensor。
 * @param [in] weights: npu device 侧的 aclTensor，self 每个值的权重，可为空指针。
 *   数据类型支持 FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL，
 *   数据格式支持 1 维 ND，且 shape 必须与 self 一致，支持非连续的 Tensor。
 * @param [in] minlength: host 侧的 int64_t，指定输出 tensor 的最小长度，要求 >= 0，为负时返回 ACLNN_ERR_PARAM_INVALID。
 * @param [in] out: npu device 侧的 aclTensor，数据类型支持 INT32、INT64、FLOAT、DOUBLE。
 *   数据格式支持 1 维 ND，长度由调用方按 max(max(self) + 1, minlength) 预分配，
 *   输出下标 k 对应原值 k，支持非连续的 Tensor。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor* self, const aclTensor* weights, int64_t minlength,
                                                    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnBincount 的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口 aclnnBincountGetWorkspaceSize 获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_BINCOUNT_H_
