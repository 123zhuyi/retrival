/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_fused_mul_add_n.h
 * @brief ACLNN L2 API 接口声明 - FusedMulAddN (y = x1 * x3[0] + x2)
 *
 * ACLNN 接口采用两段式设计：
 * - GetWorkspaceSize: 计算 workspace 大小，创建执行器
 * - aclnnFusedMulAddN: 执行计算
 *
 * 文件命名规范：
 * - L2 API: aclnn_{op}.h/cpp
 * - L0 API: {op}.h/cpp
 */

#ifndef OP_API_INC_ACLNN_FUSED_MUL_ADD_N_H_
#define OP_API_INC_ACLNN_FUSED_MUL_ADD_N_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedMulAddN 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_math
 *
 * 算子功能：逐元素融合标量乘加。
 * 计算公式：
 *   $$ y_i = x1_i \times x3[0] + x2_i $$
 *
 * @param [in] x1: npu device 侧的 aclTensor，主张量，被标量 x3[0] 乘。
 * 数据类型支持 FLOAT、FLOAT16、BFLOAT16、INT32、INT16，需要与 x2/x3/y 一致；
 * 数据格式支持 ND；shape 需要与 x2/y 一致。
 * @param [in] x2: npu device 侧的 aclTensor，逐元素加到 x1 * x3[0] 上。
 * 数据类型、shape、format 需要与 x1 一致。
 * @param [in] x3: npu device 侧的 aclTensor，单元素标量张量（ShapeSize = 1），仅取 x3[0] 作为标量乘数。
 * 数据类型需要与 x1 一致；数据格式支持 ND。
 * @param [out] y: npu device 侧的 aclTensor，输出，与 x1 同 dtype、同 shape；数据格式支持 ND。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnFusedMulAddNGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x3, aclTensor* y, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief aclnnFusedMulAddN 的第二段接口，用于执行计算。
 *
 * 算子功能：逐元素融合标量乘加。
 * 计算公式：
 *   $$ y_i = x1_i \times x3[0] + x2_i $$
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口
 *  aclnnFusedMulAddNGetWorkspaceSize 获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnFusedMulAddN(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_FUSED_MUL_ADD_N_H_
