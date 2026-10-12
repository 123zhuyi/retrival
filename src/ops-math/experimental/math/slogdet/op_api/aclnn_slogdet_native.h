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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef OP_API_INC_SLOGDET_NATIVE_H_
#define OP_API_INC_SLOGDET_NATIVE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSlogdet 的第一段接口，根据具体的计算流程，计算 workspace 大小。
 * @domain aclnn_math
 *
 * 算子功能：对输入方阵 batch 计算行列式的符号 signOut = sign(det(self))
 *           与绝对值的自然对数 logOut = log(|det(self)|)。对标 torch.linalg.slogdet。
 *
 * @param [in] self: npu device 侧 aclTensor, 数据类型支持 FLOAT(fp32). shape 满足 (*, n, n) 形式,
 *                    其中 `*` 表示 0 或更多维度 batch. 支持非连续 Tensor. 数据格式支持 ND.
 * @param [out] signOut: npu device 侧 aclTensor, 数据类型 FLOAT. shape 与 self 的 batch 一致.
 * @param [out] logOut:  npu device 侧 aclTensor, 数据类型 FLOAT. shape 与 self 的 batch 一致.
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小.
 * @param [out] executor: 返回 op 执行器，包含算子计算流程.
 * @return aclnnStatus: 成功返回 ACLNN_SUCCESS, 失败返回对应错误码.
 */
ACLNN_API aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor* self, aclTensor* signOut, aclTensor* logOut,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnSlogdet 的第二段接口，用于执行计算.
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址.
 * @param [in] workspaceSize: workspace 大小，由第一段接口获取.
 * @param [in] executor: op 执行器.
 * @param [in] stream: acl stream 流.
 * @return aclnnStatus: 返回状态码.
 */
ACLNN_API aclnnStatus aclnnSlogdet(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_SLOGDET_NATIVE_H_
