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
 * @file aclnn_acos_grad_v2.h
 * @brief ACLNN L2 API 接口声明 - AcosGradV2 算子 (A2 / Ascend910B)
 *
 * z = -dy / sqrt(1 - y^2)
 */

#ifndef ACLNN_ACOS_GRAD_V2_H_
#define ACLNN_ACOS_GRAD_V2_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnAcosGradV2GetWorkspaceSize(const aclTensor* y, const aclTensor* dy, const aclTensor* z,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnAcosGradV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_ACOS_GRAD_V2_H_
