/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/**
 * @file aclnn_view_copy.h
 * @brief ACLNN L2 API 接口声明
 */

#ifndef ACLNN_VIEWCOPY_H_
#define ACLNN_VIEWCOPY_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnViewCopyGetWorkspaceSize(const aclTensor* dst, const aclTensor* dst_size,
                                                    const aclTensor* dst_stride, const aclTensor* dst_storage_offset,
                                                    const aclTensor* src, const aclTensor* src_size,
                                                    const aclTensor* src_stride, const aclTensor* src_storage_offset,
                                                    const aclTensor* dst_out, uint64_t* workspaceSize,
                                                    aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnViewCopy(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_VIEWCOPY_H_
