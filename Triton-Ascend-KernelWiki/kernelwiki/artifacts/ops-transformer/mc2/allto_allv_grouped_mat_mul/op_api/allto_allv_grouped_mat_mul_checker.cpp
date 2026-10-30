/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>

#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include "allto_allv_grouped_mat_mul_checker.h"

namespace Mc2AlltoAllvGMMChecker {

using namespace op;

static constexpr int64_t ZERO = 0;
static constexpr size_t MAX_GROUP_LEN = 128U;

aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts, const aclTensor *gmmX, const aclTensor *gmmY)
{
    if (sendCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;
    uint64_t sendSize = 0U;
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t sendSum = 0U;
    uint64_t recvSum = 0U;
    for(uint64_t i = 0; i < recvSize; i++){
        recvSum += (*recvCounts)[i];
    }
    for(uint64_t i = 0; i < sendSize; i++){
        sendSum += (*sendCounts)[i];
    }
    if((sendSum != (gmmX->GetViewShape().GetDim(0)))){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendSum should be BSK.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if((recvSum != (gmmY->GetViewShape().GetDim(0)))){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvSum should be A.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// 检查通信域名的字符串长度是否符合要求
bool CheckGroup(const char *group)
{
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Group should not be nullptr.");
        return false;
    }
    auto len = strnlen(group, MAX_GROUP_LEN);
    if ((len >= MAX_GROUP_LEN) || (len == ZERO)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Required group name length in range (0, 128), but it is %zu.", len);
        return false;
    }
    return true;
}

} // namespace allto_allv_grouped_mat_mul_checker