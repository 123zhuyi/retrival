/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_set_pytorch_random.h"

#include <atomic>

#include "opdev/op_log.h"

namespace {
// 默认兼容A2/A3
std::atomic<int32_t> g_pytorchRandomMode{0};

constexpr int32_t PYTORCH_RANDOM_MODE_COMPAT = 0;  // 生成随机数兼容A2/A3
constexpr int32_t PYTORCH_RANDOM_MODE_PYTORCH = 1; // 对标pytorch
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnSetPytorchRandom(int32_t pytorchRandom)
{
    if (pytorchRandom != PYTORCH_RANDOM_MODE_COMPAT && pytorchRandom != PYTORCH_RANDOM_MODE_PYTORCH) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pytorchRandom must be 0 or 1, but got %d.", pytorchRandom);
        return ACLNN_ERR_PARAM_INVALID;
    }
    g_pytorchRandomMode.store(pytorchRandom, std::memory_order_relaxed);
    return ACLNN_SUCCESS;
}

ACLNN_API int32_t aclnnGetPytorchRandom() { return g_pytorchRandomMode.load(std::memory_order_relaxed); }

#ifdef __cplusplus
}
#endif
