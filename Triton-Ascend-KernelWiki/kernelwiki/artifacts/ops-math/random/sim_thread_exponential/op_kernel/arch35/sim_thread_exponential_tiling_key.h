/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sim_thread_exponential_tiling_key.h
 * \brief Tiling key enum for sim_thread_exponential operator
 */
#ifndef SIM_THREAD_EXPONENTIAL_TILING_KEY_H
#define SIM_THREAD_EXPONENTIAL_TILING_KEY_H

#include <cstdint>

enum class SimThreadExponentialTilingKey : uint32_t {
    TILING_KEY_FP16 = 1,
    TILING_KEY_BF16 = 2,
    TILING_KEY_FP32 = 3,
};

#endif // SIM_THREAD_EXPONENTIAL_TILING_KEY_H
