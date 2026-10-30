/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_bias_add_grad_tiling_base.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_TILING_BASE_H
#define GROUPED_BIAS_ADD_GRAD_TILING_BASE_H

#include <cstdint>

namespace optiling {
namespace groupedbiasaddgrad {
// 公共变量区
constexpr uint32_t GRAD_Y_INPUT_INDEX = 0;
constexpr uint32_t GROUP_IDX_INPUT_INDEX = 1;
constexpr int64_t INPUT_MAX_GROUP = 2048;
constexpr int64_t H_BASE_SIZE = 512;
constexpr int64_t ACTIVE_NODES_NUM = 3;
constexpr int64_t BUFFER_NUM = 2;

constexpr uint32_t RESERVED_UB_SIZE = 8 * 1024; // ub固定预留8k
constexpr int64_t UB_GROUP_SUM_NUM = 8;         // 少于8组在ub内分组累加，否则在workspace

constexpr int64_t WORKSPACE_BASE_CAL = 32 * 1024 * 1024; // 系统预留
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t BLOCK_NUM = BLOCK_SIZE / sizeof(int32_t);
constexpr int64_t PERF_G_NUM = 200;
constexpr int64_t TWO_NUM = 2;
constexpr int64_t THREE_NUM = 3;

enum class DtypeEnum : uint32_t
{
    FLOAT16 = 0,
    FLOAT32 = 1,
    BFLOAT16 = 2,
};

} // namespace groupedbiasaddgrad
} // namespace optiling

#endif // GROUPED_BIAS_ADD_GRAD_TILING_BASE_H