/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BERNOULLI_MASK_TILING_KEY_H
#define BERNOULLI_MASK_TILING_KEY_H

#include <cstdint>

namespace BernoulliMaskKey {
constexpr uint64_t FLOAT16 = 1;
constexpr uint64_t FLOAT = 2;
constexpr uint64_t DOUBLE = 3;
constexpr uint64_t UINT8_OR_BOOL = 4;
constexpr uint64_t INT8 = 5;
constexpr uint64_t INT16 = 6;
constexpr uint64_t INT32 = 7;
constexpr uint64_t INT64 = 8;
constexpr uint64_t BFLOAT16 = 9;
} // namespace BernoulliMaskKey

#endif // BERNOULLI_MASK_TILING_KEY_H
