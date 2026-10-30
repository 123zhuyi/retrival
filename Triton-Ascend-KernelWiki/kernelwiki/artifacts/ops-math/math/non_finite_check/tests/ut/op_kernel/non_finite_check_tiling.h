/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __NON_FINITE_CHECK_TILING_H__
#define __NON_FINITE_CHECK_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

struct NonFiniteCheckCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

enum class NonFiniteCheckTilingKey : uint64_t { KEY_FLOAT16 = 101, KEY_BF16 = 201, KEY_FLOAT = 301 };

constexpr uint16_t MAX_TENSOR_COUNT = 256;
constexpr uint16_t MAX_CORE_COUNT = 64;

#pragma pack(1)

struct NonFiniteCheckTilingData {
    uint32_t maxProcCount = 0;
    uint32_t tempValUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_COUNT] = {};
    uint16_t tensorStartList[MAX_CORE_COUNT] = {};
    uint16_t tensorEndList[MAX_CORE_COUNT] = {};
    int64_t tensorStartOffsetList[MAX_CORE_COUNT] = {};
    int64_t tensorEndOffsetList[MAX_CORE_COUNT] = {};
};

#pragma pack()

inline void InitNonFiniteCheckTilingData(uint8_t* tiling, NonFiniteCheckTilingData* const_data) {
    memcpy(const_data, tiling, sizeof(NonFiniteCheckTilingData));
}

#undef GET_TILING_DATA
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    NonFiniteCheckTilingData tiling_data;        \
    InitNonFiniteCheckTilingData(tiling_arg, &tiling_data)

#endif  // __NON_FINITE_CHECK_TILING_H__