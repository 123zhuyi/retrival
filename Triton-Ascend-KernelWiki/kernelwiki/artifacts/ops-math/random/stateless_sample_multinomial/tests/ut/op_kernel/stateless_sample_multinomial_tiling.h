/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STATELESS_SAMPLE_MULTINOMIAL_TILING_H
#define STATELESS_SAMPLE_MULTINOMIAL_TILING_H

#include <cstdint>
#include <cstring>

#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"
#include "kernel_tiling/kernel_tiling.h"

static inline unsigned long long __mul_i32toi64(unsigned int lhs, unsigned int rhs)
{
    return static_cast<unsigned long long>(lhs) * rhs;
}

#define __aicore__
#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, RandomUnifiedSimtTilingDataStruct* constData)
{
    const __gm__ uint32_t* src = reinterpret_cast<const __gm__ uint32_t*>(tiling);
    uint32_t* dst = reinterpret_cast<uint32_t*>(constData);
    for (size_t i = 0; i < sizeof(RandomUnifiedSimtTilingDataStruct) / sizeof(uint32_t); ++i) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, RandomUnifiedSimtTilingDataStruct* constData)
{
    std::memcpy(constData, tiling, sizeof(RandomUnifiedSimtTilingDataStruct));
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>(reinterpret_cast<__ubuf__ uint8_t*>(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg)    \
    RandomUnifiedSimtTilingDataStruct tilingData; \
    InitTilingData(tilingArg, &tilingData)

#endif // STATELESS_SAMPLE_MULTINOMIAL_TILING_H