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
 * \file floor_mod_tiling.h
 * \brief
 */

#ifndef _FLOOR_MOD_TILING_H_
#define _FLOOR_MOD_TILING_H_

#include <cstdint>
#include <cstring>
#include <securec.h>
#include "../../../op_kernel/floor_mod_tiling_key.h"

using namespace FloorModNs;

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, FloorModTilingData* constData)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)constData;
    for (size_t i = 0; i < sizeof(FloorModTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, FloorModTilingData* constData)
{
    if (constData != nullptr && tiling != nullptr) {
        memcpy_s(constData, sizeof(FloorModTilingData), tiling, sizeof(FloorModTilingData));
    }
}
#endif // __NPU_TILING__

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer))

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    do {                                                                 \
        tilingStruct tilingData;                                         \
        InitTilingData(tilingArg, &tilingData);                          \
    } while (0)

#define GET_TILING_DATA(tilingData, tilingArg)  \
    do {                                        \
        FloorModTilingData tilingData;          \
        InitTilingData(tilingArg, &tilingData); \
    } while (0)

#endif // _FLOOR_MOD_TILING_H_