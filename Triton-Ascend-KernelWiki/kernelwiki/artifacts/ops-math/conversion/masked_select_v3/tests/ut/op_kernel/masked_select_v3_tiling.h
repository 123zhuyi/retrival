/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MASKED_SELECT_V3_TILING_H_
#define MASKED_SELECT_V3_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

struct MaskedSelectV3TilingData {
    uint64_t formerNum = 0;
    uint64_t formerLength = 0;
    uint64_t formertileNum = 0;
    uint64_t formertileLength = 0;
    uint64_t formerlasttileLength = 0;
    uint64_t tailNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailtileNum = 0;
    uint64_t tailtileLength = 0;
    uint64_t taillasttileLength = 0;
};

#pragma pack()

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct* tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

#define GET_TILING_DATA(tilingData, tilingPointer)                                \
    MaskedSelectV3TilingData tilingData;                                          \
    INIT_TILING_DATA(MaskedSelectV3TilingData, tilingDataPointer, tilingPointer); \
    (tilingData).formerNum = tilingDataPointer->formerNum;                        \
    (tilingData).formerLength = tilingDataPointer->formerLength;                  \
    (tilingData).formertileNum = tilingDataPointer->formertileNum;                \
    (tilingData).formertileLength = tilingDataPointer->formertileLength;          \
    (tilingData).formerlasttileLength = tilingDataPointer->formerlasttileLength;  \
    (tilingData).tailNum = tilingDataPointer->tailNum;                            \
    (tilingData).tailLength = tilingDataPointer->tailLength;                      \
    (tilingData).tailtileNum = tilingDataPointer->tailtileNum;                    \
    (tilingData).tailtileLength = tilingDataPointer->tailtileLength;              \
    (tilingData).taillasttileLength = tilingDataPointer->taillasttileLength;

#endif // MASKED_SELECT_V3_TILING_H_
