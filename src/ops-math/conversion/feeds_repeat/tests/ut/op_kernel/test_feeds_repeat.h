/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FEEDS_REPEAT_TILING_H_
#define _FEEDS_REPEAT_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#define __aicore__

struct FeedsRepeatTilingData {
    uint32_t length = 0;
    uint32_t length_aligned = 0;
    int64_t elem_row = 0;
    int64_t elem_per_loop = 0;
    int64_t max_core_num = 0;
    int64_t core_per_group = 0;
    int64_t core_moreover = 0;
    int64_t empty_size = 0;
    int64_t row_per_core = 0;
    int64_t row_left = 0;
};

inline void InitFeedsRepeatTilingData(uint8_t* tiling, FeedsRepeatTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(FeedsRepeatTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    FeedsRepeatTilingData tilingData;              \
    InitFeedsRepeatTilingData(tilingPointer, &tilingData)
#endif