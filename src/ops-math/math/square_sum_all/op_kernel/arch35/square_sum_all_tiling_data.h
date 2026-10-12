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
 * \file square_sum_all_tiling_data.h
 * \brief Tiling ABI shared by the SquareSumAll host and arch35 kernel.
 */

#ifndef SQUARE_SUM_ALL_TILING_DATA_H_
#define SQUARE_SUM_ALL_TILING_DATA_H_

#include <cstdint>

struct SquareSumAllTilingData {
    int64_t totalElements;
    int64_t usedCoreNum;
    int64_t baseCoreElements;
    int64_t extraCoreCount;
    int64_t tileElements;
};

#endif // SQUARE_SUM_ALL_TILING_DATA_H_
