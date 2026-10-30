/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file population_count_tiling_data.h
 * \brief Standard C++ struct TilingData (per DESIGN.md §3.2)
 *
 * NOTE: Must be a plain-old-data (POD) struct, NOT BEGIN_TILING_DATA_DEF.
 */

#ifndef POPULATION_COUNT_TILING_DATA_H_
#define POPULATION_COUNT_TILING_DATA_H_

struct PopulationCountTilingData {
    int64_t totalNum    = 0;   // Total element count of x
    int64_t blockFactor = 0;   // Element count per AI core (aligned to ubBlockSize)
    int64_t ubFactor    = 0;   // Element count per UB-loop tile
};
#endif // POPULATION_COUNT_TILING_DATA_H_
