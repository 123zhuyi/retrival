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
 * \file truncate_mod_tiling_data.h
 * \brief TruncateMod tiling data struct (shared between host tiling and device kernel).
 */
#ifndef _TRUNCATEMOD_TILING_DATA_H_
#define _TRUNCATEMOD_TILING_DATA_H_

#include <cstdint>

// schMode (tiling key) selects the compute dtype.
#ifndef TRUNCATEMOD_TPL_SCH_MODE_0
#define TRUNCATEMOD_TPL_SCH_MODE_0 0 // half (float16)
#endif
#ifndef TRUNCATEMOD_TPL_SCH_MODE_1
#define TRUNCATEMOD_TPL_SCH_MODE_1 1 // float (float32)
#endif
#ifndef TRUNCATEMOD_TPL_SCH_MODE_2
#define TRUNCATEMOD_TPL_SCH_MODE_2 2 // bfloat16
#endif
#ifndef TRUNCATEMOD_TPL_SCH_MODE_3
#define TRUNCATEMOD_TPL_SCH_MODE_3 3 // int32
#endif
#ifndef TRUNCATEMOD_TPL_SCH_MODE_4
#define TRUNCATEMOD_TPL_SCH_MODE_4 4 // int8
#endif
#ifndef TRUNCATEMOD_TPL_SCH_MODE_5
#define TRUNCATEMOD_TPL_SCH_MODE_5 5 // uint8
#endif

// Plain-struct tiling data. All *Length / tail* fields are element counts,
// already block aligned. element-wise op -> no reduction / workspace.
struct TruncateModTilingData {
    uint64_t coreNum = 1;
    uint64_t bufferNum = 1;
    uint64_t tailElems = 0;
    uint64_t epochs = 0;
    uint64_t epochsForLastCore = 0;
    uint64_t coreLength = 0;
    uint64_t tileLength = 0;
    uint64_t tailTileLength = 0;
    uint64_t tailTileLengthForLastCore = 0;
};
#endif
