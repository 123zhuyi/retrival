/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Disclaimer: This file is generated with the assistance of an AI tool.
 * Please review carefully before use.
 */

#ifndef _FRESNEL_COS_TILING_DATA_H_
#define _FRESNEL_COS_TILING_DATA_H_

struct FresnelCosTilingData {
    uint64_t totalLength = 0; // total number of elements
    uint32_t blockNum = 0;    // actual AI Core count (dynamic)
    uint32_t baseLength = 0;  // elements per core (first blockNum-1 cores)
    uint32_t tailLength = 0;  // elements for the tail core
    uint32_t tileLength = 0;  // elements per tile (Double Buffer chunk)
    uint32_t tileNum = 0;     // number of main-loop tiles per core
    uint32_t tailTileLen = 0; // tail tile elements within a core (< tileLength)
};

#endif // _FRESNEL_COS_TILING_DATA_H_
