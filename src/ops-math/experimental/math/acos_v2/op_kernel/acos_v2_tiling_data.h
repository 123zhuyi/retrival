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

#ifndef _ACOS_TILING_DATA_H_
#define _ACOS_TILING_DATA_H_

struct AcosV2TilingData {
    uint32_t totalLength = 0;     // Total number of elements
    uint32_t tileLength = 0;      // Number of elements per UB tile
    uint32_t blockFactor = 0;     // Number of elements per core
    uint32_t blockTailFactor = 0; // Number of elements for the last core
};

#endif // _ACOS_TILING_DATA_H_
