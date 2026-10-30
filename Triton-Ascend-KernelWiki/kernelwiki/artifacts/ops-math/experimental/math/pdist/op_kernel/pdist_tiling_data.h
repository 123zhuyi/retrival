/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PDIST_TILING_DATA_H
#define PDIST_TILING_DATA_H

struct PdistTilingData {
    uint32_t rows = 0;
    uint32_t cols = 0;
    float pValue = 0.0f;
    uint64_t computeNum = 0;
    uint32_t ubTensorEachLoop = 0;
    uint32_t coreNumVar = 0;
    uint32_t tilingKey = 1;
    uint32_t reduceBufSize = 0;
    uint64_t numBlockEachCore = 0;
    uint64_t lastNumsBlocks = 0;
    uint64_t lastNumsNoneFullBlock = 0;
};

#endif
