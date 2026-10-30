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
 * \file chunk_cat_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CHUNK_CAT_TILING_DATA_H_
#define _CHUNK_CAT_TILING_DATA_H_

struct ChunkCatTilingData {
    bool isAllAlign;
    bool isHalfAlign;
    bool isOneConcat;
    int64_t inputNum;
    int64_t dim;
    int64_t numChunk;
    int64_t outputRow;
    int64_t outputCol;
    int64_t inUbSize;
    int64_t outUbSize;
    int64_t blockRowNum;
    int64_t blockColNum;
    int64_t ubRowFactor;
    int64_t ubColFactor;
    int64_t blockRowFactor;
    int64_t blockColFactor;
    int64_t tailBlockRowFactor;
    int64_t tailBlockColFactor;
};
#endif
