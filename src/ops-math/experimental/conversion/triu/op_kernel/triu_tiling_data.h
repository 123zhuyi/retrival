/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liu Jun <@kbryantttt>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file triu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __TRIU_TILLING_DATA_H__
#define __TRIU_TILLING_DATA_H__

struct TriuTilingData {
    uint32_t totalLengthAligned;
    int32_t matrixNum;
    int32_t matrixSize;
    int32_t rowLength;
    int32_t columnLength;
    int32_t diagVal;
    int32_t loopCnt;
    uint32_t fullTileLength;
    uint32_t lastTileLength;
    int32_t fullCnt;
    int32_t lastCnt;
    uint32_t alignNum;
    uint32_t typeSize;
    // 扩展其他tilling参数
};
#endif
