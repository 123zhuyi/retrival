/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pow2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __POW2_TILLING_DATA_H__
#define __POW2_TILLING_DATA_H__

struct Pow2TilingData {
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailBlockNum;

    bool is_input0_scalar;
    bool is_input1_scalar;
    uint32_t yDim;
    bool isSameX1;
    bool isSameX2;
    uint32_t strideX1[10];
    uint32_t strideX2[10];
    uint32_t strideY[10];
    uint32_t X2TotalNum;
    uint32_t X1TotalNum;
};
#endif
