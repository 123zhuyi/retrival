/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file slice_v3_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __SLICE_V3_TILLING_DATA_H__
#define __SLICE_V3_TILLING_DATA_H__

constexpr uint32_t MAX_DIM = 8;
struct SliceV3TilingData {
    int64_t dim;
    int64_t begin[MAX_DIM];
    int64_t inputCumShape[MAX_DIM];
    int64_t outputCumShape[MAX_DIM];
    int64_t lastDimSize;
    int64_t tileDataNum;
    int64_t outerLoopNum;
    int64_t tailCoreNum;
};
#endif
