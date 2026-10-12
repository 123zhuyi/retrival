/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _HISTOGRAM_FIXED_WIDTH_TILINGDATA_
#define _HISTOGRAM_FIXED_WIDTH_TILINGDATA_

#include <cstdint>

struct HistogramFixedWidthSimtTilingData {
    int32_t bins;
    uint32_t ubNumCanUse;
    uint32_t ubLoopNum;
    uint32_t needXCoreNum;
    int64_t formerLength;
    int64_t tailLength;
    uint32_t clearYCoreNum;
    int64_t clearYFactor;
    int64_t clearYTail;
    uint32_t needCoreNum;
};

#endif // _HISTOGRAM_FIXED_WIDTH_TILINGDATA_
