/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bincount_tiling_data.h
 * \brief bincount_tiling_data.h
 */

#ifndef _BINCOUNT_TILING_DATA_H_
#define _BINCOUNT_TILING_DATA_H_

struct BincountTilingData {
    int64_t size = 0;
    int64_t ubNumCanUse = 0;
    int64_t ubLoopNum = 0;
    int64_t needXCoreNum = 0;
    int64_t formerLength = 0;
    int64_t tailLength = 0;
    int64_t clearYCoreNum = 0;
    int64_t clearYFactor = 0;
    int64_t clearYTail = 0;
    int64_t binsFormerLength = 0;
    int64_t needBinsCoreNum = 0;
    int64_t binsTailLength = 0;
    int64_t arraySize = 0;
};
#endif
