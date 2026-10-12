/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_V3_TILING_DATA_ARCH35_H
#define DROP_OUT_V3_TILING_DATA_ARCH35_H

#include <sstream>

class DropOutV3TilingDataStruct {
public:
    int64_t usedCoreNum = 0;
    int64_t outputSize = 0;
    int64_t seed = 0;
    int64_t offset = 0;
    int64_t ubSize = 0;
    float prob = 0;

    uint32_t vec = 0;
    uint32_t transportMode = 0;
    int64_t totalThreads = 0;
    int64_t perCoreElements = 0;
    int64_t tailCoreElements = 0;
    int64_t ubLoopCount = 0;
    int64_t tailUbLoopCount = 0;
    int64_t ubFactorElements = 0;
    int64_t tailUbFactorElements = 0;
    int64_t tailCoreTailUbFactorElements = 0;

    std::string DumpTilingInfo() const
    {
        std::ostringstream info;
        info << "[DropOutV3TilingData] "
             << "usedCoreNum: " << usedCoreNum << ", outputSize: " << outputSize << ", seed: " << seed
             << ", offset: " << offset << ", ubSize: " << ubSize << ", prob: " << prob << ", vec: " << vec
             << ", transportMode: " << transportMode << ", totalThreads: " << totalThreads
             << ", perCoreElements: " << perCoreElements << ", tailCoreElements: " << tailCoreElements
             << ", ubLoopCount: " << ubLoopCount << ", tailUbLoopCount: " << tailUbLoopCount
             << ", ubFactorElements: " << ubFactorElements << ", tailUbFactorElements: " << tailUbFactorElements
             << ", tailCoreTailUbFactorElements: " << tailCoreTailUbFactorElements;
        return info.str();
    }
};

#endif // DROP_OUT_V3_TILING_DATA_ARCH35_H
