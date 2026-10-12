/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGN_BITS_PACK_TILING_DATA_H_
#define SIGN_BITS_PACK_TILING_DATA_H_

#include <cstdint>

constexpr int64_t kAlignUnit = 256;
constexpr int64_t kPackRate = 8;

struct SignBitsPackTilingData {
    uint8_t rank;
    int64_t inShape[1];
    int64_t outShape[2];
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint32_t bufferSize;

    int64_t n;
    int64_t sizeAttr;
    int64_t packedLen;
    int64_t padCount;
    uint32_t mask;
    uint32_t block;
    uint64_t tailElemCount;
    uint64_t tailByteCount;
    uint32_t realCoreNum;
};

#endif // SIGN_BITS_PACK_TILING_DATA_H_
