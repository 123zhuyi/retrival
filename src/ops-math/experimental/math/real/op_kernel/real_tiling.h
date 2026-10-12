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
 * \file real_tiling.h
 * \brief real tiling h
 */

#ifndef REAL_TILING_H
#define REAL_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

constexpr int64_t REAL_BLOCK_SIZE = 32;
constexpr int32_t REAL_BUFFER_NUM = 2;
constexpr int64_t RESERVED_UB_SIZE = 256;
constexpr size_t RESERVED_WORKSPACE = static_cast<size_t>(16 * 1024 * 1024);

struct RealTilingData {
    int64_t totalUsedCoreNum = 0;
    int64_t tailBlockNum = 0;
    int64_t ubPartDataNum = 0;
    int64_t smallCoreDataNum = 0;
    int64_t smallCoreLoopNum = 0;
    int64_t smallCoreTailDataNum = 0;
    int64_t bigCoreDataNum = 0;
    int64_t bigCoreLoopNum = 0;
    int64_t bigCoreTailDataNum = 0;
    int64_t tilingKey = 0;
    int64_t useNonInplace = 0;
};

struct RealTilingParam {
    int64_t totalCoreNum;
    int64_t totalUbSize;
    int64_t totalLength;
    int64_t dataTypeLength;
    bool isComplexInput;
    int64_t totalUsedCoreNum;
    int64_t tailBlockNum;
    int64_t ubPartDataNum;
    int64_t smallCoreDataNum;
    int64_t smallCoreLoopNum;
    int64_t smallCoreTailDataNum;
    int64_t bigCoreDataNum;
    int64_t bigCoreLoopNum;
    int64_t bigCoreTailDataNum;
    int64_t tilingKey;
    int64_t useNonInplace;
};

struct RealCompileInfo {
    int32_t totalCoreNum = 30;
    uint64_t ubSizePlatForm = 0;
};

enum class RealTilingKey : int64_t {
    TILINGKEY_COMPLEX32 = 1,
    TILINGKEY_COMPLEX64 = 2,
    TILINGKEY_COMPLEX128 = 3,
    TILINGKEY_FLOAT16 = 4,
    TILINGKEY_FLOAT = 5,
};

namespace optiling {

constexpr int64_t REAL_BLOCK_SIZE = 32;
constexpr int32_t REAL_BUFFER_NUM = 2;
constexpr int64_t RESERVED_UB_SIZE = 256;
constexpr size_t RESERVED_WORKSPACE = static_cast<size_t>(16 * 1024 * 1024);

} // namespace optiling
#endif // REAL_TILING_H
