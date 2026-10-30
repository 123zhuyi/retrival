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
 * \file mod_tiling_data.h
 * \brief tiling data struct
 */
#ifndef MOD_TILING_DATA_H
#define MOD_TILING_DATA_H

#include <cstdint>

namespace ModNs {

struct ModTilingData {
    uint32_t usableUbSize;
    uint32_t needCoreNum;
    uint64_t totalDataCount;
    uint64_t perCoreDataCount;
    uint64_t tailDataCoreNum;
    uint64_t lastCoreDataCount;
    bool isInput2Scalar;
    bool isInput2SameShape;
    uint32_t dimNum;
    uint64_t input1Shape[8];
    uint64_t input2Shape[8];
    uint64_t input2Stride[8];
};

} // namespace ModNs

#endif // MOD_TILING_DATA_H
