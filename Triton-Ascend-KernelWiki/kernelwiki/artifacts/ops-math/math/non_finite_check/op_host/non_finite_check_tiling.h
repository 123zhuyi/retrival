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
 * \file non_finite_check_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint16_t MAX_TENSOR_COUNT = 256;
constexpr uint16_t MAX_CORE_COUNT = 64;

struct NonFiniteCheckCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

enum class NonFiniteCheckTilingKey : uint64_t
{
    KEY_FLOAT16 = 101,
    KEY_BF16 = 201,
    KEY_FLOAT = 301
};

BEGIN_TILING_DATA_DEF(NonFiniteCheckTilingData)
TILING_DATA_FIELD_DEF(uint32_t, maxProcCount);
TILING_DATA_FIELD_DEF(uint32_t, tempValUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_COUNT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_COUNT, tensorEndList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorStartOffsetList);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_CORE_COUNT, tensorEndOffsetList);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonFiniteCheck, NonFiniteCheckTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_NON_FINITE_CHECK_TILING_H
