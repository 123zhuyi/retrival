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
 * \file acos_tiling.h
 * \brief Static tiling definition for the Acos kernel UT.
 */

#ifndef ACOS_TILING_UT_H_
#define ACOS_TILING_UT_H_

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/acos_tiling_data.h"
#include "kernel_tiling/kernel_tiling.h"

inline void InitTilingData(const uint8_t* tiling, AcosTilingData* tilingData)
{
    std::memcpy(tilingData, tiling, sizeof(AcosTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                             \
    InitTilingData(tilingArg, &tilingData)

#endif // ACOS_TILING_UT_H_
