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
 * \file fill_diagonal_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_DIAGONAL_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_DIAGONAL_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FillDiagonalV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalLength);
TILING_DATA_FIELD_DEF(uint64_t, step);
TILING_DATA_FIELD_DEF(uint64_t, end);
TILING_DATA_FIELD_DEF(uint64_t, ubSize);
TILING_DATA_FIELD_DEF(uint64_t, blockLength);
TILING_DATA_FIELD_DEF(uint64_t, lastBlockLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FillDiagonalV2, FillDiagonalV2TilingData);
struct FillDiagonalV2CompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    uint64_t sysWorkspaceSize = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_FILL_DIAGONAL_H