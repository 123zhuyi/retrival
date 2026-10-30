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
 * \file sinkhorn.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SinkhornTilingData)
TILING_DATA_FIELD_DEF(uint64_t, formerNum);    // former 数量
TILING_DATA_FIELD_DEF(uint64_t, formerRow);    // former cost行数
TILING_DATA_FIELD_DEF(uint64_t, formerLength); // former cost总长

TILING_DATA_FIELD_DEF(uint64_t, formerTileNum);        // former Tile数量
TILING_DATA_FIELD_DEF(uint64_t, formerLastTileRow);    // fomer last Tile行数
TILING_DATA_FIELD_DEF(uint64_t, formerLastTileLength); // fomer last Tile长度

TILING_DATA_FIELD_DEF(uint64_t, tailNum);    // tail 数量
TILING_DATA_FIELD_DEF(uint64_t, tailRow);    // tail cost行数
TILING_DATA_FIELD_DEF(uint64_t, tailLength); // tail cost总长

TILING_DATA_FIELD_DEF(uint64_t, tailTileNum);        // tail Tile数量
TILING_DATA_FIELD_DEF(uint64_t, tailLastTileRow);    // tail last Tile行数
TILING_DATA_FIELD_DEF(uint64_t, tailLastTileLength); // tail last Tile长度

TILING_DATA_FIELD_DEF(uint64_t, tileRow);    // Tile行数(非Last)
TILING_DATA_FIELD_DEF(uint64_t, tileLength); // Tile长度(非Last)

TILING_DATA_FIELD_DEF(uint64_t, totalRow);        // 总行数
TILING_DATA_FIELD_DEF(uint64_t, totalCol);        // 总列数
TILING_DATA_FIELD_DEF(uint64_t, totalColAligned); // 对齐后的总列数

TILING_DATA_FIELD_DEF(float, tol); // 误差
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Sinkhorn, SinkhornTilingData)
struct SinkhornCompileInfo {
    uint64_t aivNum = 40;                         // AIV核数
    uint64_t sysWorkspaceSize = 16 * 1024 * 1024; // 系统WorkSpace大小
    uint64_t ubSize = 196608;                     // UB大小
};
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SINKHORN_H