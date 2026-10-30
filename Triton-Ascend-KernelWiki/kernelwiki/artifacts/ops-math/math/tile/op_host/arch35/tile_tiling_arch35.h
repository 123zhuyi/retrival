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
 * \file tile_tiling_arch35.h
 * \brief head file of tile tiling
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_TILE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_TILE_H_
#include "conversion/broadcast_to/op_host/arch35/broadcast_to_tiling_base.h"
#include "register/tilingdata_base.h"

namespace optiling {

REGISTER_TILING_DATA_CLASS(Tile, BroadcastToTilingData);

struct TileCompileInfo {
  int64_t coreNum;
  int64_t ubSize;
  uint32_t clSize;
  uint32_t vRegSize;
  int64_t blockSize;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_TILE_H_
