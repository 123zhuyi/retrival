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
 * \file tensor_move_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_MOVE_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_MOVE_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TensorMoveTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);           // 实际使用的核数
TILING_DATA_FIELD_DEF(int64_t, blockFactor);           // 单核循环次数
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);       // 尾核循环次数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);              // 单次循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, tailBlockTailUbFactor); // 尾核尾循环要处理的数据大小
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TensorMove, TensorMoveTilingData)

struct TensorMoveTilingParam {
    int64_t totalCoreNum;
    int64_t ubSize;
    int64_t uo;
    int64_t usedCoreNum;
    int64_t bytesForOneData;
    int64_t ubFactor;
    int64_t tailBlockTailUbFactor;
    int64_t blockFactor;
    int64_t tailBlockFactor;
    int64_t tilingKey;
};

struct TensorMoveCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_TENSOR_MOVE_H_
