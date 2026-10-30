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
 * \file mul_addn_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MUL_ADDN_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MUL_ADDN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MulAddnTilingData)
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint64_t, shapeB);
TILING_DATA_FIELD_DEF(uint64_t, shapeM);
TILING_DATA_FIELD_DEF(uint64_t, shapeN);
TILING_DATA_FIELD_DEF(uint64_t, shapeNAlign);
TILING_DATA_FIELD_DEF(uint64_t, coreTaskNum);     // 每个核要处理的行数
TILING_DATA_FIELD_DEF(uint64_t, useCoreNum);      // 使用的核数
TILING_DATA_FIELD_DEF(uint64_t, lastCoreTaskNum); // 最后一个核要用的行数

TILING_DATA_FIELD_DEF(uint64_t, mNum);     // 每次计算mNum
TILING_DATA_FIELD_DEF(uint64_t, mLoopNum); // 共计算mLoopNum
TILING_DATA_FIELD_DEF(uint64_t, mNumTail); // 最后一次计算mNumTail

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulAddn, MulAddnTilingData)
} // namespace optiling
#endif