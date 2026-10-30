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
 * \file stack_ball_query_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_BALL_QUERY_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_BALL_QUERY_H_

#include "register/tilingdata_base.h"

namespace optiling {

struct StackBallQueryCompileInfo {
    uint32_t aicore_num = 0;
    int64_t ub_platform_byte_size = 0;
};

BEGIN_TILING_DATA_DEF(StackBallQueryTilingData)
TILING_DATA_FIELD_DEF(int32_t, batchSize);
TILING_DATA_FIELD_DEF(int32_t, totalLengthCenterXyz);
TILING_DATA_FIELD_DEF(int32_t, totalLengthXyz);
TILING_DATA_FIELD_DEF(int32_t, totalIdxLength);
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, centerXyzPerCore);
TILING_DATA_FIELD_DEF(int32_t, tailCenterXyzPerCore);
TILING_DATA_FIELD_DEF(float, maxRadius);
TILING_DATA_FIELD_DEF(int32_t, sampleNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StackBallQuery, StackBallQueryTilingData)
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_STACK_BALL_QUERY_H_
