/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "platform/platform_ascendc.h"
#include "op_host/util/fp16.h"
#include "util/math_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinSpaceTilingData)
TILING_DATA_FIELD_DEF(float, start);
TILING_DATA_FIELD_DEF(float, stop);
TILING_DATA_FIELD_DEF(float, scalar);
TILING_DATA_FIELD_DEF(int64_t, num);
TILING_DATA_FIELD_DEF(int64_t, matrixLen);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, innerTailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerTailLoopTail);
TILING_DATA_FIELD_DEF(int64_t, outerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, outerLoopNumTail);
TILING_DATA_FIELD_DEF(int64_t, outerTailLoopNum);
TILING_DATA_FIELD_DEF(int64_t, outerTailLoopNumTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinSpace, LinSpaceTilingData)

struct LinSpaceCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
    NpuArch npuArch = NpuArch::DAV_2201;
};

enum class LinSpaceTilingKey : int64_t
{
    TILINGKEY_101 = 101,
    TILINGKEY_102 = 102,
    TILINGKEY_103 = 103,
    TILINGKEY_201 = 201,
    TILINGKEY_202 = 202,
    TILINGKEY_203 = 203,
    TILINGKEY_301 = 301,
    TILINGKEY_302 = 302,
    TILINGKEY_303 = 303,
    TILINGKEY_401 = 401,
    TILINGKEY_402 = 402,
    TILINGKEY_403 = 403,
    TILINGKEY_501 = 501,
    TILINGKEY_502 = 502,
    TILINGKEY_503 = 503,
    TILINGKEY_601 = 601,
    TILINGKEY_602 = 602,
    TILINGKEY_603 = 603,
    TILINGKEY_701 = 701,
    TILINGKEY_702 = 702,
    TILINGKEY_703 = 703
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_LINSPACE_H_
