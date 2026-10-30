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
 * \file concat_dv2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_DV2_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONCAT_DV2_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
const uint32_t TILING_ARRAY_LENGTH = 48;
const uint32_t UB_BLOCK_SIZE = 32;
const uint32_t TILING_KEY = 0;

BEGIN_TILING_DATA_DEF(ConcatDV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, elePerLoop);
TILING_DATA_FIELD_DEF(int64_t, elePercore);
TILING_DATA_FIELD_DEF(int64_t, ubLoop);

TILING_DATA_FIELD_DEF(int64_t, eleTailCore);
TILING_DATA_FIELD_DEF(int64_t, ubLoopTail);

TILING_DATA_FIELD_DEF(int64_t, sameDimSize);

TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LENGTH, endTensorIdx);
TILING_DATA_FIELD_DEF_ARR(int64_t, TILING_ARRAY_LENGTH, endTensorOffset);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConcatDV2, ConcatDV2TilingData)

struct Tiling4ConcatDV2CompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
    uint32_t sysWorkspaceSize;
};

struct ConcatDV2Tiling {
    uint32_t elePerLoop = 0;
    int64_t elePercore = 0;
    int64_t ubLoop = 0;
    int64_t eleTailCore = 0;
    int64_t ubLoopTail = 0;
    int64_t usedCoreNum = 0;
    uint32_t dtypeSize = 0;
};

}// namespace optiling
#endif