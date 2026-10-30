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
 * \file sort_with_index_tiling.h
 * \brief sort_with_index ac tiling
 */
#ifndef SORT_WITH_INDEX_TILING_H
#define SORT_WITH_INDEX_TILING_H
#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SortWithIndexTilingDataSimt)
    TILING_DATA_FIELD_DEF(int32_t, isDescend);
    TILING_DATA_FIELD_DEF(uint32_t, sortLoopTimes);
    TILING_DATA_FIELD_DEF(uint32_t, unsortedDimParallel);
    TILING_DATA_FIELD_DEF(uint64_t, unsortedDimNum);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimNeedCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTileDataSize);
    TILING_DATA_FIELD_DEF(uint32_t, sortAcApiNeedBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, mergSortAcApiNeedBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, oneCoreRowNum);
    TILING_DATA_FIELD_DEF(uint32_t, outputLastDimValue);
    TILING_DATA_FIELD_DEF(uint32_t, isInInt32Range);
    TILING_DATA_FIELD_DEF(int64_t, lastAxisNum);
    // adatp sort add 
    TILING_DATA_FIELD_DEF(uint32_t, keyParams0);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams1);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams2);    
    TILING_DATA_FIELD_DEF(uint32_t, keyParams3);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams4);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams5);
    TILING_DATA_FIELD_DEF(uint32_t, tmpUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, modeType);           
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SortWithIndex, SortWithIndexTilingDataSimt)

struct SortWithIndexCompileInfo {
  int32_t core_num;
  int32_t num_block;
};

ge::graphStatus  SortWithIndexTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);
}
#endif