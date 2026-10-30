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
 * \file top_k_v2_tiling_arch35.h
 * \brief top_k_v2 ac tiling impl
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_TOP_K_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_TOP_K_V2_H

#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TopKV2TilingDataSimd)
    // TopK field
    TILING_DATA_FIELD_DEF(int32_t, isLargest);
    TILING_DATA_FIELD_DEF(int32_t, isSort);
    TILING_DATA_FIELD_DEF(uint64_t, sortLoopTimes);
    TILING_DATA_FIELD_DEF(uint32_t, unsortedDimParallel);
    TILING_DATA_FIELD_DEF(uint64_t, unsortedDimNum);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimNeedCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTileDataSize);
    TILING_DATA_FIELD_DEF(uint32_t, platformCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, topkAcApiTmpBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, mergSortAcApiNeedBufferSize);
    TILING_DATA_FIELD_DEF(uint64_t, oneCoreRowNum);
    TILING_DATA_FIELD_DEF(uint32_t, batchNumInUb);
    TILING_DATA_FIELD_DEF(uint32_t, tailLoopBatchNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailBatchNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, modeType);
    TILING_DATA_FIELD_DEF(uint32_t, isInInt32Range);
    TILING_DATA_FIELD_DEF(int64_t, lastAxisNum);
    TILING_DATA_FIELD_DEF(int64_t, topKRealValue);
    TILING_DATA_FIELD_DEF(int64_t, lastDimTileNum);
    TILING_DATA_FIELD_DEF(int64_t, outputLastDimValue);
    TILING_DATA_FIELD_DEF(int64_t, lastDimTileNumTimes);
    // sort field
    TILING_DATA_FIELD_DEF(uint32_t, tilingKeyForSort);
    TILING_DATA_FIELD_DEF(uint32_t, sortLoopTimesForSort);
    TILING_DATA_FIELD_DEF(uint32_t, unsortedDimParallelForSort);
    TILING_DATA_FIELD_DEF(uint64_t, unsortedDimNumForSort);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimTileNumForSort);
    TILING_DATA_FIELD_DEF(uint32_t, lastDimNeedCoreForSort);
    TILING_DATA_FIELD_DEF(uint32_t, numTileDataSizeForSort);
    TILING_DATA_FIELD_DEF(uint32_t, sortAcApiNeedBufferSizeForSort);
    TILING_DATA_FIELD_DEF(uint32_t, mergSortAcApiNeedBufferSizeForSort);
    TILING_DATA_FIELD_DEF(uint32_t, oneCoreRowNumForSort);
    TILING_DATA_FIELD_DEF(uint32_t, outputLastDimValueForSort);
    TILING_DATA_FIELD_DEF(uint32_t, isInInt32RangeForSort);
    TILING_DATA_FIELD_DEF(int64_t, lastAxisNumForSort);
    TILING_DATA_FIELD_DEF(uint32_t, modeTypeForSort);    
    TILING_DATA_FIELD_DEF(uint64_t, keyParams0);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams1);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams2);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams3);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams4);
    TILING_DATA_FIELD_DEF(uint32_t, keyParams5);
    TILING_DATA_FIELD_DEF(uint32_t, tmpUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, sortAndTopkTileDataSize);
    TILING_DATA_FIELD_DEF(uint32_t, sortAndTopkBlockTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, sortAndTopkTailTileNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TopKV2, TopKV2TilingDataSimd)
struct TopKV2CompileInfo {
  int32_t coreNum;
};
}
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_TOP_K_V2_H
